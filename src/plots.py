import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation import TOLERANCE_FRACTION, prompt_type_sort_key  # noqa: E402


PROMPT_LABEL_ORDER = {
    "none": 0,
    "$E_u$": 1,
    "$A_u$": 2,
    "$E_u{+}A_u$": 3,
    "$D_u$": 4,
    "$D_u{+}E_u$": 5,
    "$D_u{+}A_u$": 6,
    "$D_u{+}E_u{+}A_u$": 7,
}

# Okabe-Ito colorblind-safe palette; first 5 entries cover our 5 domains.
DOMAIN_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]
DOMAIN_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.pad": 1.0,
            "ytick.major.pad": 2.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def abbreviate_prompt_type(prompt_type: str) -> str:
    return (
        str(prompt_type)
        .replace("demographics", "$D_u$")
        .replace("previous_experiences", "$E_u$")
        .replace("privacy_attitudes", "$A_u$")
    )


def canonical_prompt_type_label(prompt_type: str) -> str:
    norm = str(prompt_type).strip().lower()
    if norm == "none":
        return "none"

    token_map = {
        "demographics": "$D_u$",
        "previous_experiences": "$E_u$",
        "privacy_attitudes": "$A_u$",
    }
    token_order = {"$D_u$": 0, "$E_u$": 1, "$A_u$": 2}

    parts = [p.strip().lower() for p in str(prompt_type).split("+") if p.strip()]
    mapped = [token_map[p] for p in parts if p in token_map]
    if not mapped:
        return abbreviate_prompt_type(prompt_type)

    deduped = sorted(set(mapped), key=lambda t: token_order[t])
    # Drop adjacent $...$$...$ delimiters and join with tight {+} so the
    # label renders as a single math expression (e.g. "$D_u{+}E_u$").
    inner = "{+}".join(t.strip("$") for t in deduped)
    return f"${inner}$"


def _format_domain_label(domain: str) -> str:
    raw = str(domain).strip()
    if raw.upper() == "ALL_DOMAINS":
        return "5 studies"
    if raw.upper() == "3DOMAIN_INTERSECTION":
        return "3 studies"
    label = raw.title()
    # Restore common acronyms that title() lower-cases.
    for token in ("Llm", "Ai"):
        label = label.replace(token, token.upper())
    return label


def _is_all_domains(domain: str) -> bool:
    return str(domain).strip().upper() == "ALL_DOMAINS"


def _is_intersection(domain: str) -> bool:
    return str(domain).strip().upper() == "3DOMAIN_INTERSECTION"


def _is_pooled(domain: str) -> bool:
    return _is_all_domains(domain) or _is_intersection(domain)


def _has_demographics(prompt_type: str) -> bool:
    return "demographics" in str(prompt_type).strip().lower() or str(
        prompt_type
    ).strip().upper().startswith("D")


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    output_path: Path,
    y_ticks: List[float],
    y_lim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_baseline: bool = True,
) -> None:
    prompt_order = sorted(
        df["prompt_type"].dropna().unique().tolist(),
        key=lambda p: (
            PROMPT_LABEL_ORDER.get(canonical_prompt_type_label(str(p)), 999),
            prompt_type_sort_key(str(p)),
        ),
    )
    raw_domains = df["domain"].dropna().unique().tolist()
    # Plot per-domain lines first, then the two pooled lines on top so they
    # stay visible against the colored series. ALL_DOMAINS (solid bold black)
    # is the 5-domain pool — meaningful only for non-D-prefixed prompt types,
    # so we NaN out its values on the D-prefixed prompts and the line stops
    # naturally at PE+PA. 3DOMAIN_INTERSECTION (dashed bold black) restricts
    # the pool to the three demographics-bearing domains so it is comparable
    # across all 8 prompt types.
    per_domain = sorted([d for d in raw_domains if not _is_pooled(d)])
    pooled_all = [d for d in raw_domains if _is_all_domains(d)]
    pooled_intersection = [d for d in raw_domains if _is_intersection(d)]
    domain_order = per_domain + pooled_all + pooled_intersection
    prompt_display: Dict[str, str] = {
        p: canonical_prompt_type_label(str(p)) for p in prompt_order
    }
    x_labels = [prompt_display[p] for p in prompt_order]
    x = list(range(len(prompt_order)))

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    color_idx = 0
    for domain in domain_order:
        subset = df[df["domain"] == domain]
        metric_map: Dict[str, float] = {
            row["prompt_type"]: row[metric_col] for _, row in subset.iterrows()
        }
        if _is_all_domains(domain):
            # Drop ALL_DOMAINS values for D-prefixed prompts: the underlying
            # pool there is just the 3 D-bearing domains, not 5, so labeling
            # it "All 5 Domains" would be misleading.
            y_values = [
                float("nan") if _has_demographics(pt) else metric_map.get(pt, float("nan"))
                for pt in prompt_order
            ]
            ax.plot(
                x,
                y_values,
                marker="o",
                color="black",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=_format_domain_label(domain),
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=5,
            )
        elif _is_intersection(domain):
            y_values = [metric_map.get(pt, float("nan")) for pt in prompt_order]
            ax.plot(
                x,
                y_values,
                marker="D",
                color="black",
                linestyle="--",
                linewidth=2.0,
                markersize=5,
                label=_format_domain_label(domain),
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
        else:
            y_values = [metric_map.get(pt, float("nan")) for pt in prompt_order]
            ax.plot(
                x,
                y_values,
                marker=DOMAIN_MARKERS[color_idx % len(DOMAIN_MARKERS)],
                color=DOMAIN_PALETTE[color_idx % len(DOMAIN_PALETTE)],
                label=_format_domain_label(domain),
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
            color_idx += 1

    if show_baseline:
        baseline_idx = next(
            (i for i, p in enumerate(prompt_order) if str(p).strip().lower() == "none"),
            None,
        )
        if baseline_idx is not None:
            ax.axvline(
                baseline_idx,
                color="0.4",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Prompt Type")
    ax.set_ylabel(y_label)
    ax.set_yticks(y_ticks)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    n_domains = len(domain_order)
    ax.legend(
        title="Domain",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(n_domains, 3),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


DEFAULT_PROMPT_TYPE_ORDER: Tuple[str, ...] = (
    "none",
    "previous_experiences",
    "privacy_attitudes",
    "previous_experiences+privacy_attitudes",
    "demographics",
    "demographics+previous_experiences",
    "demographics+privacy_attitudes",
    "demographics+previous_experiences+privacy_attitudes",
)


def plot_prompts_summary_grouped_bar(
    runs: Sequence[Tuple[str, Path]],
    output_dir: Path,
    *,
    metric: str = "tolerance_accuracy",
    prompt_type_order: Sequence[str] = DEFAULT_PROMPT_TYPE_ORDER,
    y_label: str = "Tolerance Accuracy (%)",
    y_lim: Tuple[float, float] = (30, 45),
    y_ticks: Sequence[float] = (30, 35, 40, 45),
    cmap_name: str = "tab10",
    figsize: Tuple[float, float] = (11.0, 4.5),
    filename_stem: str = "main_results",
) -> Path:
    """Single-panel grouped bar chart: prompt types on x-axis, models in legend.

    Bars within each prompt-type group: 1 per model (legend identifies
    each). Solid bar = 5-pool (ALL_DOMAINS), hatched bar = 3-pool
    (3DOMAIN_INTERSECTION). For D-prefixed prompts every bar is fully
    hatched because the 5-pool collapses to the 3-pool.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(runs)
    n_prompts = len(prompt_type_order)
    matrix_3pool = np.full((n_models, n_prompts), np.nan, dtype=float)
    matrix_5pool = np.full((n_models, n_prompts), np.nan, dtype=float)

    def _build_row_map(df: pd.DataFrame, target: str) -> Dict[str, float]:
        sub = df[df["domain"].astype(str).str.upper() == target.upper()]
        return {str(row["prompt_type"]): float(row[metric]) for _, row in sub.iterrows()}

    for mi, (display_name, csv_path) in enumerate(runs):
        df = pd.read_csv(csv_path)
        rm_3 = _build_row_map(df, "3DOMAIN_INTERSECTION")
        rm_5 = _build_row_map(df, "ALL_DOMAINS")
        for pi, pt in enumerate(prompt_type_order):
            if pt not in rm_3 or pt not in rm_5:
                raise KeyError(
                    f"Missing pool row for prompt_type={pt!r} ({display_name}, {csv_path})."
                )
            matrix_3pool[mi, pi] = rm_3[pt]
            matrix_5pool[mi, pi] = rm_5[pt]

    cmap = mpl.colormaps[cmap_name]
    bar_colors = [cmap(i % 10) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=figsize)

    group_width = 0.86
    bar_width = group_width / n_models
    group_centers = np.arange(n_prompts, dtype=float)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    prompt_labels = [canonical_prompt_type_label(pt) for pt in prompt_type_order]
    model_labels = [name for name, _ in runs]

    five_pool_visible = matrix_5pool.copy()
    for pi, pt in enumerate(prompt_type_order):
        if _has_demographics(pt):
            five_pool_visible[:, pi] = np.nan

    for mi in range(n_models):
        ax.bar(
            group_centers + offsets[mi],
            matrix_3pool[mi, :],
            width=bar_width,
            facecolor=bar_colors[mi],
            edgecolor="white",
            linewidth=0.4,
            hatch="///",
            zorder=2,
        )
        ax.bar(
            group_centers + offsets[mi],
            five_pool_visible[mi, :],
            width=bar_width,
            color=bar_colors[mi],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )

    # Average across all models — drawn as a line with diamond markers.
    avg_3pool = np.nanmean(matrix_3pool, axis=0)
    avg_5pool_vis = np.nanmean(matrix_5pool, axis=0)
    for pi, pt in enumerate(prompt_type_order):
        if _has_demographics(pt):
            avg_5pool_vis[pi] = np.nan
    ax.plot(group_centers, avg_3pool, color="black", linestyle="--",
            linewidth=1.6, marker="D", markersize=5, zorder=6)
    ax.plot(group_centers, avg_5pool_vis, color="black", linestyle="-",
            linewidth=1.6, marker="D", markersize=5, zorder=6)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(prompt_labels, rotation=0, ha="center")
    ax.set_ylabel(y_label)
    ax.set_yticks(list(y_ticks))
    ax.set_ylim(*y_lim)
    ax.set_xlim(-0.6, n_prompts - 0.4)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch  # local import: only used here.
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(facecolor=bar_colors[mi], edgecolor="black", linewidth=0.4, label=model_labels[mi])
        for mi in range(n_models)
    ] + [
        Line2D([0], [0], color="black", linewidth=1.6, linestyle="-",
               marker="D", markersize=5, label="Average"),
        Patch(facecolor="0.55", edgecolor="black", linewidth=0.4, label="5 studies"),
        Patch(facecolor="0.55", edgecolor="white", linewidth=0.4, hatch="///", label="3 studies"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.0,
        fontsize=7.5,
    )

    out_pdf = output_dir / f"{filename_stem}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def plot_per_model_subplots(
    runs: Sequence[Tuple[str, Path]],
    output_dir: Path,
    *,
    metric_col: str = "tolerance_accuracy",
    y_label: str = "Tolerance Accuracy (%)",
    y_ticks: Sequence[float] = (30, 35, 40, 45, 50),
    y_lim: Tuple[float, float] = (28, 52),
    figsize: Tuple[float, float] = (16.0, 4.2),
    filename_stem: str = "model_subplots",
    show_baseline: bool = True,
    nrows: int = 1,
) -> Path:
    """nrows×ceil(N/nrows) grid of per-model line plots, with one shared legend.

    Each subplot mirrors the per-model `tolerance_accuracy_tau015_by_domain.pdf`
    plot from `data/runs/<model>/plots/`: colored lines for each per-domain
    series, a solid black line for the ALL_DOMAINS pool, and a dashed black
    line for the 3DOMAIN_INTERSECTION pool. A single legend covers all
    subplots and is placed below the figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(runs)
    if n_models == 0:
        raise ValueError("plot_per_model_subplots requires at least one run.")

    dfs = [pd.read_csv(csv_path) for _, csv_path in runs]

    # Use the first dataframe to fix prompt and domain orderings; assume the
    # other models share the same set (raise on mismatch when drawing).
    prompt_order = sorted(
        dfs[0]["prompt_type"].dropna().unique().tolist(),
        key=lambda p: (
            PROMPT_LABEL_ORDER.get(canonical_prompt_type_label(str(p)), 999),
            prompt_type_sort_key(str(p)),
        ),
    )
    raw_domains = dfs[0]["domain"].dropna().unique().tolist()
    per_domain = sorted([d for d in raw_domains if not _is_pooled(d)])
    pooled_all = [d for d in raw_domains if _is_all_domains(d)]
    pooled_intersection = [d for d in raw_domains if _is_intersection(d)]
    domain_order = per_domain + pooled_all + pooled_intersection
    x_labels = [canonical_prompt_type_label(str(p)) for p in prompt_order]
    x = list(range(len(prompt_order)))

    ncols = (n_models + nrows - 1) // nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.45 if nrows > 1 else 0.08},
    )
    axes = np.atleast_1d(axes).ravel().tolist()
    # Hide any spare axes when n_models < nrows*ncols.
    for spare in axes[n_models:]:
        spare.set_visible(False)
    axes = axes[:n_models]

    legend_handles: List[Tuple[str, object]] = []  # (label, Line2D), first-subplot only

    for mi, ax in enumerate(axes):
        df = dfs[mi]
        color_idx = 0

        for domain in domain_order:
            subset = df[df["domain"] == domain]
            metric_map: Dict[str, float] = {
                row["prompt_type"]: row[metric_col] for _, row in subset.iterrows()
            }
            label = _format_domain_label(domain)
            if _is_all_domains(domain):
                # ALL_DOMAINS values are meaningful only for non-D prompts (the
                # underlying 5-domain pool collapses to 3 when D is required).
                y_values = [
                    float("nan") if _has_demographics(pt) else metric_map.get(pt, float("nan"))
                    for pt in prompt_order
                ]
                line, = ax.plot(
                    x, y_values,
                    marker="o", color="black", linestyle="-",
                    linewidth=2.0, markersize=6, label=label,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                )
            elif _is_intersection(domain):
                y_values = [metric_map.get(pt, float("nan")) for pt in prompt_order]
                line, = ax.plot(
                    x, y_values,
                    marker="D", color="black", linestyle="--",
                    linewidth=2.0, markersize=5, label=label,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                )
            else:
                y_values = [metric_map.get(pt, float("nan")) for pt in prompt_order]
                line, = ax.plot(
                    x, y_values,
                    marker=DOMAIN_MARKERS[color_idx % len(DOMAIN_MARKERS)],
                    color=DOMAIN_PALETTE[color_idx % len(DOMAIN_PALETTE)],
                    label=label,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3,
                )
                color_idx += 1

            if mi == 0:
                legend_handles.append((label, line))

        if show_baseline:
            baseline_idx = next(
                (i for i, p in enumerate(prompt_order) if str(p).strip().lower() == "none"),
                None,
            )
            if baseline_idx is not None:
                ax.axvline(
                    baseline_idx, color="0.4", linestyle=":",
                    linewidth=0.8, alpha=0.7, zorder=1,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_title(runs[mi][0], fontsize=10)
        ax.set_yticks(list(y_ticks))
        ax.set_ylim(*y_lim)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    for row_i in range(nrows):
        first_in_row = row_i * ncols
        if first_in_row < n_models:
            axes[first_in_row].set_ylabel(y_label)

    handles = [h for _, h in legend_handles]
    labels = [lab for lab, _ in legend_handles]
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles), frameon=False,
        columnspacing=1.2, handletextpad=0.5,
    )

    out_pdf = output_dir / f"{filename_stem}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create alignment and Spearman plots from evaluation summary CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to evaluation summary CSV (e.g., data/runs/<model>/evaluation_summary.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated plots (e.g., data/runs/<model>/plots).",
    )
    parser.add_argument(
        "--titles",
        action="store_true",
        help="Include in-figure titles. Off by default (paper figures use captions).",
    )
    args = parser.parse_args()

    _set_paper_style()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input summary CSV not found: {input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    required_cols = {
        "prompt_type",
        "domain",
        "alignment_score",
        "spearman_rank_correlation",
        "tolerance_accuracy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Keep ALL_DOMAINS rows so the pooled-across-domains summary appears as a
    # distinct line in each plot.
    df = df.copy()

    metric_configs = [
        {
            "metric_col": "alignment_score",
            "y_label": "Alignment Score (%)",
            "title": "Alignment Score by Domain and Prompt Type",
            "filename_suffix": "alignment_by_domain",
            "y_ticks": [70, 72, 74, 76, 78, 80],
            "y_lim": (69, 80),
        },
        {
            "metric_col": "spearman_rank_correlation",
            "y_label": r"Spearman's $\rho$",
            "title": "Spearman Rank Correlation by Domain and Prompt Type",
            "filename_suffix": "spearman_by_domain",
            "y_ticks": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55],
            "y_lim": (0.05, 0.60),
        },
        {
            "metric_col": "tolerance_accuracy",
            "y_label": "Tolerance Accuracy (%)",
            "title": "Tolerance Accuracy by Domain and Prompt Type",
            "filename_suffix": (
                f"tolerance_accuracy_tau{int(round(TOLERANCE_FRACTION * 100)):03d}_by_domain"
            ),
            "y_ticks": [25, 30, 35, 40, 45, 50, 55],
            "y_lim": (25, 55),
        },
    ]

    saved_paths: List[Path] = []
    for cfg in metric_configs:
        output_path = args.output_dir / f"{cfg['filename_suffix']}.png"
        plot_metric(
            df=df,
            metric_col=cfg["metric_col"],
            y_label=cfg["y_label"],
            output_path=output_path,
            y_ticks=cfg["y_ticks"],
            y_lim=cfg["y_lim"],
            title=cfg["title"] if args.titles else None,
        )
        saved_paths.append(output_path)
        saved_paths.append(output_path.with_suffix(".pdf"))

    print("Saved plots:")
    for p in saved_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
