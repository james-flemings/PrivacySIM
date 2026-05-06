"""Per-cluster evaluation: alignment_score and Spearman per (cluster, prompt_type).

Joins per-row evaluation of a model's outputs.csv with the cluster labels from
persona_clustering.py and aggregates with question-weighted MAE (matching
src/evaluation.py::summarize semantics) plus pooled Spearman per cell.

Outputs:
  data/persona_clustering/<model>__per_cluster_metrics.csv

Aggregation views:
  --view 5domain  : restricted to {none, PE, PA, PE+PA} across all 5 domains
                    (prompt types with full domain coverage)
  --view 3domain  : AAP/CGC/LCA only, all 8 prompt types

Usage:
  python src/per_cluster_eval.py --model gpt-5.4_medium_reasoning --view 5domain
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluation as ev


csv.field_size_limit(sys.maxsize)


NO_DEMO_PROMPTS = (
    "none",
    "previous_experiences",
    "privacy_attitudes",
    "previous_experiences+privacy_attitudes",
)
DEMO_DOMAINS = (
    "AI agent permissions",
    "Chatbot group chat",
    "LLM conversational agents",
)
ALL_DOMAINS = (
    "AI agent permissions",
    "Chatbot group chat",
    "LLM conversational agents",
    "LLM chatbot",
    "LLM healthcare consultation",
)
CLUSTER_ORDER = (
    "high / high",   # ≈ Dupree Fundamentalist
    "low / high",    # ≈ Dupree Lazy Expert
    "high / mid",    # ≈ Dupree Technician
    "mid / mid",     # ≈ Dupree Amateur
    "low / low",     # ≈ Dupree Marginally Concerned
)


def per_row_eval(outputs_csv: Path) -> pd.DataFrame:
    rows = []
    with outputs_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            domain = r.get("domain", "")
            res = ev.evaluate_single_row(
                domain=domain,
                user_raw=r.get("responses", ""),
                model_raw=r.get("final_answer", ""),
            )
            tol_acc = (
                100.0 * res.tolerance_match_count / res.compared_questions
                if res.compared_questions > 0
                else float("nan")
            )
            # Signed bias: mean(model_val - user_val) per row, normalized by the
            # domain scale width so the metric is comparable across the 4-pt,
            # 5-pt, and 0-100 domains. Positive = simulator over-predicts (more
            # permissive / higher-magnitude than the user); negative = simulator
            # under-predicts (more restrictive / lower-magnitude than the user).
            # In [-1, +1].
            domain_range = ev.DOMAIN_SCORE_RANGE.get(ev.normalize_text(domain))
            signed_bias = float("nan")
            if res.compared_questions > 0 and domain_range and domain_range > 0:
                diffs = [m - u for m, u in zip(res.model_values, res.user_values)]
                signed_bias = sum(diffs) / len(diffs) / domain_range
            rows.append(
                {
                    "uuid": r.get("uuid"),
                    "domain": r.get("domain"),
                    "prompt_type": r.get("prompt_type"),
                    "compared_questions": res.compared_questions,
                    "alignment_score": res.alignment_score,
                    "spearman_rank_correlation": res.spearman_rank_correlation,
                    "tolerance_accuracy": tol_acc,
                    "signed_bias": signed_bias,
                }
            )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Per-user means of row-level alignment_score, Spearman, and
    tolerance_accuracy (NaN Spearmans are skipped; users_dropped tracks how
    many were excluded)."""
    agg = (
        df.groupby(group_cols)
        .agg(
            n_rows=("uuid", "count"),
            sum_compared=("compared_questions", "sum"),
            alignment_score=("alignment_score", "mean"),
            spearman=("spearman_rank_correlation", "mean"),
            tolerance_accuracy=("tolerance_accuracy", "mean"),
            signed_bias=("signed_bias", "mean"),
            users_dropped=(
                "spearman_rank_correlation",
                lambda s: int(s.isna().sum()),
            ),
        )
        .reset_index()
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cluster-csv",
        type=Path,
        default=REPO_ROOT / "data" / "persona_clustering" / "cluster_labels.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "persona_clustering",
    )
    parser.add_argument(
        "--view",
        choices=("5domain", "3domain"),
        default="5domain",
    )
    parser.add_argument(
        "--quintile-heatmap",
        action="store_true",
        help=(
            "Emit a 5x5 (stated_quintile x revealed_quintile) heatmap of "
            "alignment_score and Spearman, broken down by prompt_type. "
            "Theory-neutral primary analysis for RQ2."
        ),
    )
    parser.add_argument(
        "--permutation-test",
        action="store_true",
        help=(
            "Run a permutation test (1000 shuffles) on cluster labels to "
            "establish a null distribution for cluster-spread of alignment."
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs_csv = REPO_ROOT / "data" / "runs" / args.model / "outputs.csv"
    print(f"Per-row eval of {outputs_csv} ...")
    eval_df = per_row_eval(outputs_csv)
    print(f"  {len(eval_df)} rows evaluated")

    cluster_df = pd.read_csv(args.cluster_csv)[["uuid", "cluster"]]
    cluster_full = pd.read_csv(args.cluster_csv)
    df_all = eval_df.merge(
        cluster_full[["uuid", "cluster", "stated_z", "revealed_z"]], on="uuid", how="left"
    )

    df = df_all.copy()
    df = df[df["cluster"] != "Unassigned"].dropna(subset=["cluster"])

    if args.view == "5domain":
        df = df[
            df["prompt_type"].isin(NO_DEMO_PROMPTS)
            & df["domain"].isin(ALL_DOMAINS)
        ]
    else:
        df = df[df["domain"].isin(DEMO_DOMAINS)]

    metrics = aggregate(df, group_cols=["cluster", "prompt_type"])
    metrics["model"] = args.model
    metrics["view"] = args.view

    cols = [
        "model",
        "view",
        "cluster",
        "prompt_type",
        "n_rows",
        "sum_compared",
        "alignment_score",
        "spearman",
        "tolerance_accuracy",
        "signed_bias",
        "users_dropped",
    ]
    metrics = metrics[cols]
    out_path = args.out_dir / f"{args.model}__per_cluster_metrics__{args.view}.csv"
    metrics.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Pretty-print pivots
    def _pivot(values_col: str, label: str):
        piv = metrics.pivot(index="prompt_type", columns="cluster", values=values_col)
        piv = piv.reindex(columns=[c for c in CLUSTER_ORDER if c in piv.columns])
        if args.view == "5domain":
            piv = piv.reindex(list(NO_DEMO_PROMPTS))
        print(f"\n=== {label} ({args.view} view, model={args.model}) ===")
        print(piv.round(3).to_string())

    _pivot("tolerance_accuracy", "Tolerance accuracy (per-user mean)")
    _pivot("alignment_score", "Alignment score (per-user mean)")
    _pivot("spearman", "Spearman ρ (per-user mean)")
    _pivot("signed_bias", "Signed bias (per-user mean, normalized; +=over-predict, -=under-predict)")

    # Per-cluster row counts for sanity
    print("\nRow counts per (cluster, prompt_type):")
    print(
        metrics.pivot(index="prompt_type", columns="cluster", values="n_rows")
        .reindex(list(NO_DEMO_PROMPTS) if args.view == "5domain" else None)
        .to_string()
    )

    if args.quintile_heatmap:
        # Build (stated_quintile x revealed_quintile) heatmap on the SAME
        # restricted dataframe (5domain or 3domain view), but include
        # Unassigned/CGC users in the heatmap (theory-neutral).
        heat = df_all.copy()
        if args.view == "5domain":
            heat = heat[
                heat["prompt_type"].isin(NO_DEMO_PROMPTS)
                & heat["domain"].isin(ALL_DOMAINS)
            ]
        else:
            heat = heat[heat["domain"].isin(DEMO_DOMAINS)]
        heat = heat.dropna(subset=["stated_z", "revealed_z"])

        # Quintile bins on the per-domain z-scores (already z-scored)
        heat["s_q"] = pd.qcut(heat["stated_z"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        heat["r_q"] = pd.qcut(heat["revealed_z"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")

        heat_metrics = aggregate(heat, group_cols=["s_q", "r_q", "prompt_type"])
        heat_metrics["model"] = args.model
        heat_metrics["view"] = args.view
        heat_path = args.out_dir / f"{args.model}__heatmap_metrics__{args.view}.csv"
        heat_metrics[
            [
                "model",
                "view",
                "s_q",
                "r_q",
                "prompt_type",
                "n_rows",
                "sum_compared",
                "alignment_score",
                "spearman",
                "tolerance_accuracy",
                "signed_bias",
                "users_dropped",
            ]
        ].to_csv(heat_path, index=False)
        print(f"\nWrote heatmap metrics: {heat_path}")

        for pt in ("none", "previous_experiences+privacy_attitudes"):
            sub = heat_metrics[heat_metrics["prompt_type"] == pt]
            if len(sub) == 0:
                continue
            for metric_col, label in (
                ("tolerance_accuracy", "tol_acc"),
                ("alignment_score", "alignment"),
                ("spearman", "spearman"),
            ):
                piv = sub.pivot(index="s_q", columns="r_q", values=metric_col)
                print(f"\n=== Heatmap {label} | prompt={pt} ({args.view} view) ===")
                print("rows = stated_z quintile (1=low, 5=high)")
                print("cols = revealed_z quintile (1=low, 5=high)")
                print(piv.round(3).to_string())

    if args.permutation_test:
        from numpy.random import default_rng
        rng = default_rng(42)
        # Use PE+PA prompt only for the permutation test
        sub = df[df["prompt_type"] == "previous_experiences+privacy_attitudes"].copy()
        clusters = sub["cluster"].unique()
        actual = []
        for c in clusters:
            g = sub[sub["cluster"] == c]
            vals = g["tolerance_accuracy"].dropna()
            if vals.empty:
                continue
            actual.append(float(vals.mean()))
        observed_spread = max(actual) - min(actual)

        nulls = []
        labels = sub["cluster"].to_numpy()
        tol = sub["tolerance_accuracy"].to_numpy(dtype=float)
        for _ in range(1000):
            shuffled = rng.permutation(labels)
            cluster_tol = []
            for c in clusters:
                mask = shuffled == c
                vals = tol[mask]
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                cluster_tol.append(float(vals.mean()))
            if cluster_tol:
                nulls.append(max(cluster_tol) - min(cluster_tol))
        p_val = float(np.mean(np.array(nulls) >= observed_spread))
        print(
            f"\nPermutation test (PE+PA prompt, tolerance_accuracy): observed cluster-spread = "
            f"{observed_spread:.4f}, null mean = {np.mean(nulls):.4f}, "
            f"null 99th pct = {np.percentile(nulls, 99):.4f}, p = {p_val:.4f}"
        )


if __name__ == "__main__":
    main()
