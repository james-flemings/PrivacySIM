"""Generate a single-panel grouped-bar figure summarizing tolerance accuracy
across 9 LLMs and all 8 prompt-types.

Each (model, prompt-type) cell shows two overlaid bars at the same x-position:
a solid cividis bar for the 5-study pool (`ALL_DOMAINS`) and a hatched
cividis bar drawn behind for the 3-study demographics-bearing pool
(`3DOMAIN_INTERSECTION`). When 3-pool > 5-pool the hatched cap shows above
the solid bar. For D-prefixed prompts the two values are identical (Liu and
Tran omit demographics) so the bar appears uniformly solid."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from plots import (  # noqa: E402
    DEFAULT_PROMPT_TYPE_ORDER,
    _set_paper_style,
    plot_per_model_subplots,
    plot_prompts_summary_grouped_bar,
)


# (model_dir_name, display_name) — order on x-axis is the order here.
MODELS = [
    ("gemini_3.1_pro",           "Gemini 3.1 Pro"),
    ("gemini_3_flash",           "Gemini 3 Flash"),
    ("gpt-5.4_medium_reasoning", "GPT-5.4"),
    ("sonnet_4.6",               "Sonnet 4.6"),
    ("gpt-5.4-mini",             "GPT-5.4 Mini"),
    ("gemma4_32B",               "Gemma4 32B"),
    ("qwen3.5_27B",              "Qwen3.5 27B"),
    ("qwen3.5_122B",             "Qwen3.5 122B"),
    ("nvidia_nemotron_3_120B",   "Nemotron 3 120B"),
]

RUNS_DIR = REPO_ROOT / "data" / "runs"
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"


def main() -> None:
    runs = []
    for dir_name, display_name in MODELS:
        csv_path = RUNS_DIR / dir_name / "evaluation_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing evaluation summary: {csv_path}")
        runs.append((display_name, csv_path))

    _set_paper_style()

    runs_prompts = [(name, path) for name, path in runs if "122B" not in name]
    out_pdf2 = plot_prompts_summary_grouped_bar(
        runs=runs_prompts,
        output_dir=OUTPUT_DIR,
        prompt_type_order=DEFAULT_PROMPT_TYPE_ORDER,
    )
    print(f"Saved: {out_pdf2}")

    subplot_targets = {"Gemini 3.1 Pro", "Gemini 3 Flash", "GPT-5.4", "Sonnet 4.6"}
    runs_subplots = [(name, path) for name, path in runs if name in subplot_targets]
    out_pdf3 = plot_per_model_subplots(
        runs=runs_subplots,
        output_dir=OUTPUT_DIR,
        filename_stem="domain_subplots",
    )
    print(f"Saved: {out_pdf3}")

    # Full appendix figure: all 9 models arranged in a 3x3 grid.
    out_pdf4 = plot_per_model_subplots(
        runs=runs,
        output_dir=OUTPUT_DIR,
        filename_stem="domain_subplots_full",
        figsize=(13.0, 11.0),
        nrows=3,
    )
    print(f"Saved: {out_pdf4}")


if __name__ == "__main__":
    main()
