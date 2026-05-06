"""Shared scaffolding for the closed-source batch sim backends
(openai_user_simulation.py, claude_user_simulation.py, gemini_user_simulation.py).

Holds only logic that is byte-identical across all three providers:
column names, JSONL custom-id parsing, payload coercion, the pending-rows
filter, the artifact-path derivation, and the outputs.csv writer. The
provider-specific pieces — request construction, SDK calls, polling, the
per-provider JSONL field shape, and final-answer extraction — stay in each
backend module.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, Set, Tuple

import pandas as pd


SOURCE_INDEX_COLUMN = "source_row_index"
RAW_RESPONSE_COLUMN = "raw_response"
FINAL_ANSWER_COLUMN = "final_answer"


def to_int(value) -> int:
    return int(str(value))


def partition_dataframe(df: pd.DataFrame, batch_size: int | None) -> list[pd.DataFrame]:
    """Split a DataFrame into chunks of at most `batch_size` rows."""
    n = len(df)
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        return [df]
    return [df.iloc[i:i + batch_size] for i in range(0, n, batch_size)]


def append_file(src: Path, dst: Path) -> None:
    """Append src file contents to dst, ensuring a trailing newline."""
    text = src.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    with dst.open("a", encoding="utf-8") as f:
        f.write(text)


def parse_source_index(key: str):
    """Pull the original row index out of a 'row-<int>' custom_id / key.
    Returns None if the value isn't shaped that way (silently ignored)."""
    prefix = "row-"
    if not isinstance(key, str) or not key.startswith(prefix):
        return None
    try:
        return int(key[len(prefix):])
    except ValueError:
        return None


def coerce_payload(raw_value: Any) -> dict[str, Any] | None:
    """Best-effort dict coercion for a raw_response cell. Tries JSON first,
    then ast.literal_eval (which handles single-quoted dict reprs that
    pandas sometimes round-trips through CSV)."""
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            return None
    return None


def select_pending(df: pd.DataFrame, prompt_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate `df` with SOURCE_INDEX_COLUMN and return (annotated_df, pending_df)
    where pending_df is the subset with non-empty prompt text."""
    selected = df.copy()
    selected[SOURCE_INDEX_COLUMN] = selected.index
    pending = selected[
        selected[prompt_column].notna()
        & (selected[prompt_column].astype(str).str.strip() != "")
    ]
    return selected, pending


def derive_artifact_paths(output_csv: Path) -> Tuple[Path, Path, Path, Path]:
    """Derive the four side-artifact paths from the outputs.csv path. The
    suffix (e.g. '__pmt' from 'outputs__pmt.csv') is preserved so theory /
    domain runs stay isolated."""
    run_dir = output_csv.parent
    suffix = output_csv.stem.removeprefix("outputs")
    return (
        run_dir / f"batch_response{suffix}.jsonl",
        run_dir / f"batch_errors{suffix}.jsonl",
        run_dir / f"missing_rows{suffix}.json",
        run_dir / f"batch_artifacts{suffix}",
    )


def build_outputs_csv(
    input_df: pd.DataFrame,
    prompt_column: str,
    batch_response_file: Path,
    output_csv: Path,
    missing_log_file: Path,
    max_json_errors: int,
    parse_jsonl_fn: Callable[[Path, int], Tuple[Dict[int, str], Set[int], Set[int], int]],
    extract_final_answer_fn: Callable[[Any], str],
) -> None:
    """Map batch JSONL onto input rows, extract final_answer, and write outputs.csv.

    Provider-specific JSONL parsing (`parse_jsonl_fn`) and final-answer
    extraction (`extract_final_answer_fn`) are injected so the rest of the
    bookkeeping (submitted-rows accounting, missing-row logging, stat printing)
    can be shared.
    """
    output_df = input_df.copy()
    output_df[SOURCE_INDEX_COLUMN] = output_df.index
    output_df[RAW_RESPONSE_COLUMN] = ""

    submitted_indices = {
        to_int(v)
        for v in output_df[
            output_df[prompt_column].notna()
            & (output_df[prompt_column].astype(str).str.strip() != "")
        ][SOURCE_INDEX_COLUMN].tolist()
    }

    updates, seen_indices, missing_response_body_indices, failures = parse_jsonl_fn(
        batch_response_file, max_json_errors
    )

    updated = 0
    for idx in output_df.index:
        src_idx = to_int(output_df.at[idx, SOURCE_INDEX_COLUMN])
        if src_idx in updates:
            output_df.at[idx, RAW_RESPONSE_COLUMN] = updates[src_idx]
            updated += 1

    missing_from_batch_output = sorted(submitted_indices - seen_indices)
    missing_response_body_rows = sorted(missing_response_body_indices)

    if missing_response_body_rows:
        print(
            "Rows with missing response body: "
            + ", ".join(str(i) for i in missing_response_body_rows)
        )
    if missing_from_batch_output:
        print(
            "Submitted rows missing from batch output file: "
            + ", ".join(str(i) for i in missing_from_batch_output)
        )

    missing_log_file.parent.mkdir(parents=True, exist_ok=True)
    missing_log_file.write_text(
        json.dumps(
            {
                "batch_response_file": str(batch_response_file),
                "missing_response_body_rows": missing_response_body_rows,
                "missing_from_batch_output_rows": missing_from_batch_output,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Missing-row log written to {missing_log_file}")

    output_df[FINAL_ANSWER_COLUMN] = output_df[RAW_RESPONSE_COLUMN].apply(
        extract_final_answer_fn
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    extracted_count = (
        output_df[FINAL_ANSWER_COLUMN].astype(str).str.strip() != ""
    ).sum()
    success_count = len(submitted_indices) - len(missing_from_batch_output)
    print(f"Total input rows: {len(output_df)}")
    print(f"Total submitted rows: {len(submitted_indices)}")
    print(f"Rows updated from batch output: {updated}")
    print(f"Failed items encoded in {RAW_RESPONSE_COLUMN}: {failures}")
    print(f"Successful response rows (estimated): {max(0, success_count - failures)}")
    print(f"Rows with non-empty final answer: {extracted_count}/{len(output_df)}")
    print(f"Saved outputs to {output_csv}")
