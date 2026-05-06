"""Run one OpenAI Batch API inference end-to-end over a prompts CSV.

Submits the batch, polls to completion, downloads the raw JSONL response,
maps each response back to its source row, and writes an `outputs.csv` whose
`final_answer` column is already populated by extracting the response text
from the OpenAI Responses API payload. This mirrors the contract of
`src/user_simulation.py` (the vLLM sim), so `src/pipeline.py` can dispatch to
either script based on the model's `backend` field in `configs/models.yaml`.

Side artifacts written next to --output-csv:
    batch_response.jsonl  - raw Batch API output
    batch_errors.jsonl    - raw Batch API error output (if any)
    missing_rows.json     - rows that were submitted but did not get a response body

Example:
    python src/openai_user_simulation.py \\
        --input-csv data/dataset_with_prompts.csv \\
        --output-csv data/runs/gpt-5.4/outputs.csv \\
        --model gpt-5.4 \\
        --reasoning-effort medium \\
        --reasoning-summary detailed \\
        --max-output-tokens 8192 \\
        --completion-window 24h \\
        --poll-interval-seconds 30 \\
        --prompt-column llm_prompt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import pandas as pd
from openai import OpenAI

from _batch_sim_common import (
    SOURCE_INDEX_COLUMN,
    append_file,
    build_outputs_csv,
    coerce_payload,
    derive_artifact_paths,
    parse_source_index,
    partition_dataframe,
    select_pending,
    to_int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit OpenAI Batch API inference over CSV prompts and write outputs.csv."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompt-column", type=str, default="llm_prompt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--reasoning-summary", type=str, default="detailed")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--completion-window", type=str, default="24h")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Max rows submitted per OpenAI Batch API job. If set and smaller than "
            "the number of pending prompts, the dataset is partitioned into "
            "multiple sequential batches (each submitted and waited on before the "
            "next). Use this to stay under OpenAI's enqueued-token limits."
        ),
    )
    parser.add_argument(
        "--max-error-log-lines",
        type=int,
        default=50,
        help="Maximum number of lines to print from a batch error file.",
    )
    parser.add_argument(
        "--max-json-errors",
        type=int,
        default=20,
        help="Maximum malformed JSONL lines to print before stopping parse.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="0-based start row when slicing input with --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional cap on rows processed from the input. Useful for smoke "
            "tests (e.g. --start-row 0 --limit 50)."
        ),
    )
    return parser.parse_args()


def build_batch_jsonl(
    args: argparse.Namespace,
    pending_df_chunk: pd.DataFrame,
    batch_response_file: Path,
    part_index: int | None = None,
) -> Tuple[Path, Set[int]]:
    """Create the JSONL input file for one batch (one chunk of pre-filtered rows)."""
    if pending_df_chunk.empty:
        raise ValueError("Empty chunk passed to build_batch_jsonl.")

    submitted_indices = {to_int(v) for v in pending_df_chunk[SOURCE_INDEX_COLUMN].tolist()}

    suffix = batch_response_file.stem.removeprefix("batch_response")
    batch_dir = batch_response_file.parent / f"batch_artifacts{suffix}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    part_suffix = f"_part_{part_index}" if part_index is not None else ""
    jsonl_path = batch_dir / f"openai_batch_input_{ts}{part_suffix}.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in pending_df_chunk.iterrows():
            src_idx = int(row[SOURCE_INDEX_COLUMN])
            prompt = str(row[args.prompt_column]).strip()

            request_obj = {
                "custom_id": f"row-{src_idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.model,
                    "input": [{"role": "user", "content": prompt}],
                    "reasoning": {
                        "effort": args.reasoning_effort,
                        "summary": args.reasoning_summary,
                    },
                    "max_output_tokens": args.max_output_tokens,
                },
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    return jsonl_path, submitted_indices


def submit_batch(
    client: OpenAI,
    args: argparse.Namespace,
    jsonl_path: Path,
    batch_response_file: Path,
) -> str:
    with jsonl_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    metadata = {
        "script": "openai_user_simulation.py",
        "input_csv": str(args.input_csv),
        "batch_response_file": str(batch_response_file),
    }

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata=metadata,
    )
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int):
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        print(f"Batch {batch_id} status: {status}")
        if status in terminal_states:
            return batch
        time.sleep(poll_interval_seconds)


def file_content_to_text(file_content) -> str:
    if isinstance(file_content, str):
        return file_content
    if isinstance(file_content, (bytes, bytearray)):
        return file_content.decode("utf-8")

    text_attr = getattr(file_content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        maybe_text = text_attr()
        if isinstance(maybe_text, str):
            return maybe_text

    read_attr = getattr(file_content, "read", None)
    if callable(read_attr):
        data = read_attr()
        if isinstance(data, str):
            return data
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8")

    content_attr = getattr(file_content, "content", None)
    if isinstance(content_attr, (bytes, bytearray)):
        return content_attr.decode("utf-8")

    return str(file_content)


def print_batch_diagnostics(client: OpenAI, batch, max_error_log_lines: int) -> None:
    batch_id = getattr(batch, "id", "<unknown>")
    status = getattr(batch, "status", None)
    request_counts = getattr(batch, "request_counts", None)
    errors = getattr(batch, "errors", None)
    error_file_id = getattr(batch, "error_file_id", None)
    output_file_id = getattr(batch, "output_file_id", None)

    print(f"Batch diagnostics for {batch_id}")
    print(f"status: {status}")
    print(f"request_counts: {request_counts}")
    print(f"errors: {errors}")
    print(f"error_file_id: {error_file_id}")
    print(f"output_file_id: {output_file_id}")

    if not error_file_id:
        return

    try:
        file_content = client.files.content(error_file_id)
        error_text = file_content_to_text(file_content)
    except Exception as exc:
        print(f"Failed to fetch error file {error_file_id}: {exc}")
        return

    lines = error_text.splitlines()
    if not lines:
        print("Error file is empty")
        return

    max_lines = max(1, max_error_log_lines)
    preview = lines[:max_lines]
    print(f"Error file preview (showing {len(preview)} of {len(lines)} lines):")
    for line in preview:
        print(line)
    if len(lines) > max_lines:
        print("... (truncated)")


def write_batch_files(client: OpenAI, batch, response_file: Path, error_file: Path) -> None:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(
            f"Batch {getattr(batch, 'id', '<unknown>')} completed but has no output_file_id"
        )

    response_file.parent.mkdir(parents=True, exist_ok=True)
    response_text = file_content_to_text(client.files.content(output_file_id))
    response_file.write_text(response_text, encoding="utf-8")
    print(f"Saved batch response JSONL to {response_file}")

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_text = file_content_to_text(client.files.content(error_file_id))
        error_file.write_text(error_text, encoding="utf-8")
        print(f"Saved batch error JSONL to {error_file}")


def parse_batch_jsonl(
    jsonl_path: Path, max_json_errors: int
) -> Tuple[Dict[int, str], Set[int], Set[int], int]:
    updates: Dict[int, str] = {}
    seen_indices: Set[int] = set()
    missing_response_body_indices: Set[int] = set()
    failures = 0

    bad_json_count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_json_count += 1
                if bad_json_count <= max_json_errors:
                    print(f"Malformed JSONL at line {line_no}: {exc}")
                if bad_json_count >= max_json_errors:
                    raise ValueError(
                        f"Too many malformed JSONL lines while reading {jsonl_path}"
                    )
                continue

            src_idx = parse_source_index(item.get("custom_id"))
            if src_idx is None:
                continue

            seen_indices.add(src_idx)

            error = item.get("error")
            if error:
                failures += 1
                updates[src_idx] = json.dumps({"batch_error": error}, ensure_ascii=False)
                continue

            response_payload = item.get("response", {}).get("body")
            if response_payload is None:
                failures += 1
                missing_response_body_indices.add(src_idx)
                updates[src_idx] = json.dumps(
                    {"batch_error": "Missing response.body"}, ensure_ascii=False
                )
                continue

            updates[src_idx] = json.dumps(response_payload, ensure_ascii=False)

    return updates, seen_indices, missing_response_body_indices, failures


def extract_final_answer(raw_value: Any) -> str:
    payload = coerce_payload(raw_value)
    if payload is None:
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = payload.get("output")
    if not isinstance(output_items, list):
        return ""

    for item in output_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            part_text = part.get("text")
            if part_type in {"output_text", "text"} and isinstance(part_text, str):
                cleaned = part_text.strip()
                if cleaned:
                    return cleaned

    return ""


def run_batches(
    client: OpenAI,
    args: argparse.Namespace,
    pending_df: pd.DataFrame,
    batch_response_file: Path,
    batch_error_file: Path,
) -> Set[int]:
    """Submit one or more OpenAI batches (partitioned by args.batch_size) and
    concatenate their outputs into batch_response_file. Each chunk is submitted
    and awaited sequentially so only one batch is enqueued at a time.

    Resume behavior: each chunk's output is cached under
    `<run_dir>/batch_artifacts/batch_response_part_<i>.jsonl`. On rerun, any
    chunk whose part file already exists is skipped (no resubmission). To
    force a fresh run, delete the `batch_artifacts/` directory (or specific
    part files) before invoking.
    """
    chunks = partition_dataframe(pending_df, args.batch_size)
    n_chunks = len(chunks)

    suffix = batch_response_file.stem.removeprefix("batch_response")
    artifacts_dir = batch_response_file.parent / f"batch_artifacts{suffix}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def part_response_path(i: int) -> Path:
        return artifacts_dir / f"batch_response_part_{i}.jsonl"

    def part_error_path(i: int) -> Path:
        return artifacts_dir / f"batch_errors_part_{i}.jsonl"

    def is_cached(i: int) -> bool:
        p = part_response_path(i)
        return p.exists() and p.stat().st_size > 0

    cached_count = sum(1 for i in range(1, n_chunks + 1) if is_cached(i))
    print(
        f"Partitioned {len(pending_df)} pending rows into {n_chunks} batch(es) "
        f"(batch_size={args.batch_size}); {cached_count} cached, "
        f"{n_chunks - cached_count} to submit"
    )

    all_submitted_indices: Set[int] = set()

    for i, chunk in enumerate(chunks, start=1):
        label = f"batch {i}/{n_chunks}"
        part_response_file = part_response_path(i)
        part_error_file = part_error_path(i)

        chunk_indices = {to_int(v) for v in chunk[SOURCE_INDEX_COLUMN].tolist()}
        all_submitted_indices |= chunk_indices

        if is_cached(i):
            print(
                f"\n--- {label}: reusing cached {part_response_file.name} "
                f"({len(chunk)} rows) ---"
            )
            continue

        print(f"\n--- {label}: submitting {len(chunk)} rows ---")
        part_index = i if n_chunks > 1 else None
        jsonl_path, _ = build_batch_jsonl(
            args, chunk, batch_response_file, part_index=part_index
        )
        batch_id = submit_batch(client, args, jsonl_path, batch_response_file)
        print(f"{label} batch_id: {batch_id} ({len(chunk_indices)} rows)")
        print(f"{label} input file: {jsonl_path}")

        final_batch = wait_for_batch(
            client, batch_id, poll_interval_seconds=args.poll_interval_seconds
        )
        final_status = getattr(final_batch, "status", None)
        if final_status != "completed":
            print_batch_diagnostics(client, final_batch, args.max_error_log_lines)
            raise RuntimeError(
                f"{label} ended with status '{final_status}'. batch_id={batch_id}"
            )

        write_batch_files(
            client,
            final_batch,
            response_file=part_response_file,
            error_file=part_error_file,
        )

    # All chunks present as part files; rebuild the combined files fresh.
    batch_response_file.parent.mkdir(parents=True, exist_ok=True)
    batch_response_file.write_text("", encoding="utf-8")
    if batch_error_file.exists():
        batch_error_file.unlink()

    for i in range(1, n_chunks + 1):
        append_file(part_response_path(i), batch_response_file)
        if part_error_path(i).exists():
            append_file(part_error_path(i), batch_error_file)

    print(f"\nSaved combined batch response JSONL to {batch_response_file}")
    if batch_error_file.exists():
        print(f"Saved combined batch error JSONL to {batch_error_file}")
    return all_submitted_indices


def main() -> None:
    args = parse_args()

    input_df = pd.read_csv(args.input_csv)
    if args.prompt_column not in input_df.columns:
        raise ValueError(f"Missing prompt column: {args.prompt_column}")

    if args.start_row < 0:
        raise ValueError("--start-row must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    end_row = None if args.limit is None else args.start_row + args.limit
    input_df = input_df.iloc[args.start_row:end_row].copy()

    _, pending_df = select_pending(input_df, args.prompt_column)
    if pending_df.empty:
        raise ValueError("No non-empty prompts found in input CSV.")

    batch_response_file, batch_error_file, missing_log_file, _ = derive_artifact_paths(
        args.output_csv
    )

    client = OpenAI()

    run_batches(
        client=client,
        args=args,
        pending_df=pending_df,
        batch_response_file=batch_response_file,
        batch_error_file=batch_error_file,
    )

    build_outputs_csv(
        input_df=input_df,
        prompt_column=args.prompt_column,
        batch_response_file=batch_response_file,
        output_csv=args.output_csv,
        missing_log_file=missing_log_file,
        max_json_errors=args.max_json_errors,
        parse_jsonl_fn=parse_batch_jsonl,
        extract_final_answer_fn=extract_final_answer,
    )


if __name__ == "__main__":
    main()
