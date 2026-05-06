"""Run one Anthropic Message Batches API inference end-to-end over a prompts CSV.

Submits the batch, polls to completion, serializes each result to JSONL, maps
each response back to its source row, and writes an `outputs.csv` whose
`final_answer` column is already populated by extracting the response text from
the Anthropic Messages payload. This mirrors the contract of
`src/user_simulation.py` (the vLLM sim) and `src/openai_user_simulation.py`
(the OpenAI batch sim), so `src/pipeline.py` can dispatch to any of the three
based on the model's `backend` field in `configs/models.yaml`.

Side artifacts written next to --output-csv:
    batch_response.jsonl  - one Anthropic batch result per line (Pydantic JSON)
    batch_errors.jsonl    - subset of results with type != "succeeded"
    missing_rows.json     - rows that were submitted but did not get a response

Example:
    python src/claude_user_simulation.py \\
        --input-csv data/dataset_with_prompts.csv \\
        --output-csv data/runs/sonnet_4.6/outputs.csv \\
        --model claude-sonnet-4-6 \\
        --max-tokens 16000 \\
        --batch-size 5000 \\
        --poll-interval-seconds 30 \\
        --prompt-column llm_prompt \\
        --thinking-type adaptive \\
        --effort high
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import anthropic
import pandas as pd

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
        description="Submit Anthropic Message Batches API inference over CSV prompts and write outputs.csv."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompt-column", type=str, default="llm_prompt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Max rows submitted per Anthropic Message Batches job. If set and "
            "smaller than the number of pending prompts, the dataset is "
            "partitioned into multiple sequential batches (each submitted and "
            "waited on before the next). Anthropic's per-batch caps are 100,000 "
            "requests and 256 MB; pick a smaller value to checkpoint progress "
            "and bound resubmission cost on transient failures."
        ),
    )
    parser.add_argument(
        "--thinking-type",
        type=str,
        default=None,
        choices=["adaptive", "disabled"],
        help=(
            "Optional. When set, attaches `thinking={'type': <value>}` to each "
            "request. 'adaptive' is the only on-mode for Opus 4.7 and is the "
            "recommended setting for Sonnet 4.6. Omit for models that don't "
            "support thinking (e.g. Haiku 4.5)."
        ),
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=(
            "Optional. When set, attaches `output_config={'effort': <value>}` "
            "to each request. Supported on Opus 4.5+ and Sonnet 4.6. Errors on "
            "Haiku 4.5 and older models. 'max' / 'xhigh' are Opus-tier only."
        ),
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
            "tests (e.g. --start-row 0 --limit 1)."
        ),
    )
    return parser.parse_args()


def build_request_params(args: argparse.Namespace, prompt: str) -> dict:
    params: dict = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.thinking_type:
        params["thinking"] = {"type": args.thinking_type}
    if args.effort:
        params["output_config"] = {"effort": args.effort}
    return params


def build_batch_requests(
    args: argparse.Namespace, pending_df_chunk: pd.DataFrame
) -> Tuple[list[dict], Set[int]]:
    """Build the request list for one batch (one chunk of pre-filtered rows)."""
    if pending_df_chunk.empty:
        raise ValueError("Empty chunk passed to build_batch_requests.")

    submitted_indices = {to_int(v) for v in pending_df_chunk[SOURCE_INDEX_COLUMN].tolist()}

    requests: list[dict] = []
    for _, row in pending_df_chunk.iterrows():
        src_idx = int(row[SOURCE_INDEX_COLUMN])
        prompt = str(row[args.prompt_column]).strip()
        requests.append(
            {
                "custom_id": f"row-{src_idx}",
                "params": build_request_params(args, prompt),
            }
        )

    return requests, submitted_indices


def submit_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(
    client: anthropic.Anthropic, batch_id: str, poll_interval_seconds: int
):
    terminal_states = {"ended"}
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = getattr(batch, "processing_status", None)
        counts = getattr(batch, "request_counts", None)
        if counts is not None:
            print(
                f"Batch {batch_id} status: {status} | "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored} canceled={counts.canceled} "
                f"expired={counts.expired}"
            )
        else:
            print(f"Batch {batch_id} status: {status}")
        if status in terminal_states:
            return batch
        time.sleep(poll_interval_seconds)


def write_batch_results(
    client: anthropic.Anthropic,
    batch_id: str,
    response_file: Path,
    error_file: Path,
) -> None:
    """Stream batch results, write every result to response_file and copy
    non-succeeded results to error_file. The Anthropic SDK returns Pydantic
    models; we serialize each via .model_dump_json() so downstream parsing can
    treat the file as plain JSONL."""
    response_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    n_success = n_errored = n_canceled = n_expired = 0
    with response_file.open("w", encoding="utf-8") as f_resp, error_file.open(
        "w", encoding="utf-8"
    ) as f_err:
        for result in client.messages.batches.results(batch_id):
            line = result.model_dump_json()
            f_resp.write(line + "\n")
            rtype = result.result.type
            if rtype == "succeeded":
                n_success += 1
            else:
                f_err.write(line + "\n")
                if rtype == "errored":
                    n_errored += 1
                elif rtype == "canceled":
                    n_canceled += 1
                elif rtype == "expired":
                    n_expired += 1

    print(
        f"Saved batch response JSONL to {response_file} "
        f"(succeeded={n_success}, errored={n_errored}, "
        f"canceled={n_canceled}, expired={n_expired})"
    )
    if n_errored + n_canceled + n_expired == 0 and error_file.exists():
        error_file.unlink()
    elif error_file.exists():
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

            result = item.get("result") or {}
            rtype = result.get("type")
            if rtype != "succeeded":
                failures += 1
                updates[src_idx] = json.dumps(
                    {"batch_error": {"type": rtype, "details": result}},
                    ensure_ascii=False,
                )
                continue

            message = result.get("message")
            if message is None:
                failures += 1
                missing_response_body_indices.add(src_idx)
                updates[src_idx] = json.dumps(
                    {"batch_error": "Missing result.message"}, ensure_ascii=False
                )
                continue

            updates[src_idx] = json.dumps(message, ensure_ascii=False)

    return updates, seen_indices, missing_response_body_indices, failures


def extract_final_answer(raw_value: Any) -> str:
    """Pull text out of a serialized Anthropic Message payload. Concatenates
    every `text` block in `content` (Claude can interleave thinking blocks
    with text blocks, and we only want the user-visible text)."""
    payload = coerce_payload(raw_value)
    if payload is None:
        return ""

    content = payload.get("content")
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())

    return "\n".join(texts).strip()


def run_batches(
    client: anthropic.Anthropic,
    args: argparse.Namespace,
    pending_df: pd.DataFrame,
    batch_response_file: Path,
    batch_error_file: Path,
) -> Set[int]:
    """Submit one or more Anthropic batches (partitioned by args.batch_size)
    and concatenate their outputs into batch_response_file. Each chunk is
    submitted and awaited sequentially so only one batch is in flight at a
    time.

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
        requests, _ = build_batch_requests(args, chunk)
        batch_id = submit_batch(client, requests)
        print(f"{label} batch_id: {batch_id} ({len(chunk_indices)} rows)")

        final_batch = wait_for_batch(
            client, batch_id, poll_interval_seconds=args.poll_interval_seconds
        )
        final_status = getattr(final_batch, "processing_status", None)
        if final_status != "ended":
            raise RuntimeError(
                f"{label} ended with processing_status '{final_status}'. "
                f"batch_id={batch_id}"
            )

        write_batch_results(
            client,
            batch_id,
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

    client = anthropic.Anthropic()

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
