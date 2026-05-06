"""Run one Google Gemini Batch API inference end-to-end over a prompts CSV.

Submits the batch via the google-genai SDK, polls to completion, downloads the
result JSONL, maps each response back to its source row, and writes an
`outputs.csv` whose `final_answer` column is already populated by extracting
the response text from the Gemini batch result. This mirrors the contract of
`src/user_simulation.py` (vLLM), `src/openai_user_simulation.py` (OpenAI),
and `src/claude_user_simulation.py` (Anthropic), so `src/pipeline.py` can
dispatch to any of the four based on the model's `backend` field in
`configs/models.yaml`.

Authentication: requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the env; the
google-genai client picks it up automatically.

Side artifacts written next to --output-csv:
    batch_response.jsonl  - combined batch result JSONL (one entry per row)
    batch_errors.jsonl    - subset of results with non-empty error
    missing_rows.json     - rows that were submitted but did not get a response

Resume behavior: each new chunk's output is written to
`<run_dir>/batch_artifacts/batch_response_part_<i>.jsonl`. On rerun, every
row whose `key` (`row-<src_idx>`) already appears in any cached part file is
filtered out of pending_df before partitioning, so no row is resubmitted.
This is content-keyed, not index-keyed, which means changing --batch-size
between runs is safe (new chunks are numbered after the highest existing
part index, never colliding). To force a fresh run, delete `batch_artifacts/`
(or specific part files) before invoking.

Example:
    python src/gemini_user_simulation.py \\
        --input-csv data/dataset_with_prompts.csv \\
        --output-csv data/runs/gemini_3_flash/outputs.csv \\
        --model gemini-3-flash-preview \\
        --max-output-tokens 8192 \\
        --thinking-budget -1 \\
        --batch-size 5000 \\
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
from google import genai

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


# google-genai 1.x emits JOB_STATE_* (per types.JobState). PARTIALLY_SUCCEEDED is
# terminal — per-row errors appear in the output JSONL — so we accept it as a
# success-equivalent and let the parser surface individual failures.
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Google Gemini Batch API inference over CSV prompts and write outputs.csv."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompt-column", type=str, default="llm_prompt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=(
            "Thinking token budget for reasoning-capable models (Gemini 2.5+). "
            "Use -1 for dynamic thinking, 0 to disable, or a positive cap. "
            "Omit for non-thinking models."
        ),
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Max rows per Gemini batch job. If set and smaller than the number "
            "of pending prompts, the dataset is partitioned into multiple "
            "sequential batches (each submitted and waited on before the next)."
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


def build_request_body(args: argparse.Namespace, prompt: str) -> dict:
    """Build the per-request body for one Gemini Batch API entry.

    Field names follow the REST API schema (camelCase), not the SDK Python
    schema (snake_case). The batch JSONL input is consumed directly by the
    Gemini service, which expects `generationConfig`, `maxOutputTokens`,
    `topP`, `thinkingConfig`, `thinkingBudget`. Snake-case keys are silently
    ignored by the API, which would cause generation parameters to default.
    """
    generation_config: dict = {"maxOutputTokens": args.max_output_tokens}
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.top_p is not None:
        generation_config["topP"] = args.top_p
    if args.thinking_budget is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": args.thinking_budget}

    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }


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
    jsonl_path = batch_dir / f"gemini_batch_input_{ts}{part_suffix}.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in pending_df_chunk.iterrows():
            src_idx = int(row[SOURCE_INDEX_COLUMN])
            prompt = str(row[args.prompt_column]).strip()
            request_obj = {
                "key": f"row-{src_idx}",
                "request": build_request_body(args, prompt),
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    return jsonl_path, submitted_indices


def submit_batch(
    client: genai.Client, args: argparse.Namespace, jsonl_path: Path
) -> str:
    """Upload the JSONL via the Files API, then create a batch job over it."""
    uploaded = client.files.upload(
        file=str(jsonl_path),
        config={
            "display_name": jsonl_path.name,
            "mime_type": "application/jsonl",
        },
    )
    batch = client.batches.create(model=args.model, src=uploaded.name)
    return batch.name


def _state_str(batch) -> str:
    state = getattr(batch, "state", None)
    if state is None:
        return "UNKNOWN"
    name = getattr(state, "name", None)
    return name if isinstance(name, str) else str(state)


def wait_for_batch(client: genai.Client, batch_name: str, poll_interval_seconds: int):
    while True:
        batch = client.batches.get(name=batch_name)
        state = _state_str(batch)
        print(f"Batch {batch_name} state: {state}")
        if state in TERMINAL_STATES:
            return batch
        time.sleep(poll_interval_seconds)


def _serialize_inline_entry(entry) -> dict:
    """Serialize an inlined response entry to a JSON-serializable dict.

    The entry is a Pydantic-style object with .key, .response, .error attrs;
    we round-trip via .model_dump_json() when available to flatten nested
    proto types into plain JSON.
    """
    if isinstance(entry, dict):
        return entry
    obj: dict = {}
    if hasattr(entry, "key"):
        obj["key"] = entry.key
    response = getattr(entry, "response", None)
    if response is not None:
        try:
            obj["response"] = json.loads(response.model_dump_json())
        except AttributeError:
            obj["response"] = response if isinstance(response, dict) else str(response)
    error = getattr(entry, "error", None)
    if error is not None:
        try:
            obj["error"] = json.loads(error.model_dump_json())
        except AttributeError:
            obj["error"] = error if isinstance(error, dict) else str(error)
    return obj


def write_batch_results(
    client: genai.Client,
    batch,
    response_file: Path,
    error_file: Path,
) -> None:
    """Write every result to response_file as JSONL; copy errored entries to
    error_file. Handles both inlined responses (small batches) and file-based
    results (large batches)."""
    response_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    dest = getattr(batch, "dest", None)
    if dest is None:
        raise RuntimeError(
            f"Batch {getattr(batch, 'name', '?')} completed but has no .dest"
        )

    inlined_responses = getattr(dest, "inlined_responses", None)
    file_name = getattr(dest, "file_name", None)

    n_success = n_errored = 0
    with response_file.open("w", encoding="utf-8") as f_resp, error_file.open(
        "w", encoding="utf-8"
    ) as f_err:
        if inlined_responses:
            for entry in inlined_responses:
                obj = _serialize_inline_entry(entry)
                line = json.dumps(obj, ensure_ascii=False) + "\n"
                f_resp.write(line)
                if obj.get("error"):
                    n_errored += 1
                    f_err.write(line)
                else:
                    n_success += 1
        elif file_name:
            content = client.files.download(file=file_name)
            text = (
                content.decode("utf-8")
                if isinstance(content, (bytes, bytearray))
                else str(content)
            )
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n_errored += 1
                    f_err.write(line + "\n")
                    f_resp.write(line + "\n")
                    continue
                f_resp.write(line + "\n")
                if obj.get("error"):
                    n_errored += 1
                    f_err.write(line + "\n")
                else:
                    n_success += 1
        else:
            raise RuntimeError(
                f"Batch {getattr(batch, 'name', '?')} has neither inlined_responses nor file_name"
            )

    print(
        f"Saved batch response JSONL to {response_file} "
        f"(succeeded={n_success}, errored={n_errored})"
    )
    if n_errored == 0 and error_file.exists():
        error_file.unlink()
    elif error_file.exists():
        print(f"Saved batch error JSONL to {error_file}")


def _part_idx(part_file: Path) -> int:
    """Extract numeric index from batch_response_part_<i>.jsonl /
    batch_errors_part_<i>.jsonl. Returns -1 if the suffix isn't an int."""
    try:
        return int(part_file.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def scan_completed_indices(artifacts_dir: Path) -> Set[int]:
    """Source row indices already present in any batch_response_part_*.jsonl
    under artifacts_dir. Lets main() drop already-processed rows from
    pending_df before partitioning, so changing batch_size between runs is
    safe (no duplicate API submissions). Errored responses count as completed
    (they were submitted and the failure is already encoded in the part file).
    Malformed JSON lines are skipped silently."""
    completed: Set[int] = set()
    if not artifacts_dir.exists():
        return completed
    for part_file in artifacts_dir.glob("batch_response_part_*.jsonl"):
        try:
            with part_file.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src_idx = parse_source_index(item.get("key"))
                    if src_idx is not None:
                        completed.add(src_idx)
        except OSError:
            continue
    return completed


def next_part_index(artifacts_dir: Path) -> int:
    """One past the highest existing batch_response_part_<i>.jsonl index, or 1
    if none exist. New chunks pick up from here so they don't collide with
    cached part files written under a previous batch_size."""
    if not artifacts_dir.exists():
        return 1
    max_idx = 0
    for part_file in artifacts_dir.glob("batch_response_part_*.jsonl"):
        idx = _part_idx(part_file)
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


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

            src_idx = parse_source_index(item.get("key"))
            if src_idx is None:
                continue

            seen_indices.add(src_idx)

            error = item.get("error")
            if error:
                failures += 1
                updates[src_idx] = json.dumps(
                    {"batch_error": error}, ensure_ascii=False
                )
                continue

            response = item.get("response")
            if response is None:
                failures += 1
                missing_response_body_indices.add(src_idx)
                updates[src_idx] = json.dumps(
                    {"batch_error": "Missing response"}, ensure_ascii=False
                )
                continue

            updates[src_idx] = json.dumps(response, ensure_ascii=False)

    return updates, seen_indices, missing_response_body_indices, failures


def extract_final_answer(raw_value: Any) -> str:
    """Pull text out of a Gemini generateContent response payload. Concatenates
    every text part from the first candidate's content (Gemini may interleave
    thinking parts with text parts; we only want user-visible text)."""
    payload = coerce_payload(raw_value)
    if payload is None:
        return ""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            # Skip thought parts; only collect output text.
            if part.get("thought"):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            return "\n".join(texts).strip()

    return ""


def run_batches(
    client: genai.Client,
    args: argparse.Namespace,
    pending_df: pd.DataFrame,
    batch_response_file: Path,
    batch_error_file: Path,
) -> Set[int]:
    """Submit Gemini batches for the rows in pending_df, then rebuild the
    combined batch_response_file from all part files in artifacts_dir.

    Resume model: pending_df is content-filtered upstream (in main(), via
    scan_completed_indices) so it contains only rows with no cached response.
    New chunks are numbered starting at next_part_index(artifacts_dir), so
    they never collide with part files written under a previous batch_size.
    """
    suffix = batch_response_file.stem.removeprefix("batch_response")
    artifacts_dir = batch_response_file.parent / f"batch_artifacts{suffix}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def part_response_path(i: int) -> Path:
        return artifacts_dir / f"batch_response_part_{i}.jsonl"

    def part_error_path(i: int) -> Path:
        return artifacts_dir / f"batch_errors_part_{i}.jsonl"

    all_submitted_indices: Set[int] = set()

    if pending_df.empty:
        print("No pending rows to submit; rebuilding combined files from cache.")
    else:
        chunks = partition_dataframe(pending_df, args.batch_size)
        n_new = len(chunks)
        start_idx = next_part_index(artifacts_dir)
        print(
            f"Partitioned {len(pending_df)} pending rows into {n_new} batch(es) "
            f"(batch_size={args.batch_size}); writing parts "
            f"{start_idx}..{start_idx + n_new - 1}"
        )

        for offset, chunk in enumerate(chunks):
            i = start_idx + offset
            label = f"new batch {offset + 1}/{n_new} (part {i})"
            part_response_file = part_response_path(i)
            part_error_file = part_error_path(i)

            chunk_indices = {to_int(v) for v in chunk[SOURCE_INDEX_COLUMN].tolist()}
            all_submitted_indices |= chunk_indices

            print(f"\n--- {label}: submitting {len(chunk)} rows ---")
            jsonl_path, _ = build_batch_jsonl(
                args, chunk, batch_response_file, part_index=i
            )
            batch_name = submit_batch(client, args, jsonl_path)
            print(f"{label} batch_name: {batch_name} ({len(chunk_indices)} rows)")
            print(f"{label} input file: {jsonl_path}")

            final_batch = wait_for_batch(
                client, batch_name, poll_interval_seconds=args.poll_interval_seconds
            )
            final_state = _state_str(final_batch)
            if final_state not in SUCCESS_STATES:
                raise RuntimeError(
                    f"{label} ended with state '{final_state}'. batch_name={batch_name}"
                )

            write_batch_results(
                client,
                final_batch,
                response_file=part_response_file,
                error_file=part_error_file,
            )

    # Rebuild combined files from every part file present in artifacts_dir
    # (cached + newly written), sorted by numeric part index.
    response_parts = sorted(
        artifacts_dir.glob("batch_response_part_*.jsonl"), key=_part_idx
    )
    error_parts = sorted(
        artifacts_dir.glob("batch_errors_part_*.jsonl"), key=_part_idx
    )

    batch_response_file.parent.mkdir(parents=True, exist_ok=True)
    batch_response_file.write_text("", encoding="utf-8")
    if batch_error_file.exists():
        batch_error_file.unlink()

    for part_file in response_parts:
        append_file(part_file, batch_response_file)
    for part_file in error_parts:
        append_file(part_file, batch_error_file)

    print(
        f"\nSaved combined batch response JSONL to {batch_response_file} "
        f"({len(response_parts)} part file(s) merged)"
    )
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

    batch_response_file, batch_error_file, missing_log_file, artifacts_dir = (
        derive_artifact_paths(args.output_csv)
    )

    completed_indices = scan_completed_indices(artifacts_dir)
    if completed_indices:
        before = len(pending_df)
        pending_df = pending_df[
            ~pending_df[SOURCE_INDEX_COLUMN].astype(int).isin(completed_indices)
        ]
        print(
            f"Resume: {len(completed_indices)} row(s) already cached under "
            f"{artifacts_dir}; pending_df {before} -> {len(pending_df)}."
        )

    client = genai.Client()  # picks up GEMINI_API_KEY / GOOGLE_API_KEY from env

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
