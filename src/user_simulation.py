"""Run offline vLLM inference over prompts from a CSV file.

Example:
    python src/user_simulation.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --input-csv data/dataset_with_prompts.csv \
        --output-csv data/runs/<model_name>/outputs.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Hashable, List, cast
import os

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.reasoning import ReasoningParserManager
from huggingface_hub import hf_hub_download

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "t"):
        return True
    if text in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return {"role": "system", "content": system_prompt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline vLLM generation for each prompt in a CSV file."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/dataset_with_prompts.csv"),
        help="Input CSV containing prompt text.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV with generated outputs (e.g., data/runs/<model>/outputs.csv).",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="llm_prompt",
        help="Column name containing prompts.",
    )
    parser.add_argument(
        "--thought-column",
        type=str,
        default="model_thoughts",
        help="Column name for model reasoning/thought content.",
    )
    parser.add_argument(
        "--final-answer-column",
        type=str,
        default="final_answer",
        help="Column name for model final answer content.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or local path for offline vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory for vLLM executor.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max context length override.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (for example: auto, float16, bfloat16).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="Optional KV cache dtype for vLLM (for example: auto, fp8).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional quantization mode for vLLM (for example: awq, gptq, fp8).",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Optional vLLM attention backend (e.g. TRITON_MLA, FLASHINFER, FLASH_ATTN).",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        required=True,
        help="Reasoning parser to use for vLLM (for example: qwen3, mistral).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote modeling code when loading model/tokenizer.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs and run in eager mode.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_const",
        const=True,
        default=None,
        help="Enable expert parallelism when provided; defaults to None.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=_str_to_bool,
        default=True,
        help="Pass enable_thinking to chat_template_kwargs / reasoning parser (default: True).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        help="Pass reasoning_effort to chat_template_kwargs (default: high).",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=-20,
        help="Top-k sampling cutoff (-1 disables top-k filtering).",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of prompts to generate per vLLM batch call.",
    )

    parser.add_argument(
        "--row-indices",
        type=str,
        default=None,
        help=(
            "Optional 0-based row indices/ranges to process, for example "
            "'0,2,10-20'. If set, this overrides --start-row/--limit."
        ),
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
        help="Optional cap on rows processed from the input.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help=(
            "Optional domain filter (case-insensitive match against the 'domain' "
            "column). When set, only rows in that domain are processed."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Max number of sequences to keep in memory for vLLM generation (adjust based on GPU memory and model size).",
    )

    return parser.parse_args()


def validate_inputs(df: pd.DataFrame, prompt_column: str) -> None:
    if prompt_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Prompt column '{prompt_column}' was not found. Available columns: {available}"
        )


def parse_row_indices(indices_arg: str, total_rows: int) -> List[int]:
    selected: List[int] = []

    for token in indices_arg.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{token}': end is smaller than start")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))

    if not selected:
        raise ValueError("No valid indices were provided in --row-indices")

    unique_sorted = sorted(set(selected))
    if unique_sorted[0] < 0 or unique_sorted[-1] >= total_rows:
        raise ValueError(
            f"--row-indices includes out-of-bounds values. Valid range is 0 to {total_rows - 1}."
        )

    return unique_sorted


def build_llm(args: argparse.Namespace) -> LLM:
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reasoning_parser": args.reasoning_parser,
        "trust_remote_code": args.trust_remote_code,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 2048,
        "enable_prefix_caching": True,
        "max_num_seqs": args.max_num_seqs,
    }

    model_name = args.model.lower()
    if "qwen3.5" in model_name:
        llm_kwargs["mamba_cache_mode"] = "align"
        llm_kwargs["mamba_block_size"] = 8
        llm_kwargs['language_model_only'] = True
        llm_kwargs['attention_backend'] = 'flashinfer'
        llm_kwargs['performance_mode'] = 'interactivity'

    if "nemotron" in model_name:
        llm_kwargs['swap_space'] = 0
        llm_kwargs['max_cudagraph_capture_size'] = 128
        llm_kwargs['mamba_ssm_cache_dtype'] = "float16"

    if args.enable_expert_parallel is not None:
        llm_kwargs["enable_expert_parallel"] = args.enable_expert_parallel
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    if args.dtype is not None:
        llm_kwargs["dtype"] = args.dtype
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend

    return LLM(**llm_kwargs)


def run_inference(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    llm = build_llm(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
        skip_special_tokens=False,
    )

    parser_cls = ReasoningParserManager.get_reasoning_parser(args.reasoning_parser)
    reasoning_parser = parser_cls(
        tokenizer=llm.get_tokenizer(),
        chat_kwargs={"enable_thinking": args.enable_thinking},
    )
    is_gptoss = args.reasoning_parser == "openai_gptoss"
    if is_gptoss:
        from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output

    valid_rows: List[tuple[Hashable, str]] = []
    for idx, value in df[args.prompt_column].items():
        prompt = "" if pd.isna(value) else str(value).strip()
        if prompt:
            valid_rows.append((idx, prompt))

    system_prompt_message = (
        load_system_prompt(args.model, "SYSTEM_PROMPT.txt")
        if "mistral" in args.model.lower()
        else None
    )

    total_batches = math.ceil(len(valid_rows) / args.batch_size) if valid_rows else 0
    for batch_start in tqdm(
        range(0, len(valid_rows), args.batch_size),
        total=total_batches,
        desc="Running inference",
        unit="batch",
    ):
        batch = valid_rows[batch_start: batch_start + args.batch_size]
        if system_prompt_message:
            messages = [
                [system_prompt_message, {"role": "user", "content": prompt}]
                for _, prompt in batch
            ]
        else:
            messages = [[{"role": "user", "content": prompt}] for _, prompt in batch]
        outputs = llm.chat(
            messages=cast(Any, messages),
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": args.enable_thinking,
                                  'reasoning_effort': args.reasoning_effort},
        )

        for (row_idx, _), output in zip(batch, outputs):
            response = output.outputs[0] if output.outputs else None
            if response is None:
                thought, final_answer = "", ""
            elif is_gptoss:
                token_ids = list(getattr(response, "token_ids", []) or [])
                reasoning, content, _ = parse_chat_output(token_ids)
                thought = reasoning or ""
                final_answer = content or ""
            else:
                raw_text = getattr(response, "text", getattr(response, "content", "")) or ""
                reasoning, content = reasoning_parser.extract_reasoning(raw_text, request=None)
                thought = reasoning or ""
                final_answer = content or ""
            df.at[row_idx, args.thought_column] = thought
            df.at[row_idx, args.final_answer_column] = final_answer

        # Persist progress after each completed batch.
        df.to_csv(args.output_csv, index=False)

    return df


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    validate_inputs(df=df, prompt_column=args.prompt_column)

    if args.domain:
        if "domain" not in df.columns:
            raise ValueError("--domain was set but the input CSV has no 'domain' column.")
        target = args.domain.strip().lower()
        mask = df["domain"].astype(str).str.strip().str.lower() == target
        df = df[mask].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"No rows match --domain '{args.domain}'")

    if args.row_indices:
        selected_rows = parse_row_indices(args.row_indices, total_rows=len(df))
        df = df.iloc[selected_rows].copy()
    else:
        if args.start_row < 0:
            raise ValueError("--start-row must be >= 0")
        if args.limit is not None and args.limit < 0:
            raise ValueError("--limit must be >= 0")

        end_row = None if args.limit is None else args.start_row + args.limit
        df = df.iloc[args.start_row:end_row].copy()

    df[args.thought_column] = ""
    df[args.final_answer_column] = ""

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Create an initial checkpoint file before inference starts.
    df.to_csv(args.output_csv, index=False)

    prompts_processed = sum(
        1 for value in df[args.prompt_column] if not pd.isna(value) and str(value).strip()
    )

    df = run_inference(args=args, df=df)

    print(f"Processed {prompts_processed} prompts from {args.input_csv}")
    print(f"Saved outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
