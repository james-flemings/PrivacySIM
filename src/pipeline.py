"""Orchestrate user simulation -> evaluation -> plots for configured models.

Supports four backends (selected via the `backend` field in configs/models.yaml):

- vllm   (default): dispatches sim to src/user_simulation.py
- openai          : dispatches sim to src/openai_user_simulation.py
- claude          : dispatches sim to src/claude_user_simulation.py
- gemini          : dispatches sim to src/gemini_user_simulation.py

Usage:
    python src/pipeline.py run --model qwen3.5_27B
    python src/pipeline.py run --model gpt-5.4-mini
    python src/pipeline.py run --model sonnet_4.6
    python src/pipeline.py run --model gemini_3_flash
    python src/pipeline.py run --model qwen3.5_27B --steps eval,plots
    python src/pipeline.py run --model qwen3.5_27B --limit 50    # smoke test
    python src/pipeline.py run --all --steps eval,plots
    python src/pipeline.py list

The simulation, evaluation, and plots scripts keep their own CLIs; this file
wires them together with subprocess so vLLM's CUDA context can fully release
between steps.

After every successful sim step, a run.json sidecar is written next to
outputs.csv with provenance metadata: git SHA, hostname, GPU info, package
versions, sampling params, and wall-clock timing. This makes it possible to
audit any historical artifact back to the exact code + environment that
produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
RUNS_DIR = REPO_ROOT / "data" / "runs"

STEP_ORDER = ("sim", "eval", "plots")

DEFAULT_BACKEND = "vllm"
SUPPORTED_BACKENDS = ("vllm", "openai", "claude", "gemini")
THEORIES = ("privacy_calculus", "bounded_rationality", "pmt")


def input_csv_path(theory: str | None) -> Path:
    if theory is None:
        return REPO_ROOT / "data" / "dataset_with_prompts.csv"
    if theory not in THEORIES:
        raise SystemExit(
            f"Unknown theory '{theory}'. Supported: {list(THEORIES)}"
        )
    return REPO_ROOT / "data" / f"dataset_with_prompts_{theory}.csv"

# Keys consumed by the orchestrator itself, not forwarded as CLI flags.
ENV_KEYS = {"cuda_visible_devices"}
ORCHESTRATOR_KEYS = {"backend", "env"}
# Keys that map to argparse bare-flag actions in src/user_simulation.py.
BOOL_FLAG_KEYS = {"trust_remote_code", "enforce_eager", "enable_expert_parallel"}


def load_models() -> dict[str, dict[str, Any]]:
    with CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f) or {}
    defaults_by_backend = raw.get("defaults") or {}
    models: dict[str, dict[str, Any]] = {}
    for name, overrides in (raw.get("models") or {}).items():
        overrides = overrides or {}
        backend = overrides.get("backend", DEFAULT_BACKEND)
        if backend not in SUPPORTED_BACKENDS:
            raise SystemExit(
                f"Model '{name}' has unsupported backend '{backend}'. "
                f"Supported: {list(SUPPORTED_BACKENDS)}"
            )
        merged = dict(defaults_by_backend.get(backend) or {})
        merged.update(overrides)
        merged["backend"] = backend
        models[name] = merged
    return models


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def domain_slug(domain: str | None) -> str | None:
    if not domain:
        return None
    return domain.strip().lower().replace(" ", "_")


def theory_slug(theory: str | None) -> str | None:
    if not theory:
        return None
    if theory not in THEORIES:
        raise SystemExit(
            f"Unknown theory '{theory}'. Supported: {list(THEORIES)}"
        )
    return theory


def _suffix(slug: str | None) -> str:
    return f"__{slug}" if slug else ""


def _combined_suffix(
    theory: str | None, domain: str | None, limit: int | None = None
) -> str:
    """Order matters: theory → domain → limit. Keeps full-run artifacts under
    the unsuffixed path; smoke / domain / theory variants live alongside."""
    extra = f"__limit_{limit}" if limit is not None and limit > 0 else ""
    return _suffix(theory_slug(theory)) + _suffix(domain_slug(domain)) + extra


def sim_output_path(
    name: str,
    theory: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> Path:
    return run_dir(name) / f"outputs{_combined_suffix(theory, domain, limit)}.csv"


def eval_summary_path(
    name: str,
    theory: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> Path:
    return run_dir(name) / f"evaluation_summary{_combined_suffix(theory, domain, limit)}.csv"


def plots_dir(
    name: str,
    theory: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> Path:
    return run_dir(name) / f"plots{_combined_suffix(theory, domain, limit)}"


def plots_exist(
    name: str,
    theory: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> bool:
    d = plots_dir(name, theory=theory, domain=domain, limit=limit)
    return d.exists() and any(d.glob("*.png"))


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def _emit_flags(cmd: list[str], cfg: dict[str, Any], allowed: set[str] | None = None) -> None:
    """Append --snake-case flags for cfg entries, skipping orchestrator/env-only keys."""
    for key, value in cfg.items():
        if key in ENV_KEYS or key in ORCHESTRATOR_KEYS or value is None:
            continue
        if allowed is not None and key not in allowed:
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOL_FLAG_KEYS:
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])


def build_vllm_sim_command(
    name: str,
    cfg: dict[str, Any],
    domain: str | None = None,
    theory: str | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    if cfg.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])

    extra_env = cfg.get("env")
    if extra_env is not None:
        if not isinstance(extra_env, dict):
            raise SystemExit(
                f"Model '{name}' has non-dict 'env' entry in configs/models.yaml; "
                f"expected mapping of env-var name to value, got {type(extra_env).__name__}."
            )
        for k, v in extra_env.items():
            env[str(k)] = str(v)

    cmd: list[str] = [
        sys.executable,
        "src/user_simulation.py",
        "--input-csv", rel(input_csv_path(theory)),
        "--output-csv", rel(sim_output_path(name, theory=theory, domain=domain, limit=limit)),
    ]
    if domain:
        cmd.extend(["--domain", domain])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    _emit_flags(cmd, cfg)
    return cmd, env


# Per-backend script + the cfg keys forwarded as CLI flags. vllm is handled
# separately because it forwards every cfg key (no allowlist) and supports
# --domain / extra env-var injection.
CLOSED_SIM_BACKENDS: dict[str, tuple[str, set[str]]] = {
    "openai": (
        "src/openai_user_simulation.py",
        {
            "model", "prompt_column", "reasoning_effort", "reasoning_summary",
            "max_output_tokens", "completion_window", "poll_interval_seconds",
            "batch_size", "limit",
        },
    ),
    "claude": (
        "src/claude_user_simulation.py",
        {
            "model", "prompt_column", "max_tokens", "batch_size",
            "poll_interval_seconds", "thinking_type", "effort", "start_row",
            "limit",
        },
    ),
    "gemini": (
        "src/gemini_user_simulation.py",
        {
            "model", "prompt_column", "max_output_tokens", "temperature",
            "top_p", "thinking_budget", "batch_size", "poll_interval_seconds",
            "start_row", "limit",
        },
    ),
}


def build_closed_sim_command(
    name: str,
    cfg: dict[str, Any],
    backend: str,
    theory: str | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    script, allowed_keys = CLOSED_SIM_BACKENDS[backend]
    env = os.environ.copy()
    cfg = dict(cfg)
    if limit is not None:
        cfg["limit"] = limit
    cmd: list[str] = [
        sys.executable,
        script,
        "--input-csv", rel(input_csv_path(theory)),
        "--output-csv", rel(sim_output_path(name, theory=theory, limit=limit)),
    ]
    _emit_flags(cmd, cfg, allowed=allowed_keys)
    return cmd, env


def build_sim_command(
    name: str,
    cfg: dict[str, Any],
    domain: str | None = None,
    theory: str | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    backend = cfg.get("backend", DEFAULT_BACKEND)
    if backend in CLOSED_SIM_BACKENDS:
        return build_closed_sim_command(name, cfg, backend, theory=theory, limit=limit)
    return build_vllm_sim_command(name, cfg, domain=domain, theory=theory, limit=limit)


def build_eval_command(
    name: str,
    domain: str | None = None,
    theory: str | None = None,
    limit: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        "src/evaluation.py",
        "--input-csv", rel(sim_output_path(name, theory=theory, domain=domain, limit=limit)),
        "--summary-csv", rel(eval_summary_path(name, theory=theory, domain=domain, limit=limit)),
    ]


def build_plots_command(
    name: str,
    domain: str | None = None,
    theory: str | None = None,
    limit: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        "src/plots.py",
        "--input-csv", rel(eval_summary_path(name, theory=theory, domain=domain, limit=limit)),
        "--output-dir", rel(plots_dir(name, theory=theory, domain=domain, limit=limit)),
    ]


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True)


def _git_sha() -> str | None:
    """Best-effort current commit. Returns None if not a git repo or git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=5,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _gpu_info() -> list[str] | None:
    """Names of visible GPUs from nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if result.returncode != 0:
            return None
        names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
        return names or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    import importlib.metadata as md
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return versions


def write_run_metadata(
    name: str,
    cfg: dict[str, Any],
    sim_path: Path,
    started_at: float,
    finished_at: float,
    *,
    domain: str | None,
    theory: str | None,
    limit: int | None,
) -> None:
    """Write data/runs/<model>/run.json with provenance for the sim step. Path
    suffixing matches outputs.csv so theory / domain / limit variants are
    self-describing."""
    suffix = _combined_suffix(theory, domain, limit)
    sidecar_path = sim_path.parent / f"run{suffix}.json"
    backend = cfg.get("backend", DEFAULT_BACKEND)
    payload: dict[str, Any] = {
        "model_alias": name,
        "backend": backend,
        "model_id": cfg.get("model"),
        "wall_clock": {
            "started_at_utc": dt.datetime.fromtimestamp(started_at, dt.timezone.utc).isoformat(),
            "finished_at_utc": dt.datetime.fromtimestamp(finished_at, dt.timezone.utc).isoformat(),
            "elapsed_seconds": round(finished_at - started_at, 2),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "gpus": _gpu_info(),
        },
        "git": {"sha": _git_sha(), "dirty": _git_dirty()},
        "package_versions": _package_versions(
            ["vllm", "torch", "anthropic", "openai", "google-genai",
             "huggingface-hub", "pandas", "numpy"]
        ),
        "config": {k: v for k, v in cfg.items() if k != "env"},
        "qualifiers": {"domain": domain, "theory": theory, "limit": limit},
        "outputs": {"sim_csv": str(sim_path.relative_to(REPO_ROOT))},
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote run metadata: {sidecar_path.relative_to(REPO_ROOT)}")


def run_pipeline(
    name: str,
    cfg: dict[str, Any],
    steps: list[str],
    domain: str | None = None,
    theory: str | None = None,
    limit: int | None = None,
) -> None:
    backend = cfg.get("backend", DEFAULT_BACKEND)
    if backend in ("openai", "claude", "gemini") and domain:
        raise SystemExit(
            f"--domain is not supported for {backend} models; omit it and run over the full dataset."
        )

    qualifiers = []
    if theory:
        qualifiers.append(f"theory={theory}")
    if domain:
        qualifiers.append(f"domain={domain}")
    if limit is not None:
        qualifiers.append(f"limit={limit}")
    label = f"{name} [{backend}]"
    if qualifiers:
        label = f"{name} [{backend}, {', '.join(qualifiers)}]"
    print(f"\n=== {label}: {','.join(steps)} ===")
    if "sim" in steps:
        cmd, env = build_sim_command(name, cfg, domain=domain, theory=theory, limit=limit)
        started_at = time.time()
        run_command(cmd, env=env)
        write_run_metadata(
            name, cfg,
            sim_path=sim_output_path(name, theory=theory, domain=domain, limit=limit),
            started_at=started_at, finished_at=time.time(),
            domain=domain, theory=theory, limit=limit,
        )
    if "eval" in steps:
        run_command(build_eval_command(name, domain=domain, theory=theory, limit=limit))
    if "plots" in steps:
        run_command(build_plots_command(name, domain=domain, theory=theory, limit=limit))


def parse_steps(raw: str) -> list[str]:
    requested = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STEP_ORDER]
    if unknown:
        raise SystemExit(
            f"Unknown steps: {unknown}. Allowed: {list(STEP_ORDER)}"
        )
    return [s for s in STEP_ORDER if s in requested]


def cmd_list(models: dict[str, dict[str, Any]]) -> None:
    name_w = max(len("model"), *(len(n) for n in models)) if models else len("model")
    backend_w = max(len("backend"), *(len(m.get("backend", DEFAULT_BACKEND)) for m in models.values())) if models else len("backend")
    theory_w = max(len("theory"), *(len(t) for t in THEORIES))
    print(
        f"{'model':<{name_w}}  {'backend':<{backend_w}}  {'theory':<{theory_w}}  "
        f"{'sim':>4}  {'eval':>4}  {'plots':>5}"
    )
    print("-" * (name_w + 2 + backend_w + 2 + theory_w + 2 + 4 + 2 + 4 + 2 + 5))
    for name, cfg in models.items():
        backend = cfg.get("backend", DEFAULT_BACKEND)
        for theory in (None, *THEORIES):
            sim = "yes" if sim_output_path(name, theory=theory).exists() else "-"
            ev = "yes" if eval_summary_path(name, theory=theory).exists() else "-"
            pl = "yes" if plots_exist(name, theory=theory) else "-"
            print(
                f"{name:<{name_w}}  {backend:<{backend_w}}  "
                f"{(theory or 'baseline'):<{theory_w}}  "
                f"{sim:>4}  {ev:>4}  {pl:>5}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sim -> eval -> plots pipeline for configured models."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run pipeline for one or all models.")
    target = run_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Model name as defined in configs/models.yaml.")
    target.add_argument("--all", action="store_true", help="Run all configured models.")
    run_parser.add_argument(
        "--steps",
        default="sim,eval,plots",
        help="Comma-separated subset of sim,eval,plots (default: all three).",
    )
    run_parser.add_argument(
        "--domain",
        default=None,
        help=(
            "Restrict the run to a single domain value from dataset_with_prompts.csv "
            "(e.g., 'chatbot group chat'). Writes to suffixed paths "
            "(outputs__<slug>.csv, evaluation_summary__<slug>.csv, plots__<slug>/) "
            "so the full-dataset artifacts are preserved. "
            "Not supported for OpenAI models."
        ),
    )
    run_parser.add_argument(
        "--theory",
        choices=THEORIES,
        default=None,
        help=(
            "Use a privacy-theory variant of the prompts dataset "
            "(data/dataset_with_prompts_<theory>.csv). Outputs are written to "
            "theory-suffixed paths (outputs__<theory>.csv, etc.) so baseline "
            "artifacts are preserved. Default: baseline (no theory)."
        ),
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Smoke-test mode: cap the run to the first N rows of the prompts CSV. "
            "Outputs are written to limit-suffixed paths "
            "(outputs__limit_N.csv, evaluation_summary__limit_N.csv, "
            "plots__limit_N/) so full-dataset artifacts are preserved. "
            "Useful for verifying the sim->eval->plots pipeline without "
            "committing to a multi-hour run."
        ),
    )

    sub.add_parser("list", help="List configured models and which outputs exist.")

    args = parser.parse_args()
    models = load_models()

    if args.command == "list":
        cmd_list(models)
        return

    steps = parse_steps(args.steps)
    if args.all:
        targets = list(models)
    else:
        if args.model not in models:
            raise SystemExit(
                f"Unknown model '{args.model}'. Configured: {list(models)}"
            )
        targets = [args.model]

    for name in targets:
        run_pipeline(
            name,
            models[name],
            steps,
            domain=args.domain,
            theory=args.theory,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
