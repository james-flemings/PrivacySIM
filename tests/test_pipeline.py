"""Smoke tests for src/pipeline.py command-builder.

These tests don't run any models — they just check that the orchestrator
produces well-formed CLI invocations for each backend, that --domain /
--theory / --limit plumb through correctly, and that load_models reads the
YAML config without error.
"""
from __future__ import annotations

import pipeline


def test_load_models_reads_config():
    """configs/models.yaml parses, every model has a known backend, and
    defaults are merged into per-model overrides."""
    models = pipeline.load_models()
    assert models, "configs/models.yaml has no models"
    for name, cfg in models.items():
        assert "backend" in cfg, f"{name} missing backend after merge"
        assert cfg["backend"] in pipeline.SUPPORTED_BACKENDS, (
            f"{name} has unsupported backend: {cfg['backend']}"
        )
        assert "model" in cfg, f"{name} missing model id"


def test_build_sim_command_vllm_baseline():
    models = pipeline.load_models()
    name, cfg = next(
        (n, c) for n, c in models.items() if c["backend"] == "vllm"
    )
    cmd, env = pipeline.build_sim_command(name, cfg)
    assert "src/user_simulation.py" in cmd
    # Defaults are forwarded
    assert "--temperature" in cmd
    assert "--reasoning-parser" in cmd
    # Output goes to baseline (no suffix)
    assert "data/runs/" in " ".join(cmd)
    assert "outputs.csv" in " ".join(cmd)
    assert "outputs__" not in " ".join(cmd)


def test_build_sim_command_with_limit_uses_suffix():
    models = pipeline.load_models()
    name, cfg = next(
        (n, c) for n, c in models.items() if c["backend"] == "vllm"
    )
    cmd, env = pipeline.build_sim_command(name, cfg, limit=42)
    flat = " ".join(cmd)
    assert "--limit 42" in flat
    assert "outputs__limit_42.csv" in flat


def test_build_sim_command_with_theory_and_limit_stacks_suffixes():
    models = pipeline.load_models()
    name, cfg = next(
        (n, c) for n, c in models.items() if c["backend"] == "vllm"
    )
    cmd, env = pipeline.build_sim_command(name, cfg, theory="pmt", limit=10)
    flat = " ".join(cmd)
    assert "data/dataset_with_prompts_pmt.csv" in flat
    assert "outputs__pmt__limit_10.csv" in flat


def test_build_sim_command_each_closed_backend():
    """Each closed backend dispatches to the right script + forwards --limit."""
    models = pipeline.load_models()
    for backend in ("openai", "claude", "gemini"):
        match = next(
            ((n, c) for n, c in models.items() if c["backend"] == backend),
            None,
        )
        if match is None:
            continue
        name, cfg = match
        cmd, _ = pipeline.build_sim_command(name, cfg, limit=5)
        flat = " ".join(cmd)
        assert f"src/{backend}_user_simulation.py" in flat
        assert "--limit 5" in flat


def test_build_eval_and_plots_commands_carry_suffix():
    eval_cmd = pipeline.build_eval_command("gemma4_32B", limit=3)
    plots_cmd = pipeline.build_plots_command("gemma4_32B", limit=3)
    assert "outputs__limit_3.csv" in " ".join(eval_cmd)
    assert "evaluation_summary__limit_3.csv" in " ".join(eval_cmd)
    assert "plots__limit_3" in " ".join(plots_cmd)


def test_combined_suffix_order():
    """Suffix order is theory → domain → limit, separators consistent."""
    assert pipeline._combined_suffix(None, None, None) == ""
    assert pipeline._combined_suffix("pmt", None, None) == "__pmt"
    assert pipeline._combined_suffix(None, "Chatbot Group Chat", None) == "__chatbot_group_chat"
    assert pipeline._combined_suffix(None, None, 50) == "__limit_50"
    assert pipeline._combined_suffix("pmt", "LLM chatbot", 50) == "__pmt__llm_chatbot__limit_50"
