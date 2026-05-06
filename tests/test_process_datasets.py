"""Smoke tests for the dataset-processing helpers.

These don't process the real datasets (which require curl-downloaded files
and would be too slow for a CI smoke run). Instead they validate that the
pure helper functions in scripts/process_datasets.py behave correctly given
synthetic inputs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_process_datasets():
    """scripts/ is not a package; load process_datasets.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "process_datasets", REPO_ROOT / "scripts" / "process_datasets.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["process_datasets"] = module
    spec.loader.exec_module(module)
    return module


def test_module_imports():
    pd_mod = _load_process_datasets()
    assert callable(pd_mod.process_llm_chatbot)
    assert callable(pd_mod.process_llm_health)
    assert callable(pd_mod.process_agent_permissions)
    assert callable(pd_mod.process_chatbot_group_chat)
    assert callable(pd_mod.process_llm_conversational_agent)
    assert callable(pd_mod.finalize_dataset)
    assert callable(pd_mod.gen_response_prompts)


def test_output_columns_includes_preferences():
    pd_mod = _load_process_datasets()
    assert "preferences" in pd_mod.OUTPUT_COLUMNS
    assert "responses" in pd_mod.OUTPUT_COLUMNS


def test_is_informative_previous_experiences():
    pd_mod = _load_process_datasets()
    fn = pd_mod._is_informative_previous_experiences
    # All-skipped dict → not informative
    assert fn("{'q1': '(Skipped)', 'q2': 'nan'}") is False
    # At least one real value → informative
    assert fn("{'q1': '(Skipped)', 'q2': 'Yes'}") is True
    # Empty / nan → not informative
    assert fn("") is False


def test_powerset_enumerates_facets():
    pd_mod = _load_process_datasets()
    cols = ["demographics", "previous_experiences", "privacy_attitudes"]
    sets = pd_mod._powerset_columns(cols)
    # Empty set + 3 singletons + 3 pairs + 1 triple = 8
    assert len(sets) == 8
    assert () in sets
    assert tuple(cols) in sets


def test_theory_lenses_exist():
    pd_mod = _load_process_datasets()
    assert set(pd_mod.THEORY_LENSES) == {
        "privacy_calculus", "bounded_rationality", "pmt"
    }
    for body in pd_mod.THEORY_LENSES.values():
        assert body.startswith("PRIVACY DECISION LENS")


def test_uuid5_is_deterministic():
    pd_mod = _load_process_datasets()
    row = {"user_id": 7, "domain": "AI agent permissions"}
    a = pd_mod._row_uuid5(row)
    b = pd_mod._row_uuid5(row)
    assert a == b
    # Different domain → different uuid
    row["domain"] = "LLM chatbot"
    assert pd_mod._row_uuid5(row) != a
