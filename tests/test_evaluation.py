"""Smoke tests for src/evaluation.py.

Builds a tiny synthetic outputs.csv with known-good user / model answer pairs
and asserts that run_evaluation produces:
- a valid summary CSV with the expected columns
- per-row alignment / Spearman / tolerance numbers in the right ranges
- the pooled ALL_DOMAINS row aggregates correctly
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

import evaluation as ev


def _write_outputs_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "uuid", "user_id", "domain", "prompt_type", "responses", "final_answer",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_single_row_perfect_match():
    """All Q-keys agree → alignment=100, Spearman well-defined, tol=4 matches."""
    user_raw = "{'Q1': 'Somewhat acceptable', 'Q2': 'Completely unacceptable', 'Q3': 'Neutral', 'Q4': 'Somewhat unacceptable'}"
    model_raw = '{"Q1": "Somewhat acceptable", "Q2": "Completely unacceptable", "Q3": "Neutral", "Q4": "Somewhat unacceptable"}'
    res = ev.evaluate_single_row("LLM healthcare consultation", user_raw, model_raw)
    assert res.compared_questions == 4
    assert res.alignment_score == pytest.approx(100.0)
    assert res.tolerance_match_count == 4
    assert res.spearman_rank_correlation == pytest.approx(1.0)


def test_evaluate_single_row_max_disagreement():
    """Opposite ends of the 5-point scale → alignment=0."""
    user_raw = "{'Q1': 'Completely acceptable'}"
    model_raw = '{"Q1": "Completely unacceptable"}'
    res = ev.evaluate_single_row("LLM healthcare consultation", user_raw, model_raw)
    assert res.compared_questions == 1
    assert res.alignment_score == pytest.approx(0.0)
    assert res.tolerance_match_count == 0


def test_evaluate_single_row_numeric_domain():
    """LLM chatbot uses 0..100 scale; tolerance at tau=0.15 is ±15 points."""
    user_raw = "{'Q1': 50, 'Q2': 100}"
    model_raw = '{"Q1": 60, "Q2": 100}'
    res = ev.evaluate_single_row("LLM chatbot", user_raw, model_raw)
    assert res.compared_questions == 2
    # Q1 off by 10 (< 15 tolerance), Q2 off by 0
    assert res.tolerance_match_count == 2
    # mean normalized error = (10+0)/2 / 100 = 0.05; alignment = 100*(1-0.05) = 95
    assert res.alignment_score == pytest.approx(95.0)


def test_run_evaluation_produces_expected_summary(tmp_path: Path):
    """End-to-end: 4 synthetic rows → summary CSV with the expected schema."""
    inp = tmp_path / "outputs.csv"
    out = tmp_path / "summary.csv"
    rows = [
        {
            "uuid": "u1", "user_id": 1,
            "domain": "AI agent permissions", "prompt_type": "none",
            "responses": "{'Q1': 'Yes, always share', 'Q2': 'No, never share'}",
            "final_answer": '{"Q1": "Yes, always share", "Q2": "No, never share"}',
        },
        {
            "uuid": "u2", "user_id": 2,
            "domain": "AI agent permissions", "prompt_type": "none",
            "responses": "{'Q1': 'Yes, always share', 'Q2': 'No, never share'}",
            "final_answer": '{"Q1": "No, never share", "Q2": "Yes, always share"}',
        },
        {
            "uuid": "u3", "user_id": 3,
            "domain": "Chatbot group chat", "prompt_type": "none",
            "responses": '{"Q1": "Strongly agree", "Q2": "Strongly disagree"}',
            "final_answer": '{"Q1": "Strongly agree", "Q2": "Strongly disagree"}',
        },
        {
            "uuid": "u4", "user_id": 4,
            "domain": "Chatbot group chat", "prompt_type": "demographics",
            "responses": '{"Q1": "Strongly agree", "Q2": "Strongly disagree"}',
            "final_answer": '{"Q1": "Strongly agree", "Q2": "Strongly disagree"}',
        },
    ]
    _write_outputs_csv(inp, rows)
    n_rows, with_match = ev.run_evaluation(input_csv=inp, summary_csv=out)
    assert n_rows == 4
    assert with_match == 4
    assert out.exists()

    with out.open() as f:
        summary = list(csv.DictReader(f))
    fields = {
        "prompt_type", "domain", "rows", "alignment_score",
        "spearman_rank_correlation", "tolerance_accuracy",
    }
    assert fields.issubset(summary[0].keys())

    # ALL_DOMAINS pool exists and pools across the 3-row no-prompt subset
    pooled = [r for r in summary if r["domain"] == "ALL_DOMAINS"]
    assert {r["prompt_type"] for r in pooled} == {"none", "demographics"}


def test_prompt_type_sort_key_order():
    keys = ["demographics+previous_experiences", "none", "previous_experiences", "demographics"]
    keys.sort(key=ev.prompt_type_sort_key)
    # 'none' first (rank 0), then single facets (rank 1), then doubles (rank 2)
    assert keys[0] == "none"
    assert keys[-1] == "demographics+previous_experiences"


def test_compute_spearman_handles_constant():
    """All-tied user values → NaN Spearman."""
    rho = ev.compute_spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert math.isnan(rho)


def test_compute_spearman_perfect():
    rho = ev.compute_spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert rho == pytest.approx(1.0)
