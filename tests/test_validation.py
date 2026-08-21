"""Tests for blind sample generation and external QC comparison."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.validation import build_blind_sample, compare_qc_labels


def _report(path: Path, dataset: str, episodes: list[dict]) -> None:
    path.write_text(json.dumps({
        "dataset": {"path": f"/private/{dataset}", "num_episodes": len(episodes)},
        "episodes": episodes,
    }), encoding="utf-8")


def _episode(index: int, verdict: str, spikes: int, motion: float, hard_fail: bool = False) -> dict:
    metrics = {
        "action_discontinuity": {"availability": "available", "measurement": {"spike_count": spikes}, "passed": True},
        "idle_ratio": {"availability": "available", "measurement": {"effective_motion_ratio": motion}, "passed": True},
    }
    if hard_fail:
        metrics["invalid_values"] = {"availability": "available", "measurement": {"nan_count": 1}, "passed": False}
    return {"episode_index": index, "num_frames": 10, "verdict": verdict, "metrics": metrics}


def test_blind_sample_does_not_expose_rda_fields(tmp_path: Path):
    report = tmp_path / "demo.json"
    _report(report, "demo", [
        _episode(0, "EXCLUDE", 0, 0.9, hard_fail=True),
        _episode(1, "REVIEW", 100, 0.1),
        _episode(2, "PASS", 0, 0.9),
    ])
    annotator = tmp_path / "annotator.csv"
    private = tmp_path / "mapping.json"
    result = build_blind_sample([report], annotator, private, per_dataset=3, exclude_datasets=())

    text = annotator.read_text(encoding="utf-8")
    assert result["sample_count"] == 3
    assert "rda_verdict" not in text
    assert "risk_score" not in text
    assert "hard_fail" not in text
    assert "human_label" in text
    mapping = json.loads(private.read_text(encoding="utf-8"))
    assert {row["rda_verdict"] for row in mapping["samples"]} == {"PASS", "REVIEW", "EXCLUDE"}


def test_qc_comparison_requires_complete_labels_and_reports_proxies(tmp_path: Path):
    report = tmp_path / "demo.json"
    _report(report, "demo", [
        _episode(0, "EXCLUDE", 0, 0.9, hard_fail=True),
        _episode(1, "REVIEW", 100, 0.1),
        _episode(2, "PASS", 0, 0.9),
    ])
    annotator = tmp_path / "annotator.csv"
    private = tmp_path / "mapping.json"
    build_blind_sample([report], annotator, private, per_dataset=3, exclude_datasets=())

    rows = list(csv.DictReader(annotator.open(encoding="utf-8")))
    labels = ["REMOVE", "KEEP", "KEEP"]
    for row, label in zip(rows, labels):
        row["human_label"] = label
        row["review_seconds"] = "12"
    with annotator.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = compare_qc_labels(private, annotator)
    assert result["sample_count"] == 3
    assert result["binary_counts"] == {"TP": 1, "FP": 1, "FN": 0, "TN": 1}
    assert result["proxy_metrics"]["proxy_precision"] == 0.5
    assert result["proxy_metrics"]["proxy_recall"] == 1.0
    assert result["coverage"]["hard_fail_overlap_count"] == 1
    assert result["review_effort"]["total_review_seconds"] == 36.0


def test_qc_comparison_rejects_missing_labels(tmp_path: Path):
    report = tmp_path / "demo.json"
    _report(report, "demo", [_episode(0, "PASS", 0, 0.9)])
    annotator = tmp_path / "annotator.csv"
    private = tmp_path / "mapping.json"
    build_blind_sample([report], annotator, private, per_dataset=1, exclude_datasets=())
    with pytest.raises(ValueError, match="Missing QC labels"):
        compare_qc_labels(private, annotator)
