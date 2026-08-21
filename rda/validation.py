"""Blind validation helpers for comparing RDA with external QC labels.

This module deliberately keeps sample selection metadata separate from the
annotator-facing manifest. RDA verdicts and metric values must not be exposed
before a human or customer QC process has produced its labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HARD_FAIL_METRICS = {
    "missing_dropout",
    "invalid_values",
    "schema_consistency",
    "timestamp_validity",
    "joint_limit",
}
QC_LABELS = {"KEEP", "REVIEW", "REMOVE"}
RDA_VERDICTS = {"PASS", "REVIEW", "EXCLUDE"}


def _read_report(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError(f"Unsupported RDA JSON report: {path}")
    return data


def _metric_measurement(episode: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metric = episode.get("metrics", {}).get(name, {})
    measurement = metric.get("measurement", {})
    return measurement if isinstance(measurement, dict) else {}


def _has_hard_fail(episode: Mapping[str, Any]) -> bool:
    for name in HARD_FAIL_METRICS:
        metric = episode.get("metrics", {}).get(name, {})
        if metric.get("availability") == "available" and metric.get("passed") is False:
            return True
    return False


def _risk_score(episode: Mapping[str, Any]) -> float:
    """Return a selection score, not a quality label.

    The score is used only to choose a manageable blind sample. It must not be
    shown to the annotator or interpreted as a ground-truth confidence.
    """
    score = 0.0
    spikes = _metric_measurement(episode, "action_discontinuity").get("spike_count", 0)
    motion = _metric_measurement(episode, "idle_ratio").get("effective_motion_ratio")
    accel = _metric_measurement(episode, "velocity_acceleration").get("acceleration_spikes", 0)
    try:
        score += min(float(spikes) / 25.0, 20.0)
    except (TypeError, ValueError):
        pass
    try:
        if motion is not None:
            score += max(0.0, (0.5 - float(motion)) * 20.0)
    except (TypeError, ValueError):
        pass
    try:
        score += min(float(accel) / 10.0, 10.0)
    except (TypeError, ValueError):
        pass
    return round(score, 6)


def _episode_records(report_paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for report_path in sorted(Path(p) for p in report_paths):
        report = _read_report(report_path)
        dataset = report.get("dataset", {})
        dataset_name = Path(str(dataset.get("path", report_path.stem))).name or report_path.stem
        for episode in report["episodes"]:
            if not isinstance(episode, dict):
                continue
            if int(episode.get("num_frames", 0) or 0) <= 0:
                continue
            verdict = str(episode.get("verdict", "")).upper()
            if verdict not in RDA_VERDICTS:
                continue
            records.append({
                "source_report": str(report_path),
                "dataset_name": dataset_name,
                "dataset_path": str(dataset.get("path", "")),
                "episode_index": int(episode["episode_index"]),
                "rda_verdict": verdict,
                "hard_fail": _has_hard_fail(episode),
                "risk_score": _risk_score(episode),
                "risk_signal": _risk_score(episode) > 0,
            })
    return records


def _stable_sample(records: List[Dict[str, Any]], count: int, salt: str) -> List[Dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda r: hashlib.sha256(
            f"{salt}:{r['source_report']}:{r['episode_index']}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:count]


def build_blind_sample(
    report_paths: Sequence[str | Path],
    output_csv: str | Path,
    private_mapping_json: str | Path,
    *,
    per_dataset: int = 4,
    exclude_datasets: Iterable[str] = ("bridge_sample",),
) -> Dict[str, Any]:
    """Build an annotator-safe sample CSV and a private RDA mapping JSON.

    Sampling is stratified per dataset: one available hard-fail example, one
    high-risk example, one lower-risk control, then deterministic fill. The
    returned mapping is private and contains the RDA verdict; the CSV does not.
    """
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    excluded = set(exclude_datasets)
    all_records = [r for r in _episode_records(report_paths) if r["dataset_name"] not in excluded]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        grouped[record["dataset_name"]].append(record)

    selected: List[Dict[str, Any]] = []
    for dataset_name in sorted(grouped):
        pool = grouped[dataset_name]
        chosen: List[Dict[str, Any]] = []
        hard = [r for r in pool if r["hard_fail"]]
        risk = sorted(
            [r for r in pool if not r["hard_fail"]],
            key=lambda r: (r["risk_score"], r["episode_index"]),
            reverse=True,
        )
        controls = sorted(
            [r for r in pool if not r["hard_fail"]],
            key=lambda r: (r["risk_score"], r["episode_index"]),
        )
        for candidate in (hard[:1] + risk[:1] + controls[:1]):
            if candidate not in chosen:
                chosen.append(candidate)
        remaining = [r for r in pool if r not in chosen]
        chosen.extend(_stable_sample(remaining, max(0, per_dataset - len(chosen)), dataset_name))
        selected.extend(chosen[:per_dataset])

    selected.sort(key=lambda r: (r["dataset_name"], r["episode_index"]))
    mapping: List[Dict[str, Any]] = []
    rows: List[Dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        sample_id = f"S{index:04d}"
        mapping.append({"sample_id": sample_id, **record})
        rows.append({
            "sample_id": sample_id,
            "dataset_alias": record["dataset_name"],
            "dataset_path": record["dataset_path"],
            "episode_index": str(record["episode_index"]),
            "human_label": "",
            "reviewer_id": "",
            "review_seconds": "",
            "notes": "",
        })

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "sample_id", "dataset_alias", "dataset_path", "episode_index",
            "human_label", "reviewer_id", "review_seconds", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)

    mapping_path = Path(private_mapping_json)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps({
            "schema_version": "blind-validation-v1",
            "annotator_file": str(output_csv_path),
            "selection_policy": {
                "per_dataset": per_dataset,
                "excluded_datasets": sorted(excluded),
                "hard_fail_and_risk_controls": True,
            },
            "samples": mapping,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "sample_count": len(rows),
        "dataset_count": len(grouped),
        "annotator_csv": str(output_csv_path),
        "private_mapping_json": str(mapping_path),
    }


def _normalise_qc_label(value: str) -> str:
    label = str(value or "").strip().upper()
    aliases = {"GOOD": "KEEP", "BAD": "REMOVE", "DROP": "REMOVE", "EXCLUDE": "REMOVE"}
    label = aliases.get(label, label)
    if label not in QC_LABELS:
        raise ValueError(f"QC label must be one of {sorted(QC_LABELS)}: {value!r}")
    return label


def compare_qc_labels(
    private_mapping_json: str | Path,
    labels_csv: str | Path,
) -> Dict[str, Any]:
    """Compare external QC labels with RDA using an explicit proxy definition.

    ``proxy_precision`` and ``proxy_recall`` are against the customer's QC
    process, not independently verified ground truth. They must not be
    presented as scientific precision/recall without a validated label source.
    """
    mapping_data = json.loads(Path(private_mapping_json).read_text(encoding="utf-8"))
    mapping = {str(row["sample_id"]): row for row in mapping_data.get("samples", [])}
    labels: Dict[str, Dict[str, str]] = {}
    with Path(labels_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id not in mapping:
                raise ValueError(f"Unknown sample_id in labels CSV: {sample_id!r}")
            if sample_id in labels:
                raise ValueError(f"Duplicate sample_id in labels CSV: {sample_id!r}")
            raw_label = str(row.get("human_label", "")).strip()
            if not raw_label:
                raise ValueError(f"Missing QC labels for sample_id: {sample_id!r}")
            labels[sample_id] = {
                "human_label": _normalise_qc_label(raw_label),
                "review_seconds": str(row.get("review_seconds", "")).strip(),
            }

    missing = sorted(set(mapping) - set(labels))
    if missing:
        raise ValueError(f"Missing QC labels for {len(missing)} samples; first: {missing[:5]}")

    matrix: Dict[str, Dict[str, int]] = {
        qc: {rda: 0 for rda in ("PASS", "REVIEW", "EXCLUDE")}
        for qc in ("KEEP", "REVIEW", "REMOVE")
    }
    binary = Counter(TP=0, FP=0, FN=0, TN=0)
    hard_fail_overlap = 0
    qc_remove = 0
    qc_keep = 0
    qc_keep_rda_review = 0
    review_seconds: List[float] = []
    for sample_id, qc_row in labels.items():
        record = mapping[sample_id]
        qc = qc_row["human_label"]
        rda = record["rda_verdict"]
        matrix[qc][rda] += 1
        qc_bad = qc in {"REVIEW", "REMOVE"}
        rda_flagged = rda in {"REVIEW", "EXCLUDE"}
        if qc_bad and rda_flagged:
            binary["TP"] += 1
        elif not qc_bad and rda_flagged:
            binary["FP"] += 1
        elif qc_bad and not rda_flagged:
            binary["FN"] += 1
        else:
            binary["TN"] += 1
        if qc == "REMOVE":
            qc_remove += 1
            if rda == "EXCLUDE":
                hard_fail_overlap += 1
        if qc == "KEEP":
            qc_keep += 1
            if rda == "REVIEW":
                qc_keep_rda_review += 1
        if qc_row["review_seconds"]:
            review_seconds.append(float(qc_row["review_seconds"]))

    total = len(labels)
    denom = lambda n: round(n / total, 6) if total else None
    flagged = binary["TP"] + binary["FP"]
    bad = binary["TP"] + binary["FN"]
    return {
        "schema_version": "qc-comparison-v1",
        "sample_count": total,
        "confusion_matrix_qc_rows_rda_columns": matrix,
        "binary_definition": {
            "qc_bad": ["REVIEW", "REMOVE"],
            "rda_flagged": ["REVIEW", "EXCLUDE"],
            "note": "QC is an external process baseline; without independent ground truth these are proxy metrics.",
        },
        "proxy_metrics": {
            "agreement_rate_exact_3way": round(sum(matrix[label][label_map] for label, label_map in (("KEEP", "PASS"), ("REVIEW", "REVIEW"), ("REMOVE", "EXCLUDE"))) / total, 6) if total else None,
            "proxy_precision": round(binary["TP"] / flagged, 6) if flagged else None,
            "proxy_recall": round(binary["TP"] / bad, 6) if bad else None,
            "qc_keep_to_rda_review_rate": round(qc_keep_rda_review / qc_keep, 6) if qc_keep else None,
            "rda_exclude_overlap_with_qc_remove": round(hard_fail_overlap / qc_remove, 6) if qc_remove else None,
        },
        "binary_counts": dict(binary),
        "review_effort": {
            "labeled_samples_with_seconds": len(review_seconds),
            "total_review_seconds": round(sum(review_seconds), 3),
            "mean_review_seconds": round(sum(review_seconds) / len(review_seconds), 3) if review_seconds else None,
        },
        "coverage": {
            "qc_label_counts": dict(Counter(row["human_label"] for row in labels.values())),
            "rda_verdict_counts": dict(Counter(mapping[sid]["rda_verdict"] for sid in labels)),
            "hard_fail_overlap_count": hard_fail_overlap,
        },
    }
