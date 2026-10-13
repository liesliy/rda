"""Tests for pii_scan — the privacy/PII exposure metric (v0.9.22).

These tests defend four things, in priority order:

1. **The report must not become the leak.** No raw identifier may ever reach
   ``measurement``, ``details`` or ``message``.
2. **No silent degradation.** When a detector cannot run, the episode must
   say so in ``not_scanned`` — never look like "scanned and clean".
3. **WARNING by default, EXCLUDE only when explicitly promoted.** Adding a
   metric that silently EXCLUDEs on a heuristic would be worse than not
   detecting anything at all.
4. **Precision over recall.** Lookalikes (``translation``, arbitrary
   18-digit runs, private IPs) must stay clean, or operators will learn to
   ignore the whole metric.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rda.audit.rules import AuditVerdict
from rda.io.schema import EpisodeData
from rda.metrics.base import MetricAvailability
from rda.metrics.pii_scan import (
    PIIScanMetric,
    _fingerprint,
    _idcard_valid,
    _ipv4_kind,
    _opencv_missing,
    _redact,
    _resolve_cascade,
    _token_is_hint,
    redact_text,
)

CN_ID = "11010519491231002X"      # canonical example that satisfies checksum
CN_MOBILE = "13812345678"
CN_EMAIL = "wusheng@niusutech.com"
CN_MAC = "00:1B:44:11:3A:B7"


def _ep(meta=None, observation=None, action=None, n=5):
    return EpisodeData(
        episode_index=0,
        num_frames=n,
        timestamps=np.arange(n, dtype=float),
        observation=observation if observation is not None else {"state": np.zeros((n, 3))},
        action=action if action is not None else {"position": np.zeros((n, 3))},
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# 1. Validators
# ---------------------------------------------------------------------------

class TestValidators:
    def test_idcard_checksum_rejects_lookalikes(self):
        """Mod-11-2 is what separates an ID from any 18-digit run."""
        assert _idcard_valid(CN_ID) is True
        assert _idcard_valid("110105194912310021") is False   # tampered last digit
        assert _idcard_valid("123456789012345678") is False   # sequential digits
        assert _idcard_valid("110105") is False               # too short

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("8.8.8.8", "public"),
            ("203.0.113.5", "public"),
            ("192.168.1.1", "private"),
            ("10.0.0.4", "private"),
            ("172.16.5.9", "private"),
            ("127.0.0.1", "private"),
            ("999.1.1.1", None),
            ("1.2.3", None),
        ],
    )
    def test_ipv4_kind(self, raw, expected):
        assert _ipv4_kind(raw) == expected

    def test_token_match_separates_coordinates_from_lookalikes(self):
        lat_hints = ("lat", "latitude")
        for name in ("lat", "latitude", "gps_lat", "robot_lat"):
            assert _token_is_hint(name, lat_hints), name
        # 'lat' appears inside these, but they are not coordinates
        for name in ("translation", "plate", "relation", "isolation"):
            assert not _token_is_hint(name, lat_hints), name

    def test_redaction_destroys_the_middle(self):
        masked = _redact(CN_MOBILE)
        assert masked != CN_MOBILE
        assert CN_MOBILE[3:-4] not in masked       # the revealed tail is gone
        assert masked.startswith("138") and masked.endswith("5678")
        assert _redact("ab") == "**"

    def test_fingerprint_is_stable_and_one_way(self):
        assert _fingerprint(CN_MOBILE) == _fingerprint(CN_MOBILE)
        assert _fingerprint(CN_MOBILE) != _fingerprint("13812345679")
        assert CN_MOBILE not in _fingerprint(CN_MOBILE)


# ---------------------------------------------------------------------------
# 2. Detection behaviour
# ---------------------------------------------------------------------------

class TestDetection:
    @pytest.mark.parametrize(
        "surface_value,expected_type",
        [
            ("contact me at 13812345678 please", "cn_mobile"),
            (f"id card {CN_ID} on file", "cn_resident_id"),
            (f"mail {CN_EMAIL}", "email"),
            (f"host mac {CN_MAC}", "mac_address"),
            ("edge node at 203.0.113.5", "ipv4"),
            (f"reachable on {CN_EMAIL} or 13812345678", "email"),
        ],
    )
    def test_each_detector_fires_on_a_real_hit(self, surface_value, expected_type):
        r = PIIScanMetric().compute(_ep(meta={"task_description": surface_value}))
        found = {f["type"] for f in r.details["findings"]}
        assert expected_type in found

    def test_clean_episode_is_silent(self):
        r = PIIScanMetric().compute(
            _ep(meta={"task_description": "pick up the red block and place it on shelf 3"})
        )
        assert r.measurement["finding_count"] == 0
        assert r.has_finding is False
        assert r.availability == MetricAvailability.AVAILABLE

    def test_precision_lookalikes_stay_clean(self):
        """False positives train operators to ignore the metric entirely."""
        cases = [
            _ep(observation={"translation": np.linspace(0.1, 0.2, 5),
                             "plate_pose": np.zeros((5, 3))}),
            _ep(meta={"trace": "123456789012345678"}),
            _ep(meta={"ip": "192.168.1.1"}),
            _ep(meta={"version": "1.2.3"}),
        ]
        for ep in cases:
            r = PIIScanMetric().compute(ep)
            # private IPs are still reported (topology exposure), the rest are not
            types = {f["type"] for f in r.details["findings"]}
            assert types <= {"ipv4"}, f"unexpected findings {types}"

    def test_geolocation_pair_is_reported_with_city_granularity(self):
        ep = _ep(observation={
            "gps_lat": np.linspace(30.25, 30.26, 5),
            "gps_lon": np.linspace(120.10, 120.11, 5),
        })
        r = PIIScanMetric().compute(ep)
        geo = [f for f in r.details["findings"] if f["type"].startswith("geolocation")]
        assert len(geo) == 1
        assert geo[0]["type"] == "geolocation_pair"
        # coordinates are reported rounded, never raw per-frame values
        assert "30.2" in geo[0]["evidence"]

    def test_gnss_feature_without_pair_is_a_weaker_signal(self):
        ep = _ep(observation={"gnss_signal": np.linspace(1.0, 2.0, 5)})
        r = PIIScanMetric().compute(ep)
        geo = [f for f in r.details["findings"] if f["type"] == "geolocation_feature"]
        assert len(geo) == 1
        assert geo[0]["confidence"] == "medium"


# ---------------------------------------------------------------------------
# 3. The report must not be the leak
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_raw_identifiers_never_appear_in_the_result(self):
        ep = _ep(meta={
            "task_description": f"operator reachable at {CN_MOBILE}",
            "contact": CN_EMAIL,
            "id_no": CN_ID,
            "mac": CN_MAC,
        })
        r = PIIScanMetric().compute(ep)
        blob = f"{r.measurement} {r.details} {r.message}"
        for secret in (CN_MOBILE, CN_EMAIL, CN_ID, CN_MAC):
            assert secret not in blob, f"LEAK: {secret} present in result"

    def test_evidence_carries_a_fingerprint_for_dedupe(self):
        ep = _ep(meta={"task_description": CN_MOBILE})
        r = PIIScanMetric().compute(ep)
        hit = r.details["findings"][0]
        assert len(hit["detail"]["fingerprint"]) == 12
        assert hit["detail"]["fingerprint"] == _fingerprint(CN_MOBILE)


# ---------------------------------------------------------------------------
# 4. Verdict policy — WARNING by default, EXCLUDE only when promoted
# ---------------------------------------------------------------------------

class TestVerdictPolicy:
    def _dirty(self):
        return _ep(meta={"task_description": f"contact {CN_MOBILE}"})

    def test_default_mode_flags_without_changing_verdict(self):
        from rda.audit.rules import AuditVerdict, classify_episode

        r = PIIScanMetric().compute(self._dirty())
        assert r.has_finding is True
        assert r.assessment["status"] == "pass"   # flagged, not failed
        assert classify_episode([r]) == AuditVerdict.PASS

    def test_strict_mode_promotes_the_finding_to_exclude(self):
        from rda.audit.rules import AuditVerdict, classify_episode, critical_with

        r = PIIScanMetric().compute(self._dirty())
        assert classify_episode([r], critical_metrics=critical_with(["pii_scan"])) == (
            AuditVerdict.EXCLUDE
        )

    def test_critical_with_does_not_mutate_the_default_list(self):
        from rda.audit.rules import CRITICAL_METRICS, critical_with

        extended = critical_with(["pii_scan"])
        assert "pii_scan" not in CRITICAL_METRICS   # must stay opt-in
        assert extended[-1] == "pii_scan"
        assert len(critical_with(["pii_scan", "pii_scan"])) == len(extended)

    def test_pii_scan_is_not_critical_by_default(self):
        from rda.audit.rules import CRITICAL_METRICS, DIAGNOSTIC_METRICS

        assert "pii_scan" not in CRITICAL_METRICS
        assert "pii_scan" not in DIAGNOSTIC_METRICS

    def test_episode_auditor_honours_the_critical_override(self):
        from rda.audit.episode_audit import EpisodeAuditor
        from rda.audit.rules import AuditVerdict, critical_with

        ep = self._dirty()
        default = EpisodeAuditor(metrics=[PIIScanMetric()]).audit(ep)
        strict = EpisodeAuditor(
            metrics=[PIIScanMetric()], critical_metrics=critical_with(["pii_scan"])
        ).audit(ep)
        assert default.verdict == AuditVerdict.PASS
        assert strict.verdict == AuditVerdict.EXCLUDE
        assert strict.exclude_reasons and "pii_scan" in strict.exclude_reasons[0]


# ---------------------------------------------------------------------------
# 5. Loud degradation — never "no findings" when we could not look
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_video_is_marked_unscanned_by_default(self):
        r = PIIScanMetric().compute(_ep())
        assert "video_not_scanned_pass_pii_video_to_enable" in r.details["not_scanned"]
        assert any("--pii-video" in h for h in r.details["not_scanned_detail"])

    def test_opencv_absent_is_recorded_not_hidden(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "cv2", None)
        assert _opencv_missing() == "opencv_missing"

        r = PIIScanMetric(scan_video=True).compute(
            _ep(meta={"task_description": CN_MOBILE})
        )
        assert "opencv_missing" in r.details["not_scanned"]
        # the text scan must still have run
        assert r.measurement["finding_count"] >= 1
        assert any("opencv-python-headless" in h for h in r.details["not_scanned_detail"])

    def test_opencv_installed_without_cascade_is_recorded(self, monkeypatch):
        """OpenCV 5.x ships no Haar cascades — must degrade loudly."""
        import rda.metrics.pii_scan as mod

        monkeypatch.setattr(mod, "_resolve_cascade", lambda: None)
        assert _opencv_missing() == "haar_cascade_missing"

        r = PIIScanMetric(scan_video=True).compute(_ep())
        assert "haar_cascade_missing" in r.details["not_scanned"]
        assert any("OpenCV 5.x" in h for h in r.details["not_scanned_detail"])

    def test_env_override_can_supply_the_cascade(self, monkeypatch, tmp_path):
        fake = tmp_path / "haarcascade_frontalface_default.xml"
        fake.write_text("<stub/>", encoding="utf-8")
        monkeypatch.setenv("RDA_HAAR_CASCADE", str(fake))
        assert _resolve_cascade() == fake

    def test_missing_video_file_is_reported_per_camera(self, tmp_path):
        import rda.metrics.pii_scan as mod

        real_av = True
        try:
            import av  # noqa: F401
        except ImportError:
            real_av = False

        monkeypatch_free_available = _opencv_missing() is None
        if not (real_av and monkeypatch_free_available):
            pytest.skip("av + opencv cascade required for this path")

        ep = _ep(meta={
            "dataset_root": str(tmp_path),
            "fps": 30,
            "video_features": {
                "observation.images.top": {
                    "chunk_index": 0, "file_index": 0,
                    "from_timestamp": 0.0, "to_timestamp": 1.0,
                    "episode_index": 0,
                }
            },
        })
        findings, not_scanned = mod._face_findings(ep, 3)
        assert findings == []
        assert any(code.startswith("video_not_found") for code in not_scanned)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_report_serializes_with_numpy_scalars(self):
        """Regression: ``_NumpySafeEncoder`` referenced ``np`` without importing it.

        Any metric that puts a numpy scalar into ``details`` (e.g.
        ``temporal_structure``'s ``idle_prefix_frames``:int64) made the whole
        JSON report explode with ``NameError: name 'np' is not defined`` —
        a latent bug that only shows up on real data.
        """
        from rda.report.json_report import _NumpySafeEncoder

        payload = {"n": np.int64(3), "f": np.float64(0.5), "b": np.bool_(True)}
        blob = json.dumps(payload, cls=_NumpySafeEncoder)
        assert json.loads(blob)["n"] == 3

    def test_pii_measurements_are_plain_python_types(self):
        """pii_scan must not rely on the encoder to rescue it."""
        ep = _ep(observation={
            "gps_lat": np.linspace(30.25, 30.26, 5),
            "gps_lon": np.linspace(120.10, 120.11, 5),
        })
        r = PIIScanMetric().compute(ep)
        json.dumps(r.details, cls=json.JSONEncoder)  # plain encoder must suffice


# ---------------------------------------------------------------------------
# 7. Tier interaction & real-video false-positive guard
# ---------------------------------------------------------------------------

class TestTierInteraction:
    def test_pii_scan_still_runs_with_no_video_tier(self):
        """Text PII survives --no-video; only the face scan is pointless there."""
        from rda.audit.execution_tier import ExecutionTier, filter_metrics_by_tier

        allowed = filter_metrics_by_tier({"pii_scan", "idle_ratio"}, ExecutionTier.NO_VIDEO)
        assert "pii_scan" in allowed

    def test_pii_scan_excluded_from_video_only_tier(self):
        from rda.audit.execution_tier import ExecutionTier, filter_metrics_by_tier

        allowed = filter_metrics_by_tier({"pii_scan", "video_freeze"}, ExecutionTier.VIDEO_ONLY)
        assert "pii_scan" not in allowed

    @pytest.mark.skipif(
        _opencv_missing() is not None,
        reason="requires opencv with a Haar cascade (opencv-python-headless<5)",
    )
    def test_no_false_faces_on_real_robot_video(self):
        """Real arm-camera footage must not trip the face detector."""
        import os
        from pathlib import Path

        from rda.io import load_lerobot_dataset

        root = Path(os.environ.get(
            "RDA_CLEAN_DATASET", "D:/workbuddy-data/datasets/rda_inject_clean"))
        if not root.exists():
            pytest.skip("fixture dataset not available")

        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

        metric = PIIScanMetric(scan_video=True)
        episodes = list(iter_episodes(str(root), max_episodes=2))
        if not episodes:
            pytest.skip("no episodes loaded")
        for ep in episodes:
            r = metric.compute(ep)
            faces = [f for f in r.details["findings"] if f["type"] == "face_in_video"]
            assert faces == [], f"false positive face detection: {faces}"
            # Guard against a vacuous pass: if the video could not be read
            # the test below proves nothing about false positives.
            assert r.measurement["video_scanned"] is True, (
                f"video was not actually decoded: {r.details['not_scanned']}"
            )
            assert r.details["not_scanned"] == [], r.details["not_scanned"]


# ===========================================================================
# 8. Report-side redaction — the report must not re-emit what it detected
# ===========================================================================

class TestReportRedaction:
    """The tool must not leak through the channel it audits.

    Before v0.9.22 the JSON report echoed ``task_description`` verbatim —
    the richest operator-authored surface in the whole dataset — so a run
    that flagged three identifiers emitted all three of them in cleartext
    one layer further down. Detecting a leak and printing it in the same
    document moves the leak; it does not close it.
    """

    DIRTY_TASK = (
        f"operator reachable at {CN_MOBILE} or {CN_EMAIL}, "
        f"employee id {CN_ID}, pick up the red block"
    )

    def _ep_result(self, task_text=None, index=0):
        from types import SimpleNamespace

        return SimpleNamespace(
            episode_index=index,
            num_frames=100,
            verdict=AuditVerdict.PASS,
            metrics={},
            task_index=0,
            task_description=task_text,
            exclude_reasons=[],
        )

    def test_redact_text_replaces_each_type(self):
        out, hits = redact_text(self.DIRTY_TASK)
        for secret in (CN_MOBILE, CN_EMAIL, CN_ID):
            assert secret not in out
        assert hits["cn_mobile"] == 1
        assert hits["email"] == 1
        assert hits["cn_resident_id"] == 1
        # Task semantics survive; only the identifiers are removed.
        assert "pick up the red block" in out
        assert "[REDACTED:cn_mobile]" in out

    def test_redact_text_is_a_no_op_on_clean_text(self):
        clean = "pick up the red block and place it on shelf 3"
        out, hits = redact_text(clean)
        assert out == clean
        assert hits == {}

    def test_redact_text_is_deterministic(self):
        a, hits_a = redact_text(self.DIRTY_TASK)
        b, hits_b = redact_text(self.DIRTY_TASK)
        assert a == b and hits_a == hits_b

    def test_redact_text_covers_mac_and_ipv4(self):
        out, hits = redact_text(f"backup {CN_MAC} at 8.8.8.8")
        assert CN_MAC not in out and "8.8.8.8" not in out
        assert hits["mac_address"] == 1 and hits["ipv4"] == 1

    def test_text_beyond_scan_budget_is_not_echoed(self):
        """No un-scanned tail: text past the budget is dropped, not emitted."""
        filler = "x" * 100_000
        out, hits = redact_text(filler + CN_MOBILE)
        assert CN_MOBILE not in out
        assert hits == {}                       # the hit lives past the budget
        assert "beyond scan budget" in out      # and the omission is stated

    def test_engine_format_report_carries_no_raw_identifier(self):
        from rda.report.json_report import _episode_result_to_dict

        d = _episode_result_to_dict(self._ep_result(self.DIRTY_TASK))
        blob = json.dumps(d)
        for secret in (CN_MOBILE, CN_EMAIL, CN_ID):
            assert secret not in blob
        assert d["task_description_redacted"] is True
        assert "pick up the red block" in d["task_description"]

    def test_clean_task_flag_is_false(self):
        from rda.report.json_report import _episode_result_to_dict

        d = _episode_result_to_dict(self._ep_result("pick up the red block"))
        assert d["task_description_redacted"] is False
        assert d["task_description"] == "pick up the red block"

    def test_issue_code_is_mapped_for_the_p0_dimension(self):
        """MVP Spec §5.1 code must exist, or the issue never reaches the user."""
        from rda.report.json_report import _issue_code_for_metric

        assert _issue_code_for_metric("pii_scan") == "PRV-01"

    def test_absent_task_stays_none_not_empty_string(self):
        """Backward compat: a missing field must still serialize as null."""
        from rda.report.json_report import _episode_result_to_dict

        d = _episode_result_to_dict(self._ep_result(None))
        assert d["task_description"] is None
        assert d["task_description_redacted"] is False

    def test_mvp_format_report_carries_no_raw_identifier(self):
        from types import SimpleNamespace

        from rda.audit.dataset_audit import DatasetAuditResult
        from rda.audit.rules import AuditVerdict
        from rda.io.schema import DatasetInfo
        from rda.report.json_report import generate_episode_report

        info = DatasetInfo(path="fixture", num_episodes=1, total_frames=100,
                           modalities=[], meta={})
        result = DatasetAuditResult(
            dataset_info=info,
            episodes={0: self._ep_result(self.DIRTY_TASK)},
            verdict_counts={AuditVerdict.PASS: 1},
        )
        report = generate_episode_report(result, 0)
        blob = json.dumps(report)
        for secret in (CN_MOBILE, CN_EMAIL, CN_ID):
            assert secret not in blob
        assert report["task_description_redacted"] is True

    def test_dataset_task_map_is_redacted(self):
        from rda.report.json_report import _safe_task_map

        mapped = _safe_task_map({0: self.DIRTY_TASK, 1: "pick up the mug"})
        assert CN_MOBILE not in mapped[0]
        assert "pick up the mug" == mapped[1]

    def test_blind_report_drops_free_text_entirely(self):
        """`--blind` is documented as externally shareable.

        Detector redaction cannot recognise arbitrary names or home
        addresses, so blind mode does not rely on it — every task string
        becomes a stable hash stub.
        """
        from rda.report.json_report import anonymize_report

        report = {
            "dataset": {"path": "/data/secret", "tasks": {0: self.DIRTY_TASK}},
            "episodes": [
                {"episode_index": 0, "task_description": self.DIRTY_TASK},
                {"episode_index": 1, "task_description": "pick up the mug"},
            ],
        }
        anon = anonymize_report(report)
        blob = json.dumps(anon)
        for secret in (CN_MOBILE, CN_EMAIL, CN_ID, "/data/secret"):
            assert secret not in blob
        ep0, ep1 = anon["episodes"]
        assert ep0["task_description"].startswith("redacted:")
        # Even *unflagged* free text is dropped in blind mode.
        assert ep1["task_description"].startswith("redacted:")
        assert ep1["task_description"] != "pick up the mug"
        # Equal text maps to an equal stub, so tasks stay groupable.
        assert ep0["task_description"] == anon["dataset"]["tasks"][0]

    def test_metric_and_report_disagree_on_nothing(self):
        """What the metric flags is exactly what the report hides."""
        ep = _ep(meta={"task_description": self.DIRTY_TASK})
        flagged = {f["type"] for f in PIIScanMetric().compute(ep).details["findings"]}
        _, hits = redact_text(self.DIRTY_TASK)
        # Every type the report hides must be one the metric would flag —
        # otherwise the report is censoring things it never reported.
        for t in hits:
            assert t in flagged, f"report hid {t} but the metric never flagged it"
        # And everything the metric flagged here is hidden by the report.
        assert not (flagged & {"geolocation_features", "face_in_video"})
