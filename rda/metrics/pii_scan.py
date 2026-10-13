"""Privacy / PII exposure scan (P2-12, v0.9.22).

Scans a robot dataset episode for personally identifiable information and
other privacy-sensitive exposure, covering YD/T 6771-2026's 安全性 dimension
(the only P0 admission gate among the eight quality dimensions).

Scope of this metric — what it does and does NOT claim:

  It detects **objective, reproducible evidence of exposure**:
    - Direct identifiers in text surfaces (Chinese mobile numbers, Chinese
      resident ID numbers validated by their embedded mod-11-2 checksum,
      email addresses, MAC addresses, IPv4 addresses).
    - Precise geolocation recorded as numeric features (lat/lon/GNSS/UTM).
    - Human faces present in sampled video frames (requires opencv).

  It does NOT claim the dataset is legally compliant, that consent was
  obtained, or that every face in every frame was found. It is a
  pre-delivery triage tool that produces **items a human must review**.

Verdict semantics — deliberately WARNING-level by default:

  Like ``numeric_sanity``, this metric never changes the episode verdict on
  its own. A false-positive face detection must not silently discard good
  data. Findings are exposed as ``has_finding=True`` with
  ``assessment.status="pass"``, so they are visible in the report and in the
  UI but do not affect PASS/REVIEW/EXCLUDE.

  To use it as an acceptance gate, run with ``--pii-strict``: that promotes
  ``pii_scan`` into the critical set for the run, so findings then trigger
  EXCLUDE. The choice stays with the operator, not with the metric.

Evidence handling — the report must not become the leak:

  A privacy report that prints the identifier it found has simply moved the
  problem. Every finding therefore carries a **redacted** snippet plus a
  SHA-256 fingerprint (first 12 hex chars) so recurring hits can be
  correlated across episodes without disclosing the value. Raw values are
  never written to ``measurement`` or ``details``.

Graceful degradation:

  Video face scanning requires both PyAV (decode) and opencv-python
  (cascade). Either missing records a reason in ``details["not_scanned"]``
  and the text scan still runs. A missing detector never becomes "clean".
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricAvailability, MetricBase, MetricResult

# --- Scan budget ---
_MAX_TEXT_PER_SURFACE = 100_000   # chars scanned per text surface (cost bound)
_VIDEO_SAMPLE_FRAMES = 5          # frames sampled per camera for face scan
_FACE_CONFIDENT_FRAMES = 2        # frames-with-face needed before reporting
_META_DEPTH = 4                   # recursion depth for nested meta values
_META_LIST_CAP = 50               # max list elements walked per meta list
_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# Geo coordinate hints in feature names
_LAT_HINTS = ("lat", "latitude")
_LON_HINTS = ("lon", "lng", "longitude")
_GEO_HINTS = ("gps", "gnss", "utm", "geolocation", "geo_")

_LAT_RANGE = (-90.0, 90.0)
_LON_RANGE = (-180.0, 180.0)


# ---------------------------------------------------------------------------
# Redaction — never echo the identifier back
# ---------------------------------------------------------------------------

def _fingerprint(raw: str) -> str:
    """Short stable fingerprint of the raw hit (for cross-episode dedupe)."""
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _redact(raw: str) -> str:
    """Mask the middle of a hit, keeping enough shape to recognise the type.

    Rules: keep first 3 + last 4 for long numeric identifiers (mobile/ID),
    first 2 + last 2 for everything else. Short hits are fully masked.
    """
    n = len(raw)
    if n <= 4:
        return "*" * n
    digits = raw.isdigit()
    head, tail = (3, 4) if (digits and n >= 11) else (2, 2)
    if head + tail >= n:
        return raw[0] + "*" * (n - 2) + raw[-1]
    return f"{raw[:head]}{'*' * (n - head - tail)}{raw[-tail:]}"


# ---------------------------------------------------------------------------
# Validators — anything beyond regex structure, to suppress false positives
# ---------------------------------------------------------------------------

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CHARS = "10X98765432"


def _idcard_valid(raw: str) -> bool:
    """Validate a Chinese resident ID number's mod-11-2 checksum.

    The checksum is what separates a real identifier from an 18-digit number
    that merely looks like one — without it any timestamp-like digit run is
    a false positive. Implementation follows GB 11643-1999.
    """
    s = raw.upper()
    if len(s) != 18:
        return False
    try:
        total = sum(int(s[i]) * _ID_WEIGHTS[i] for i in range(17))
    except ValueError:
        return False
    return _ID_CHECK_CHARS[total % 11] == s[17]


def _ipv4_kind(raw: str) -> str | None:
    """Classify an IPv4 literal, or reject it.

    Returns ``"public"`` for routable addresses (geolocatable — real
    exposure), ``"private"`` for RFC1918/loopback/link-local (network
    topology exposure, lower severity), ``None`` if not a valid address.
    """
    parts = raw.split(".")
    if len(parts) != 4:
        return None
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return None
    if not all(0 <= o <= 255 for o in octets):
        return None
    if octets[0] == 127 or raw in ("0.0.0.0", "255.255.255.255"):
        return "private"
    if octets[0] == 10:
        return "private"
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return "private"
    if octets[0] == 192 and octets[1] == 168:
        return "private"
    if octets[0] == 169 and octets[1] == 254:
        return "private"
    return "public"


def _id_kind(raw: str) -> str | None:
    return "idcard" if _idcard_valid(raw) else None


def _mobile_kind(raw: str) -> str | None:
    return "mobile" if len(raw) == 11 and "3" <= raw[1] <= "9" else None


def _always(raw: str) -> str | None:  # noqa: ARG001
    return "match"


# Detector table: (name, regex, validator -> Optional[kind], confidence, severity)
_TEXT_DETECTORS: List[Tuple[str, "re.Pattern[str]", Callable[[str], Optional[str]], str, str]] = [
    ("cn_resident_id",
     re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
                r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
     _id_kind, "high", "high"),
    ("cn_mobile",
     re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
     _mobile_kind, "high", "high"),
    # NOTE: the domain tail uses ``*`` before the TLD group — several anchor
    # groups would require two dots and silently miss single-level domains
    # such as ``name@company.com``.
    ("email",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
     _always, "high", "medium"),
    ("mac_address",
     re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])"),
     _always, "medium", "low"),
    ("ipv4",
     re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
     _ipv4_kind, "medium", "medium"),
]


def _valid_hits(text: str) -> List[Tuple[int, int, str]]:
    """Non-overlapping ``(start, end, detector_name)`` spans that passed
    their validator, left-to-right. Overlaps keep the longer span."""
    spans: List[Tuple[int, int, str]] = []
    for dname, pattern, validate, _conf, _sev in _TEXT_DETECTORS:
        for m in pattern.finditer(text[:_MAX_TEXT_PER_SURFACE]):
            if validate(m.group(0)) is None:
                continue
            spans.append((m.start(), m.end(), dname))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: List[Tuple[int, int, str]] = []
    last_end = -1
    for start, end, dname in spans:
        if start < last_end:
            continue
        kept.append((start, end, dname))
        last_end = end
    return kept


def redact_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Replace every detected identifier in free text with a typed token.

    This is the **report-side** counterpart of the scan. Detecting an
    identifier and then printing it in the same document does not close the
    leak — it moves it. Every free-text field the report echoes back
    (task descriptions, the task map, any future notes field) therefore goes
    through this first.

    The replacement loses the value but keeps the type, which is what a
    reader actually needs::

        "operator reachable at 13812345678"
        → "operator reachable at [REDACTED:cn_mobile]"

    Uses the same detector table and the same validators as the metric, so
    what the report hides is exactly what the report flags — no divergence
    between the two.

    Args:
        text: Free text that may contain identifiers.

    Returns:
        ``(redacted_text, hits_by_detector)`` where the dict maps detector
        name to occurrence count (empty when nothing matched). Text beyond
        :data:`_MAX_TEXT_PER_SURFACE` is not echoed unredacted; it is
        replaced by an explicit truncation marker.
    """
    if not text:
        return text, {}

    body = text[:_MAX_TEXT_PER_SURFACE]
    hits = _valid_hits(body)
    hits_by_type: Dict[str, int] = {}
    pieces: List[str] = []
    cursor = 0
    for start, end, dname in hits:
        pieces.append(body[cursor:start])
        pieces.append(f"[REDACTED:{dname}]")
        hits_by_type[dname] = hits_by_type.get(dname, 0) + 1
        cursor = end
    pieces.append(body[cursor:])
    out = "".join(pieces)

    # Nothing past the budget is echoed: scanning stopped there, so the tail
    # is *unknown* — printing it would defeat the point of redacting.
    unscanned = len(text) - len(body)
    if unscanned > 0:
        out += f"[{unscanned} further characters omitted: beyond scan budget]"
    return out, hits_by_type


# ---------------------------------------------------------------------------
# Text surface enumeration
# ---------------------------------------------------------------------------

def _iter_meta_strings(
    obj: Any, prefix: str, depth: int = 0
) -> Iterable[Tuple[str, str]]:
    """Yield ``(surface_name, text)`` for every string inside episode meta."""
    if depth > _META_DEPTH:
        return
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_meta_strings(v, f"{prefix}.{k}", depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:_META_LIST_CAP]):
            yield from _iter_meta_strings(v, f"{prefix}[{i}]", depth + 1)


def _collect_text_surfaces(episode: EpisodeData) -> List[Tuple[str, str]]:
    """All text surfaces worth scanning, with their provenance labels."""
    surfaces: List[Tuple[str, str]] = []
    meta = episode.meta or {}
    for key, text in _iter_meta_strings(meta, "meta"):
        surfaces.append((key, text[:_MAX_TEXT_PER_SURFACE]))
    # Feature names ship with the dataset and can embed operator names.
    for kind, mapping in (("observation", episode.observation),
                          ("action", episode.action)):
        for name in (mapping or {}):
            surfaces.append((f"{kind}.{name}", str(name)))
    return surfaces


# ---------------------------------------------------------------------------
# Geo detection
# ---------------------------------------------------------------------------

def _numeric_features(episode: EpisodeData) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for kind, mapping in (("observation", episode.observation),
                          ("action", episode.action)):
        for name, arr in (mapping or {}).items():
            if isinstance(arr, np.ndarray) and arr.size and arr.dtype.kind in "fiu":
                out[f"{kind}.{name}"] = arr
    return out


def _token_is_hint(base: str, hints: Tuple[str, ...]) -> bool:
    """True when any delimited token of a feature name equals a hint.

    Whole-token matching, not substring: ``translation`` / ``plate`` /
    ``relation`` all contain ``lat`` but are not coordinates, while
    ``gps_lat`` and ``robot.lon`` must still be recognised.
    """
    return any(tok in hints for tok in re.split(r"[^a-z0-9]+", base) if tok)


def _geo_findings(episode: EpisodeData) -> List[Dict[str, Any]]:
    """Detect precise geolocation recorded as numeric features."""
    feats = _numeric_features(episode)
    lat_keys: List[str] = []
    lon_keys: List[str] = []
    geo_keys: List[str] = []
    for key in feats:
        base = key.lower().split(".")[-1]
        if any(h in base for h in _GEO_HINTS):
            geo_keys.append(key)
        if _token_is_hint(base, _LAT_HINTS):
            lat_keys.append(key)
        if _token_is_hint(base, _LON_HINTS):
            lon_keys.append(key)

    findings: List[Dict[str, Any]] = []
    if lat_keys and lon_keys:
        lats = np.concatenate([feats[k].astype(float).ravel() for k in lat_keys])
        lons = np.concatenate([feats[k].astype(float).ravel() for k in lon_keys])
        lats = lats[np.isfinite(lats)]
        lons = lons[np.isfinite(lons)]
        if lats.size and lons.size:
            lat_in = float(np.mean((lats >= _LAT_RANGE[0]) & (lats <= _LAT_RANGE[1])))
            lon_in = float(np.mean((lons >= _LON_RANGE[0]) & (lons <= _LON_RANGE[1])))
            if lat_in >= 0.95 and lon_in >= 0.95:
                findings.append({
                    "type": "geolocation_pair",
                    "confidence": "high",
                    "severity": "medium",
                    "surface": f"{lat_keys[0]} + {lon_keys[0]}",
                    "evidence": (
                        f"lat range [{lats.min():.1f}, {lats.max():.1f}] "
                        f"lon range [{lons.min():.1f}, {lons.max():.1f}] "
                        f"(rounded to 0.1 deg)"
                    ),
                    "detail": {
                        "lat_keys": lat_keys,
                        "lon_keys": lon_keys,
                        "lat_in_range_frac": round(lat_in, 4),
                        "lon_in_range_frac": round(lon_in, 4),
                    },
                })
    elif geo_keys:
        # Presence alone warrants review; we cannot prove the values here.
        findings.append({
            "type": "geolocation_feature",
            "confidence": "medium",
            "severity": "medium",
            "surface": ", ".join(geo_keys),
            "evidence": "feature name indicates GNSS/geolocation; values not validated",
            "detail": {"keys": geo_keys},
        })
    return findings


# ---------------------------------------------------------------------------
# Face detection (optional — PyAV + opencv)
# ---------------------------------------------------------------------------

_NOT_SCANNED_HINTS: Dict[str, str] = {
    "opencv_missing": (
        "face detection skipped — install it with: "
        "pip install 'opencv-python-headless>=4.8,<5'"
    ),
    "haar_cascade_missing": (
        "face detection skipped — opencv is installed but ships no Haar "
        "cascade (OpenCV 5.x dropped them). Pin opencv-python-headless<5, "
        "or point RDA_HAAR_CASCADE at a haarcascade_frontalface_default.xml"
    ),
    "video_deps_missing": (
        "face detection skipped — install video support with: pip install av"
    ),
    "video_path_unknown": (
        "face detection skipped — loader did not provide "
        "dataset_root/fps/video_features"
    ),
    "video_not_scanned_pass_pii_video_to_enable": (
        "video frames were NOT scanned — pass --pii-video to include them"
    ),
}


def _not_scanned_blurb(codes: List[str]) -> List[str]:
    """Human-readable expansion of ``not_scanned`` codes.

    A privacy report must state what it did NOT check, otherwise "no
    findings" is silently read as "checked and clean".
    """
    out = []
    for code in codes:
        base = code.split(":", 1)[0]
        hint = _NOT_SCANNED_HINTS.get(base)
        if hint:
            out.append(f"{code} — {hint}")
        elif code.startswith("video_not_found") or code.startswith("video_not_decodable"):
            out.append(f"{code} — camera listed in the dataset but could not be read")
        else:
            out.append(code)
    return out


def _resolve_cascade() -> Optional[Path]:
    """Locate a usable frontal-face Haar cascade, or None.

    Resolution order:
      1. ``RDA_HAAR_CASCADE`` — explicit override, for environments that
         keep the XML outside opencv's data directory.
      2. ``cv2.data.haarcascades`` — what opencv normally ships.

    Verified 2026-10-13: ``opencv-python-headless`` 4.14.0 ships 19 cascade
    XMLs including ``haarcascade_frontalface_default.xml`` (930 KB), while
    **5.0.0 ships none** (``cv2/data`` contains only ``__init__.py``). The
    5.x case must therefore degrade loudly rather than report "no faces".
    """
    override = os.environ.get("RDA_HAAR_CASCADE")
    if override:
        p = Path(override)
        if p.exists():
            return p
    try:
        import cv2
    except ImportError:
        return None
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if not data_dir:
        return None
    p = Path(data_dir) / _CASCADE_FILE
    return p if p.exists() else None


def _opencv_missing() -> Optional[str]:
    """None when a usable opencv cascade is available, else reason code."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return "opencv_missing"
    return None if _resolve_cascade() is not None else "haar_cascade_missing"


def _decode_frames_bgr(video_path: Path, fps: float, n: int) -> List[np.ndarray]:
    """Decode up to ``n`` uniformly spaced frames as BGR uint8 arrays."""
    import av

    frames: List[np.ndarray] = []
    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                return frames
            stream = container.streams.video[0]
            if not stream.duration:
                return frames
            duration = float(stream.duration * stream.time_base)
            for k in range(n):
                frac = k / max(n, 1)
                target = min(duration * frac, max(duration - 0.05, 0.0))
                try:
                    container.seek(int(target / stream.time_base),
                                   stream=stream, backward=True)
                except (OSError, ValueError):
                    continue
                for frame in container.decode(stream):
                    frames.append(frame.to_ndarray(format="bgr24"))
                    break
    except (OSError, ValueError, IndexError, StopIteration):
        return frames
    return frames


def _face_findings(
    episode: EpisodeData, n_samples: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan sampled video frames for human faces (requires opencv)."""
    try:
        import av  # noqa: F401
    except ImportError:
        return [], ["video_deps_missing"]

    missing = _opencv_missing()
    if missing:
        return [], [missing]

    meta = episode.meta or {}
    video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
    dataset_root = meta.get("dataset_root")
    fps = meta.get("fps")
    if not video_features or not dataset_root or not fps:
        return [], ["video_path_unknown"]

    root = Path(dataset_root)
    import cv2

    cascade_path = _resolve_cascade()
    if cascade_path is None:  # pragma: no cover - guarded by _opencv_missing
        return [], ["haar_cascade_missing"]
    cascade = cv2.CascadeClassifier(str(cascade_path))
    findings: List[Dict[str, Any]] = []
    not_scanned: List[str] = []

    for feature, info in sorted(video_features.items()):
        chunk, file_idx = info.get("chunk_index"), info.get("file_index")
        if None in (chunk, file_idx):
            continue
        video_path = (
            root / "videos" / feature
            / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
        )
        if not video_path.exists():
            ep_index = info.get("episode_index")
            if ep_index is not None:
                video_path = (
                    root / "videos" / f"chunk-{int(chunk):03d}" / feature
                    / f"episode_{int(ep_index):06d}.mp4"
                )
        if not video_path.exists():
            not_scanned.append(f"video_not_found:{feature}")
            continue

        frames = _decode_frames_bgr(video_path, float(fps), n_samples)
        if not frames:
            not_scanned.append(f"video_not_decodable:{feature}")
            continue

        hits = 0
        largest = 0
        for arr in frames:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            if len(faces):
                hits += 1
                largest = max(largest, int(max(w * h for (_, _, w, h) in faces)))

        if hits >= _FACE_CONFIDENT_FRAMES:
            findings.append({
                "type": "face_in_video",
                "confidence": "medium" if hits < max(2, len(frames) // 2) else "high",
                "severity": "high",
                "surface": f"video.{feature}",
                "evidence": (
                    f"{hits}/{len(frames)} sampled frames contain a frontal face "
                    f"(largest detection {largest} px^2)"
                ),
                "detail": {
                    "frames_with_face": hits,
                    "frames_sampled": len(frames),
                    "largest_face_area_px": largest,
                    "detector": "haarcascade_frontalface_default",
                },
            })

    return findings, not_scanned


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

class PIIScanMetric(MetricBase):
    """Privacy / PII exposure scan over text surfaces, features and video.

    WARNING-level by default — see module docstring for the verdict policy.

    Args:
        scan_video: Also sample video frames for face detection. Off by
            default because it is the expensive part; enable via
            ``--pii-video``.
        video_sample_frames: Frames sampled per camera when ``scan_video``.
        text_allowlist: Optional iterable of surface names to skip.
    """

    name = "pii_scan"
    description = (
        "Privacy/PII exposure scan: direct identifiers, geolocation features "
        "and faces in video (WARNING-level by default; see --pii-strict)."
    )

    def __init__(
        self,
        scan_video: bool = False,
        video_sample_frames: int = _VIDEO_SAMPLE_FRAMES,
        text_allowlist: Optional[Iterable[str]] = None,
    ) -> None:
        self.scan_video = bool(scan_video)
        self.video_sample_frames = max(int(video_sample_frames), 1)
        self.text_allowlist = set(text_allowlist or ())

    def compute(self, episode: EpisodeData) -> MetricResult:
        findings: List[Dict[str, Any]] = []
        not_scanned: List[str] = []
        surfaces_seen = 0

        for surface, text in _collect_text_surfaces(episode):
            if surface in self.text_allowlist or not text:
                continue
            surfaces_seen += 1
            for dname, pattern, validate, conf, sev in _TEXT_DETECTORS:
                for m in pattern.finditer(text):
                    raw = m.group(0)
                    kind = validate(raw)
                    if kind is None:
                        # Structurally similar, semantically not a hit.
                        continue
                    findings.append({
                        "type": dname,
                        "confidence": conf,
                        "severity": sev,
                        "surface": surface,
                        "evidence": _redact(raw),
                        "detail": {
                            "kind": kind,
                            "char_offset": m.start(),
                            "length": len(raw),
                            "fingerprint": _fingerprint(raw),
                        },
                    })

        findings.extend(_geo_findings(episode))

        video_scanned = False
        if self.scan_video:
            face_findings, face_not_scanned = _face_findings(
                episode, self.video_sample_frames
            )
            findings.extend(face_findings)
            not_scanned.extend(face_not_scanned)
            video_scanned = not bool(face_not_scanned)
        else:
            not_scanned.append("video_not_scanned_pass_pii_video_to_enable")

        high = [f for f in findings if f["severity"] == "high"]
        details: Dict[str, Any] = {
            "findings": findings,
            "not_scanned": not_scanned,
            "not_scanned_detail": _not_scanned_blurb(not_scanned),
            "redaction_policy": (
                "identifiers are masked; 'fingerprint' is the first 12 hex "
                "chars of SHA-256 over the raw hit, for cross-episode dedupe"
            ),
            "detectors": [d[0] for d in _TEXT_DETECTORS] + ["geolocation_features"],
            "coverage_limits": (
                "text surfaces, numeric features and (opt-in) sampled video "
                "frames only. No OCR / licence-plate recognition, no voice "
                "biometrics, no consent or licensing review."
            ),
        }

        measurement: Dict[str, Any] = {
            "finding_count": len(findings),
            "high_severity_count": len(high),
            "surfaces_scanned": surfaces_seen,
            "types_found": sorted({f["type"] for f in findings}),
            "video_scanned": video_scanned,
            "score_compat": 1.0 if not findings else 0.0,
        }

        if not findings:
            return MetricResult.make_pass(
                name=self.name,
                measurement=measurement,
                message=(
                    "No PII indicators found in scanned surfaces."
                    if not not_scanned
                    else "No PII indicators found — but some surfaces were NOT scanned."
                ),
                details=details,
            )

        summary = ", ".join(
            f"{t} x{sum(1 for f in findings if f['type'] == t)}"
            for t in measurement["types_found"]
        )
        return MetricResult(
            name=self.name,
            availability=MetricAvailability.AVAILABLE,
            measurement=measurement,
            assessment={
                "status": "pass",
                "severity": "high" if high else "medium",
                "reason": None,
                "note": (
                    "WARNING-level finding: does NOT change the verdict. "
                    "Run with --pii-strict to gate on it."
                ),
            },
            details=details,
            message=f"Potential privacy exposure detected: {summary}",
            has_finding=True,
        )
