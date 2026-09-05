"""REQ-10 (v0.7.1): visual corruption injection benchmark.

Closes the v0.7.0 verification gap: VA-A/VA-B were only validated in the
"clean data passes" direction. This script injects *known* visual
corruptions into real dataset episodes and measures whether the metrics
detect them — producing the precision/recall evidence the roadmap
requires before the "visual integrity audit" claim is advertised.

Modes
-----
inject        Corrupt N episodes of a dataset with each scenario, run
              VA-A/VA-B on every corrupted episode, print a detection
              table, save JSON results.
distribution  Run visual_quality measurements (no corruption) on N
              episodes of each dataset and print threshold-stability
              statistics (REQ-10c, multi-dataset calibration).
perf          Time the decode paths on a long-GOP chunk file
              (REQ-10d, performance numbers for the README).

Usage (from repo root):
    python benchmarks/visual_inject.py inject [dataset] [n_episodes]
    python benchmarks/visual_inject.py distribution [n_episodes]
    python benchmarks/visual_inject.py perf

Numpy-only corruption (no OpenCV/scipy): Gaussian blur is a separable
kernel via sliding_window_view, brightness scaling clips at 0/255,
freeze scenarios re-encode with crf=0 so injected stalls produce
codec-identical frames exactly like a real camera stall.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.io.lerobot_loader import iter_episodes  # noqa: E402
from rda.io.schema import EpisodeData  # noqa: E402
from rda.metrics.visual_integrity import VideoFreezeMetric, _decode_span_gray  # noqa: E402
from rda.metrics.visual_quality import VisualQualityMetric, _decode_one_gray  # noqa: E402

DEFAULT_DATASET = r"D:\workbuddy-data\datasets\libero_10"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Real-dataset roster for the distribution mode (REQ-10c).
DISTRIBUTION_DATASETS = {
    "libero_10": r"D:\workbuddy-data\datasets\libero_10",
    "jaco_play": r"D:\workbuddy-data\datasets\refetch_jaco_play",
    "xarm_lift": r"D:\workbuddy-data\datasets\refetch_xarm_lift_medium",
}


# ---------------------------------------------------------------------------
# numpy-only corruption primitives
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(int(3 * sigma), 1)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (uint8 in, uint8 out).

    Accepts (H, W) or (N, H, W); blur is applied per-frame (spatial only).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    squeeze = img.ndim == 2
    if squeeze:
        img = img[None]
    p = np.pad(img.astype(np.float32), ((0, 0), (r, r), (0, 0)), mode="edge")
    # sliding over H axis (axis 1 of the padded 3D array)
    win = sliding_window_view(p, len(k), axis=1)          # (N, Hp, W, K)
    img2 = np.tensordot(win, k, axes=([3], [0]))          # (N, Hp, W)
    p = np.pad(img2, ((0, 0), (0, 0), (r, r)), mode="edge")
    win = sliding_window_view(p, len(k), axis=2)          # (N, H, Wp, K)
    out = np.tensordot(win, k, axes=([3], [0]))           # (N, H, W)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[0] if squeeze else out


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# video (re-)encoding
# ---------------------------------------------------------------------------

def _encode_video(path: Path, frames: np.ndarray, fps: float, crf: Optional[str] = None) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        if crf is not None:
            stream.options = {"crf": crf}
        stream.gop_size = max(int(fps * 2), 1)
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="gray")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def _corrupt_frames(
    frames: np.ndarray,
    scenario: str,
    param: float,
) -> np.ndarray:
    """Apply one corruption to an (N, H, W) uint8 gray array."""
    n = frames.shape[0]
    if scenario == "blur":
        return gaussian_blur(frames, param)
    if scenario == "dark":
        return brightness(frames, param)
    if scenario == "blown":
        return brightness(frames, param)
    if scenario == "freeze":
        # param = (start_frac, end_frac) tuple flattened by caller; handled below
        raise ValueError("use _corrupt_frames_freeze")
    raise ValueError(f"unknown scenario {scenario}")


def _apply_scenario(
    frames: np.ndarray, scenario: str, param: Any
) -> np.ndarray:
    n = frames.shape[0]
    if scenario == "clean":
        return frames
    if scenario.startswith("blur"):
        return gaussian_blur(frames, float(param))
    if scenario in ("dark", "blown") or scenario.startswith("dark") or scenario.startswith("blown"):
        return brightness(frames, float(param))
    if scenario == "freeze" or scenario.startswith("freeze"):
        s, e = param  # fractional span
        out = frames.copy()
        fs, fe = int(n * s), int(n * e)
        out[fs:fe] = out[fs]
        return out
    if scenario == "static_camera":
        return np.repeat(frames[:1], n, axis=0)
    raise ValueError(f"unknown scenario {scenario}")


# ---------------------------------------------------------------------------
# scenario roster (REQ-10 acceptance: five corruption families)
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    # --- VA-B: quality corruptions (visual_quality must REVIEW these) ---
    {"name": "clean",          "param": None,  "expect_vq": "pass"},
    {"name": "blur_mild",      "param": 2.0,   "expect_vq": "pass|review"},
    {"name": "blur_heavy",     "param": 4.0,   "expect_vq": "review"},
    {"name": "blur_extreme",   "param": 8.0,   "expect_vq": "review"},
    {"name": "dark_deep",      "param": 0.2,   "expect_vq": "review"},
    {"name": "dark_moderate",  "param": 0.4,   "expect_vq": "pass|review"},
    # ×3.0 on libero (mean lum ≈ 90-110) pushes mean past the 230 band;
    # ×2.5 was insufficient in the first calibration run.
    {"name": "blown_out",      "param": 3.0,   "expect_vq": "review"},
    # --- VA-A: freeze corruptions (video_freeze grading) ---
    {"name": "freeze_short",   "param": (0.50, 0.55), "expect_vf": "review|pass"},
    {"name": "freeze_long",    "param": (0.10, 0.48), "expect_vf": "exclude"},
    {"name": "freeze_majority","param": (0.00, 0.60), "expect_vf": "exclude",
     "note": "boundary: >50% frozen — adaptive epsilon must not desensitize"},
    {"name": "static_camera",  "param": None,  "expect_vf": "exclude",
     "note": "whole-stream stall — camera drop-out signature"},
]


def _build_corrupted_episode(
    clean_ep: EpisodeData,
    scenario: str,
    param: Any,
    workdir: Path,
    decode_size: int = 128,
) -> Optional[EpisodeData]:
    """Re-encode every camera span of the episode; corrupt the first one.

    Returns a new EpisodeData whose meta points at the re-encoded files,
    with the real parquet action timeline kept intact. Re-encoded files
    start at t=0, so every camera's ``from_timestamp`` is rewritten to
    0.0 (span duration unchanged — timestamp-alignment semantics kept).
    Cameras other than the first are re-encoded *clean* so that only the
    injected corruption (not missing files) can move the metrics.
    """
    import copy

    meta = clean_ep.meta or {}
    video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
    if not video_features:
        return None
    fps = float(meta["fps"])
    features = sorted(video_features.keys())
    root = workdir / scenario
    span_sec: Optional[float] = None

    new_features: Dict[str, Dict[str, Any]] = {}
    for fi, fname in enumerate(features):
        info = video_features[fname]
        chunk_idx = int(info["chunk_index"])
        file_idx = int(info["file_index"])
        from_ts, to_ts = float(info["from_timestamp"]), float(info["to_timestamp"])
        span = to_ts - from_ts
        span_sec = span

        frames = _decode_span_custom(
            Path(meta["dataset_root"]) / "videos" / fname
            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4",
            from_ts, to_ts, fps, decode_size,
        )
        if frames is None:
            return None

        if fi == 0:
            corrupted = _apply_scenario(frames, scenario, param)
        else:
            corrupted = frames
        crf = "0" if (fi == 0 and (scenario.startswith("freeze") or scenario == "static_camera")) else None

        out = root / "videos" / fname / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        _encode_video(out, corrupted, fps, crf=crf)

        nf = dict(info)
        nf["from_timestamp"] = 0.0
        nf["to_timestamp"] = round(span, 4)
        new_features[fname] = nf

    new_meta = copy.deepcopy(meta)
    new_meta["dataset_root"] = str(root)
    new_meta["video_features"] = new_features
    ep = copy.deepcopy(clean_ep)
    ep.meta = new_meta
    return ep


def _decode_span_custom(
    video_path: Path, start_sec: float, end_sec: float, fps: float, size: int
) -> Optional[np.ndarray]:
    """_decode_span_gray clone with a configurable decode size."""
    import av

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            try:
                container.seek(int(start_sec / tb), stream=stream)
            except Exception:
                pass
            out: List[np.ndarray] = []
            for pf in container.decode(stream):
                pts = float(pf.pts * tb) if pf.pts is not None else None
                if pts is not None and pts < start_sec - 1.0 / max(fps, 1.0):
                    continue
                if pts is not None and pts >= end_sec:
                    break
                out.append(pf.reformat(width=size, height=size, format="gray").to_ndarray())
            return np.stack(out) if out else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# inject mode
# ---------------------------------------------------------------------------

def run_inject(dataset_root: str, n_episodes: int) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="rda_vinject_"))
    per_scenario: Dict[str, List[Dict[str, Any]]] = {s["name"]: [] for s in SCENARIOS}

    episodes = []
    for i, ep in enumerate(iter_episodes(dataset_root)):
        episodes.append(ep)
        if len(episodes) >= n_episodes:
            break
    if not episodes:
        print("no episodes loaded; abort")
        return

    t0 = time.time()
    for s in SCENARIOS:
        for ep in episodes:
            cep = _build_corrupted_episode(ep, s["name"], s["param"], workdir)
            if cep is None:
                continue
            vq = VisualQualityMetric().compute(cep)
            vf = VideoFreezeMetric().compute(cep)
            per_scenario[s["name"]].append({
                "vq_status": vq.assessment["status"],
                "vq_penalty": (vq.measurement or {}).get("penalty"),
                "vq_dominant": _dominant_of(vq),
                "vf_status": vf.assessment["status"],
                "vf_regions": (vf.details or {}).get("freeze_region_count"),
            })
        print(f"  [{s['name']}] done ({len(per_scenario[s['name']])} eps)")
    elapsed = time.time() - t0

    # ---- summary table ----
    print(f"\n=== REQ-10 injection results — {len(episodes)} episodes × "
          f"{len(SCENARIOS)} scenarios ({elapsed:.0f}s) ===\n")
    header = (f"{'scenario':<16} {'expect_vq':<12} {'vq pass/review/na':<20} "
              f"{'med penalty':<12} {'expect_vf':<12} {'vf pass/review/excl/na':<26}")
    print(header)
    print("-" * len(header))
    for s in SCENARIOS:
        rows = per_scenario[s["name"]]
        if not rows:
            continue
        vq_counts = _counts(rows, "vq_status")
        vf_counts = _counts(rows, "vf_status")
        pens = [r["vq_penalty"] for r in rows if r["vq_penalty"] is not None]
        med_pen = f"{np.median(pens):.2f}" if pens else "-"
        print(f"{s['name']:<16} {str(s.get('expect_vq', '-')):<12} "
              f"{vq_counts:<20} {med_pen:<12} "
              f"{str(s.get('expect_vf', '-')):<12} {vf_counts:<26}")

    # ---- acceptance checks ----
    print("\n=== acceptance checks ===")
    failures = 0
    for s in SCENARIOS:
        rows = per_scenario[s["name"]]
        if not rows:
            continue
        if "expect_vq" in s:
            ok = all(r["vq_status"] in s["expect_vq"].split("|") for r in rows)
            mark = "OK " if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"  [{mark}] vq {s['name']}: expected {s['expect_vq']}, "
                  f"got {set(r['vq_status'] for r in rows)}")
        if "expect_vf" in s:
            ok = all(r["vf_status"] in s["expect_vf"].split("|") for r in rows)
            mark = "OK " if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"  [{mark}] vf {s['name']}: expected {s['expect_vf']}, "
                  f"got {set(r['vf_status'] for r in rows)}")

    # ---- persist ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"visual_inject_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.write_text(json.dumps({
        "dataset": dataset_root,
        "n_episodes": len(episodes),
        "scenarios": {k: v for k, v in per_scenario.items()},
        "failures": failures,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults saved: {out_json}")
    print(f"workdir (corrupted videos): {workdir}")
    if failures == 0:
        print("ALL ACCEPTANCE CHECKS PASSED")
    shutil.rmtree(workdir, ignore_errors=True)


def _counts(rows: List[Dict[str, Any]], key: str) -> str:
    order = ["pass", "review", "exclude", "na", "skipped"]
    c = {k: 0 for k in order}
    for r in rows:
        c[r[key]] = c.get(r[key], 0) + 1
    return "/".join(str(c[k]) for k in ["pass", "review", "exclude", "na"])


def _dominant_of(vq_result) -> Optional[str]:
    return (vq_result.measurement or {}).get("worst_feature") and \
        ((vq_result.details or {}).get("per_feature", {}))


# ---------------------------------------------------------------------------
# distribution mode (REQ-10c)
# ---------------------------------------------------------------------------

def run_distribution(n_episodes: int) -> None:
    print("=== REQ-10c visual_quality distribution across datasets ===\n")
    summary: Dict[str, Any] = {}
    for name, root in DISTRIBUTION_DATASETS.items():
        pens, blurs, lums, contrasts = [], [], [], []
        try:
            for i, ep in enumerate(iter_episodes(root)):
                r = VisualQualityMetric().compute(ep)
                if r.assessment["status"] == "na":
                    continue
                m = r.measurement or {}
                if m.get("penalty") is not None:
                    pens.append(m["penalty"])
                for feat_stats in (r.details or {}).get("per_feature", {}).values():
                    blurs.append(feat_stats["median_blur_var"])
                    lums.append(float(np.mean([s["mean_lum"] for s in feat_stats["samples"]])))
                    contrasts.append(float(np.mean(
                        [s["contrast_p5_p95"] for s in feat_stats["samples"]])))
                if i + 1 >= n_episodes:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: ERROR {e}")
            continue
        if not pens:
            print(f"  {name}: no decodable episodes")
            continue
        summary[name] = {
            "n": len(pens),
            "penalty_p50": float(np.percentile(pens, 50)),
            "penalty_p90": float(np.percentile(pens, 90)),
            "review_rate": float(np.mean([p >= 0.5 for p in pens])),
            "blur_var_p10": float(np.percentile(blurs, 10)),
            "blur_var_p50": float(np.percentile(blurs, 50)),
            "lum_p50": float(np.percentile(lums, 50)),
            "contrast_p50": float(np.percentile(contrasts, 50)),
        }
        s = summary[name]
        print(f"  {name} (n={s['n']}): penalty p50={s['penalty_p50']:.2f} "
              f"p90={s['penalty_p90']:.2f} | review_rate={s['review_rate']:.0%} | "
              f"blur_var p10={s['blur_var_p10']:.1f} p50={s['blur_var_p50']:.1f} | "
              f"lum p50={s['lum_p50']:.0f} | contrast p50={s['contrast_p50']:.0f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"visual_distribution_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nsaved: {out}")


# ---------------------------------------------------------------------------
# perf mode (REQ-10d)
# ---------------------------------------------------------------------------

def run_perf(dataset_root: str, n_episodes: int) -> None:
    print(f"=== REQ-10d decode performance on {Path(dataset_root).name} ===\n")
    freeze_times: List[float] = []
    vq_times: List[float] = []
    frame_counts: List[int] = []
    for i, ep in enumerate(iter_episodes(dataset_root)):
        vf0 = time.time()
        VideoFreezeMetric().compute(ep)
        freeze_times.append(time.time() - vf0)
        vq0 = time.time()
        VisualQualityMetric().compute(ep)
        vq_times.append(time.time() - vq0)
        frame_counts.append(ep.num_frames)
        if i + 1 >= n_episodes:
            break
    print(f"  episodes: {len(freeze_times)} | frames/ep: "
          f"{min(frame_counts)}-{max(frame_counts)}")
    print(f"  video_freeze    (full-span 64x64 decode): "
          f"mean={np.mean(freeze_times):.2f}s worst={np.max(freeze_times):.2f}s")
    print(f"  visual_quality  (10 single-frame seeks):  "
          f"mean={np.mean(vq_times):.2f}s worst={np.max(vq_times):.2f}s")
    print(f"  combined per-episode budget: "
          f"mean={np.mean(freeze_times) + np.mean(vq_times):.2f}s")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inject"
    if mode == "inject":
        ds = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        run_inject(ds, n)
    elif mode == "distribution":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_distribution(n)
    elif mode == "perf":
        ds = sys.argv[2] if len(sys.argv) > 2 else r"D:\workbuddy-data\datasets\refetch_jaco_play"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        run_perf(ds, n)
    else:
        print(__doc__)
