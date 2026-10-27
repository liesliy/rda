"""rda/conform/video.py 的单元测试。

重点钉三条容易被"看起来能跑"掩盖的属性：

1. **数值精度**：合成一段每帧位移已知的视频，解码后光流必须恢复出该位移
   （误差 < 0.5 px）。测不出真值的输出没有资格进符合性报告。
2. **依赖缺失 → 降级而不是崩溃**：av / cv2 不在时给 ImportError + 可用标志
   置 False，措辞指向本地环境、不指向数据集。
3. **不可解码 / 无视频轨 / 帧率不可知**：返回结构化错误信息，不抛异常、不
   伪造数字。尤其是 fps：**读不到就是 None，绝不猜 30/60**。
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from rda.conform.video import (
    DIAG_REJECT_RATIO,
    FramePairSet,
    VideoProbe,
    displacement_is_blur_upper_bound,
    lk_displacement,
    stream_fps,
    summarize_displacements,
)

pytest.importorskip("av")
av = pytest.importorskip("av")
pytest.importorskip("cv2")


def _detect_encoders() -> list:
    """Return the usable video encoders, preferring libx264 then mpeg4."""
    out = []
    for name in ("libx264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
        except Exception:  # noqa: BLE001 — 编码器不可用不是产品缺陷
            continue
        out.append(name)
    return out


AVAILABLE_ENCODERS = _detect_encoders()
if not AVAILABLE_ENCODERS:  # pragma: no cover — 取决于本地 FFmpeg 构建
    pytest.skip(
        "本地环境没有 libx264 / mpeg4 编码器，无法合成参照视频；"
        "光流数值精度的验证结果为**未确认**，不得据此声称达标。",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# 合成视频：**真 mp4 + 已知位移**，不是 mock
# ---------------------------------------------------------------------------


def _checkerboard(size: int = 40, cell: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)


def make_translation_video(
    path: Path,
    *,
    n_frames: int = 20,
    width: int = 320,
    height: int = 240,
    shift_px: float = 3.0,
    fps: float = 30.0,
    encoder: str = "libx264",
) -> dict:
    """Encode a video whose only content translates by ``shift_px`` per frame.

    返回真值元数据（``shift_px`` / 尺寸 / 帧数 / fps / 编码器）供断言对照。
    **只用真编码器写真 mp4**：用内存里的裸 ndarray 当"参考帧"会绕开 seek 与
    解码这条路径，也就测不到真实误差——那样跑出来的"通过"是自欺欺人。
    """
    patch = _checkerboard()
    ph, pw = patch.shape
    frames = []
    x0, y0 = 60, 100
    for i in range(n_frames):
        img = np.zeros((height, width), dtype=np.uint8)
        x = x0 + int(round(i * shift_px))
        if x + pw <= width and y0 + ph <= height:
            img[y0:y0 + ph, x:x + pw] = patch
        frames.append(img)

    with av.open(str(path), "w") as container:
        # PyAV 的 rate 要的是 Fraction，传 float 会报
        # "'float' object has no attribute 'numerator'"
        stream = container.add_stream(encoder, rate=Fraction(int(fps), 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        if encoder == "libx264":
            # 无损：压缩噪声会磨圆格子边缘，进而污染 0.5 px 级的精度断言。
            stream.options = {"crf": "0", "preset": "ultrafast"}
        for img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="gray")
            for packet in stream.encode(frame.reformat(format="yuv420p")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {"shift_px": shift_px, "n_frames": n_frames, "width": width,
            "height": height, "fps": fps, "encoder": encoder, "path": path}


@pytest.fixture(scope="module")
def encoder_name() -> str:
    return AVAILABLE_ENCODERS[0]


# ---------------------------------------------------------------------------
# 1. 数值精度：光流必须复原已知位移
# ---------------------------------------------------------------------------


def test_lk_recovers_known_translation(tmp_path: Path, encoder_name: str):
    """每帧平移 3 px → 光流位移中位数与真值的误差必须 < 0.5 px。

    这条用例是"视频层给出的数字可以直接写进第三方测评报告"的前提：运动模糊
    判定用的就是这个数，测不准就说明判定不成立。
    """
    truth = make_translation_video(tmp_path / "t3.mp4", n_frames=16,
                                   shift_px=3.0, encoder=encoder_name)
    probe = VideoProbe()
    span = truth["n_frames"] / truth["fps"]
    ps = probe.read_consecutive_pairs(truth["path"], 0.0, span,
                                      n_pairs=truth["n_frames"] - 1)
    assert ps.decode_error is None, ps.decode_error
    assert ps.n_pairs >= 10, f"只解出 {ps.n_pairs} 对帧（需要 ≥10 对）"

    medians = []
    for a, b in ps.pairs:
        res = lk_displacement(a, b)
        assert res.n_features > 0, "没有找到可跟踪角点"
        assert res.n_valid > 0, f"跟踪点全被剔除：{res}"
        medians.append(float(np.median(res.displacements)))
    observed = float(np.median(medians))
    err = abs(observed - truth["shift_px"])
    assert err < 0.5, (
        f"光流恢复的位移中位数 {observed:.4f} px 与真值 "
        f"{truth['shift_px']} px 相差 {err:.4f} px，超过 0.5 px 容差"
        f"（编码器 {truth['encoder']}，逐帧中位数 {medians}）")


def test_lk_rejects_implausible_tracks(tmp_path: Path, encoder_name: str):
    """位移超过图像对角线 10% 的跟踪点必须被剔除，且剔除数可核对。"""
    truth = make_translation_video(tmp_path / "jump.mp4", n_frames=12,
                                   shift_px=3.0, encoder=encoder_name)
    probe = VideoProbe()
    ps = probe.read_consecutive_pairs(truth["path"], 0.0,
                                      truth["n_frames"] / truth["fps"],
                                      n_pairs=4)
    a, b = ps.pairs[0]
    teleported = np.roll(b, b.shape[1] // 2, axis=1)  # 人为把画面"传送"半幅
    res = lk_displacement(a, b)
    res_bad = lk_displacement(a, teleported)
    # 正常相邻帧：几乎不剔除
    assert res.n_rejected <= res.n_features
    # 被破坏的帧：**剔除数之与必须等于初始点数**（有效点 + 剔除点 = 全部）
    assert res_bad.n_valid + res_bad.n_rejected == res_bad.n_features
    assert res_bad.n_rejected > 0, "位移超过对角线 10% 的跟踪点应当判为失败"
    assert res_bad.n_rejected == (
        res_bad.n_rejected_status + res_bad.n_rejected_displacement)
    assert DIAG_REJECT_RATIO == 0.10


def test_decoding_keeps_original_resolution(tmp_path: Path, encoder_name: str):
    """解码必须保持原分辨率——降采样会先抹掉 5 px 尺度的信息。"""
    truth = make_translation_video(tmp_path / "res.mp4", width=320, height=240,
                                   encoder=encoder_name)
    probe = VideoProbe()
    ps = probe.read_consecutive_pairs(truth["path"], 0.0, 0.6, n_pairs=3)
    assert ps.pairs[0][0].shape == (240, 320), ps.pairs[0][0].shape
    assert probe.probe_fps(truth["path"]) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 2. 依赖缺失 → 降级（措辞必须指向本地环境）
# ---------------------------------------------------------------------------


def _block_module(monkeypatch, name: str):
    import builtins

    real_import = builtins.__import__

    def fake_import(mod, *args, **kwargs):
        if mod == name or mod.startswith(name + "."):
            raise ImportError(f"No module named '{name}'")
        return real_import(mod, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_av_degrades_to_flag_and_import_error(monkeypatch):
    """av 缺失：`av_available` 置 False + ImportError，不许悄悄返回空结果。"""
    _block_module(monkeypatch, "av")
    probe = VideoProbe()
    with pytest.raises(ImportError) as ei:
        probe._require_av()
    assert probe.av_available is False
    assert "install" in str(ei.value).lower()
    with pytest.raises(ImportError):
        probe.read_consecutive_pairs(Path("x.mp4"), 0.0, 1.0, n_pairs=2)
    # 帧率读不到是因为没装 av，不是因为文件里写了 fps=0：必须原样抛，
    # 不许被"解码失败"这类措辞掩盖。
    with pytest.raises(ImportError):
        probe.probe_fps(Path("x.mp4"))


def test_missing_cv2_degrades_to_flag_and_import_error(monkeypatch,
                                                       tmp_path: Path,
                                                       encoder_name: str):
    """cv2 缺失：光流无法计算，必须 ImportError（上层据此判 UNDETERMINED）。"""
    _block_module(monkeypatch, "cv2")
    probe = VideoProbe()
    with pytest.raises(ImportError):
        probe._require_cv2()
    assert probe.cv2_available is False
    truth = make_translation_video(tmp_path / "nocv.mp4", encoder=encoder_name)
    ps = probe.read_consecutive_pairs(truth["path"], 0.0, 0.5, n_pairs=2)
    assert ps.pairs  # 解码只依赖 av，cv2 不在也能解
    # 但**光流前处理**（高斯平滑）要用 cv2：不可用必须显式抛、不许静默跳过
    with pytest.raises(ImportError):
        probe.read_consecutive_pairs(truth["path"], 0.0, 0.5, n_pairs=2,
                                     blur_ksize=3)
    with pytest.raises(ImportError):
        probe.track_displacements(ps.pairs[0])


# ---------------------------------------------------------------------------
# 3. 坏输入：结构化错误，不崩溃、不猜 fps
# ---------------------------------------------------------------------------


def test_garbage_file_returns_error_not_exception(tmp_path: Path):
    """16 字节假 mp4（旧 fixture 的写法）必须给出原因，不能抛 Unicode 哭 error。"""
    p = tmp_path / "garbage.mp4"
    p.write_bytes(b"\x00" * 16)
    probe = VideoProbe()
    ps = probe.read_consecutive_pairs(p, 0.0, 1.0, n_pairs=2)
    assert ps.pairs == []
    assert ps.decode_error, "必须写明解码失败原因，不许静默返回空结果"
    assert probe.probe_fps(p) is None  # 读不到 fps 就是 None，绝不返回 30/60


def test_container_without_video_stream_is_distinguished(tmp_path: Path):
    """只有音频轨的容器：'无视频轨' 必须与 '不可解码' 区分开写。"""
    p = tmp_path / "audio_only.mkv"
    try:
        with av.open(str(p), "w") as container:
            stream = container.add_stream("aac", rate=44100)
            frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
            frame.sample_rate = 44100
            frame.time_base = Fraction(1, 44100)
            for _ in range(4):
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as e:  # noqa: BLE001 — 没有音频编码器时不编这个用例
        pytest.skip(f"本地环境无 aac 编码器：{e}")
    probe = VideoProbe()
    ps = probe.read_consecutive_pairs(p, 0.0, 1.0, n_pairs=2)
    assert ps.pairs == []
    assert "视频轨" in (ps.decode_error or "")


def test_no_temp_files_are_left_behind(tmp_path: Path, encoder_name: str):
    """视频层计算不得落盘：运行结束后目录里除源 mp4 外不应有新文件。"""
    truth = make_translation_video(tmp_path / "src.mp4", encoder=encoder_name)
    probe = VideoProbe()
    probe.read_consecutive_pairs(truth["path"], 0.0,
                                 truth["n_frames"] / truth["fps"], n_pairs=4)
    leftovers = sorted(x.name for x in tmp_path.iterdir())
    assert leftovers == ["src.mp4"], f"产生了额外文件：{leftovers}"


def test_stream_fps_never_guesses():
    """没有帧率元数据时不许回落成一个"常见值"。"""

    class _NoRate:
        average_rate = None
        base_rate = None
        guessed_rate = None

    class _ZeroRate:
        average_rate = 0
        base_rate = None
        guessed_rate = None

    class _GoodRate:
        average_rate = 30
        base_rate = None
        guessed_rate = None

    assert stream_fps(_NoRate()) is None
    assert stream_fps(_ZeroRate()) is None
    assert stream_fps(_GoodRate()) == pytest.approx(30.0)


def test_robust_upper_bound_is_reported_alongside_max(tmp_path: Path,
                                                      encoder_name: str):
    """max 是单点极值；p99.9 / n_above_p99 必须一并给出，让野值可辨识。

    背景（真实缺陷）：lerobot_folding 的 full 模式曾报出 max = 75.0426 px，
    而同帧其余 199 个点最大只有 10.02 px —— 只有一个点，却进了正式报告。
    判定不受影响（用 p95），但第三方复核会读错，因此必须披露而非掩盖。
    """
    s = summarize_displacements(list(np.zeros(999)) + [75.0])
    assert s["max"] == pytest.approx(75.0)         # 单点极值：读起来像"有 75 px 的运动"
    assert s["p99_9"] == pytest.approx(0.075)      # 稳健上界：线性插值，几乎不受那一点影响
    assert s["p95"] == pytest.approx(0.0)
    assert s["n"] == 1000
    # 超过 p99 的点数只有 1 个 —— 一眼看出尾部是"一个野值"而不是"一群点"
    assert s["n_above_p99"] == 1


def test_lk_track_error_is_captured_not_discarded(tmp_path: Path,
                                                  encoder_name: str):
    """`calcOpticalFlowPyrLK` 的第三个返回值（逐点残差）必须被接出来。

    它此前被 `_err` 接收后丢弃。这次的要求是**披露**：不给 err 设任何魔法
    阈值（我们没有依据），只把它连同统计量报出去。
    """
    truth = make_translation_video(tmp_path / "err.mp4", n_frames=16,
                                   shift_px=3.0, encoder=encoder_name)
    probe = VideoProbe()
    ps = probe.read_consecutive_pairs(truth["path"], 0.0,
                                      truth["n_frames"] / truth["fps"],
                                      n_pairs=8)
    res = lk_displacement(*ps.pairs[0])
    assert res.n_valid > 0
    # 残差必须与位移**逐点对齐**，否则 err@max 那条证据就是错位的
    assert res.track_errors.shape == res.displacements.shape
    assert np.isfinite(res.track_errors).all()
    # 合成平移是干净的：残差应当接近 0
    assert float(np.median(res.track_errors)) < 1.0


def test_track_err_at_max_points_at_the_outlier():
    """err@max = 产生 max 的那个点自己的残差——复核单点野值时最直接的证据。"""
    disp = [1.0, 1.0, 2.0, 75.0]
    errs = [0.01, 0.02, 0.03, 66.74]
    s = summarize_displacements(disp, errs)
    assert s["max"] == pytest.approx(75.0)
    assert s["p99_9"] == pytest.approx(74.781)  # 4 个点，线性插值
    assert s["track_err_at_max"] == pytest.approx(66.74)
    assert s["track_err_p50"] == pytest.approx(0.025)
    assert s["track_err_n"] == 4


def test_missing_track_error_is_nan_not_zero():
    """残差不可用（长度对不上 / OpenCV 没返回）时必须表现为缺失，不许用 0 冒充。"""
    s = summarize_displacements([1.0, 2.0, 3.0], [0.1])  # 长度对不上
    assert s["n"] == 3                     # 位移本身照样统计
    assert s["track_err_p50"] is None      # 残差按缺失处理
    assert s["track_err_at_max"] is None
    assert s["track_err_n"] == 0


def test_summarize_empty_sample_is_none_not_zero():
    """空样本返回 None（→ UNDETERMINED），绝不返回 0（→ 会被误读成 PASS）。"""
    s = summarize_displacements(np.zeros((0,), dtype=float))
    assert s["n"] == 0
    assert s["p50"] is None and s["p95"] is None and s["max"] is None
    s2 = summarize_displacements([1.0, 2.0, 3.0, 4.0])
    assert s2["p50"] == pytest.approx(2.5)
    assert s2["max"] == pytest.approx(4.0)
    assert summarize_displacements(None)["n"] == 0


def test_upper_bound_statement_cites_its_source():
    """进报告的 method 文案必须带上界语义与出处，不写沦为无法复算的形容词。"""
    text = displacement_is_blur_upper_bound()
    assert "上界" in text
    assert "arXiv:2303.10247" in text
    assert FramePairSet().n_pairs == 0
