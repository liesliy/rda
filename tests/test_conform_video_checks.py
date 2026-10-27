"""视频层判定项（6.2.5.2 a) 1) 运动模糊）与附录 A 观测项的判定语义测试。

这里是**口径锁**。被钉住的几条边界一旦被改动，就会产生"没法归因的结论"：

* 视频不可得 / 依赖缺失 / ``--video=off`` → UNDETERMINED，**不是 FAIL**；
* 位移 > 5 px 但曝光分数 α 未声明 → UNDETERMINED，**不是 FAIL**（单边判定）；
* α 已声明 → ``L = α·d`` 才可双边判 PASS / FAIL；
* 无视频特征 → NOT_APPLICABLE；
* 附录 A.2 在没有音频特征时必须 NOT_APPLICABLE，且不许拿视频帧的 2D FFT
  冒充频谱流量（方法不符）；
* 每路相机的载荷必须按**特征级**判断：videos/ 目录在不在不算数。

视频用的是**真 mp4 + 已知位移**（合成线性平移），不是把光流 monkeypatch 掉的
假绿——那样测不到判定逻辑真正依赖的那个数。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rda.cli.main import cli
from rda.conform import (
    ConformOptions,
    Verdict,
    load_profile,
    render_conformance_markdown,
    run_conformance,
)
from tests.test_conform import write_dataset
from tests.test_conform_video import AVAILABLE_ENCODERS, make_translation_video

pytest.importorskip("av")
pytest.importorskip("cv2")

SHIFT_SMALL = 3.0   # < 5 px → 未声明 α 时也可判 PASS（上界蕴含合规）
SHIFT_LARGE = 12.0  # > 5 px → 未声明 α 时必须 UNDETERMINED，绝不 FAIL


def dataset_with_real_video(
    tmp_path: Path,
    shift_px: float,
    *,
    cameras=("cam0",),
    n_episodes: int = 2,
    frames_per_ep: int = 12,
    fps: int = 30,
    with_video_files: bool = True,
    broken_file: bool = False,
) -> Path:
    """Write a dataset whose ``videos/`` holds a real translation mp4.

    ``broken_file=True`` 时写 16 字节假 mp4——旧 fixture 的写法，用来证明
    "不可解码 ≠ 通过"。``with_video_files=False`` 时干脆不给载荷文件，模拟
    "视频没下载"。
    """
    ds = write_dataset(tmp_path / "ds", n_episodes=n_episodes,
                       frames_per_ep=frames_per_ep, fps=fps, cameras=cameras,
                       videos_with_files=True)
    for cam in cameras:
        d = ds / "videos" / f"observation.images.{cam}" / "chunk-000"
        target = d / "file-000.mp4"
        if not with_video_files:
            target.unlink(missing_ok=True)
            continue
        if broken_file:
            target.write_bytes(b"\x00" * 16)
            continue
        make_translation_video(target, n_frames=16, shift_px=shift_px, fps=float(fps))
    return ds


def item(result, metric: str):
    for it in result.items:
        if it.metric == metric:
            return it
    raise AssertionError(f"metric {metric!r} not found in "
                         f"{[i.metric for i in result.items]}")


# ---------------------------------------------------------------------------
# 1. 三条支路：STA 未声明 α
# ---------------------------------------------------------------------------


def test_pass_when_displacement_under_threshold_without_alpha(tmp_path: Path):
    """d ≤ 5 px 且 α 未声明 → PASS，但必须是"上界判定"的性质，value 留空。"""
    ds = dataset_with_real_video(tmp_path / "a", SHIFT_SMALL)
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.PASS
    assert it.binding == "宜"
    # α 未声明时 value 必须是 None —— 我们没测出核长度，只测出它的上界
    assert it.value is None
    assert "upper bound" in it.method or "上界" in it.method
    assert "arXiv:2303.10247" in it.method
    assert "Lucas-Kanade" in it.method
    assert "蕴含合规" in it.note
    # 红线：不许拿 Laplacian 方差之类的清晰度代理指标凑这条判定
    assert "Laplacian" in it.method  # 以"明确排除"的形式出现
    cams = it.details["cameras"]
    assert cams and cams[0]["displacement_px"]["p95"] is not None
    assert cams[0]["displacement_px"]["p95"] < 5.0


def test_undetermined_not_fail_when_displacement_over_threshold(tmp_path: Path):
    """d > 5 px 且 α 未声明 → UNDETERMINED，**绝不判 FAIL**。"""
    ds = dataset_with_real_video(tmp_path / "b", SHIFT_LARGE)
    result = run_conformance(ds)
    it = item(result, "motion_blur_kernel_px")
    assert it.verdict == Verdict.UNDETERMINED
    assert it.value is None
    assert "不得判 FAIL" in it.note or "绝不判 FAIL" in it.note
    assert it.details["upper_bound_p95_px"] > 5.0
    assert "motion_blur_kernel_px" not in [
        i.metric for i in result.items if i.verdict == Verdict.FAIL]


# ---------------------------------------------------------------------------
# 2. α 已声明： bilateral PASS / FAIL
# ---------------------------------------------------------------------------


def _profile_with_alpha(tmp_path: Path, alpha: float, **extra) -> Path:
    payload = {"exposure_fraction": alpha}
    payload.update(extra)
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_declared_exposure_fraction_gives_pass(tmp_path: Path):
    """α = 0.2，d ≈ 12 px → L ≈ 2.4 px ≤ 5 → PASS（value 为换算后的 L）。"""
    ds = dataset_with_real_video(tmp_path / "c", SHIFT_LARGE)
    prof = load_profile(_profile_with_alpha(tmp_path, 0.2))
    it = item(run_conformance(ds, profile=prof), "motion_blur_kernel_px")
    assert it.verdict == Verdict.PASS
    assert it.value is not None
    assert it.value < 5.0
    assert it.details["exposure_fraction"] == 0.2
    assert "委托方声明值" in it.note or "声明" in it.note


def test_declared_exposure_fraction_gives_advisory_fail(tmp_path: Path):
    """α = 0.9，d ≈ 12 px → L ≈ 10.8 px > 5 → FAIL（宜类，汇总单列）。"""
    ds = dataset_with_real_video(tmp_path / "d", SHIFT_LARGE)
    prof = load_profile(_profile_with_alpha(tmp_path, 0.9))
    result = run_conformance(ds, profile=prof)
    it = item(result, "motion_blur_kernel_px")
    assert it.verdict == Verdict.FAIL
    assert it.value > 5.0
    assert it.binding == "宜"
    assert result.summary["advisory_fail"] >= 1


def test_profile_can_override_the_threshold_value(tmp_path: Path):
    """阈值必须能被 profile 覆盖（0.10.x 起 motion_blur 没注册进 all_thresholds）。"""
    ds = dataset_with_real_video(tmp_path / "e", SHIFT_SMALL)
    p = tmp_path / "th.json"
    p.write_text(json.dumps({
        "thresholds": {"motion_blur_kernel_px": {"value": 1.0,
                                                 "comparator": "<="}},
        "sampling": {"video_episodes": 2, "video_frame_pairs": 6},
    }), encoding="utf-8")
    prof = load_profile(p)
    assert prof.threshold("motion_blur_kernel_px").value == 1.0
    it = item(run_conformance(ds, profile=prof), "motion_blur_kernel_px")
    # d ≈ 3 px > 收紧后的 1 px，且 α 未声明 → 只能 UNDETERMINED
    assert it.verdict == Verdict.UNDETERMINED
    assert "≤ 1 px" in it.note or "1 px" in it.note


# ---------------------------------------------------------------------------
# 3. 输入不足 → UNDETERMINED（措辞必须是"本地副本/本地环境"）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cameras", [("cam0",), ("cam0", "cam1")])
def test_missing_payload_files_is_undetermined(tmp_path: Path, cameras):
    """声明了视频却没有 payload 文件 → UNDETERMINED，不许当数据集缺陷。"""
    ds = write_dataset(tmp_path / "f", cameras=cameras, videos_with_files=False)
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.UNDETERMINED
    assert "本地副本不完整" in it.note
    assert "不是数据集本身" in it.note


def test_video_mode_off_is_undetermined(tmp_path: Path):
    """--video=off → 明确写着"没查"，不许默认是 sample。"""
    ds = dataset_with_real_video(tmp_path / "g", SHIFT_SMALL)
    result = run_conformance(ds, options=ConformOptions(video="off"))
    it = item(result, "motion_blur_kernel_px")
    assert it.verdict == Verdict.UNDETERMINED
    assert it.details["mode"] == "off"
    assert "--video=off" in it.note


def test_missing_av_is_undetermined_not_dataset_defect(tmp_path: Path, monkeypatch):
    """av 没装 → UNDETERMINED，措辞必须落在"本地环境"而不是数据集。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av" or name.startswith("av."):
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ds = dataset_with_real_video(tmp_path / "h", SHIFT_SMALL)
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.UNDETERMINED
    assert "本地环境未安装 av" in it.note
    assert "非数据集缺陷" in it.note


def test_broken_mp4_is_undetermined(tmp_path: Path):
    """16 字节假 mp4 必须导致 UNDETERMINED（不可解码 ≠ 通过、≠ 失败）。"""
    ds = dataset_with_real_video(tmp_path / "i", SHIFT_SMALL, broken_file=True)
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.UNDETERMINED
    assert it.details["measurement_status"] == "NO_DATA"
    assert it.details["cameras"]


def test_episode_without_local_file_is_skipped_not_borrowed(tmp_path: Path):
    """某个 episode 的载荷文件本地没有 → 跳过并披露，**不许拿别的 mp4 顶替**。

    顶替是最危险的一类错：同一个 mp4 里多个 episode 的 from/to 偏移不同，
    拿错文件会把别人的画面记成这个 episode 的位移，报告上却看不出来。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = dataset_with_real_video(tmp_path / "s", SHIFT_SMALL, n_episodes=2)
    meta = ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(meta)
    cols = tbl.to_pydict()
    key = "videos/observation.images.cam0/file_index"
    assert key in cols
    # episode 1 指向 file-001.mp4 —— 这份副本里不存在
    cols[key] = [0, 1]
    pq.write_table(pa.table(cols), meta)

    it = item(run_conformance(ds), "motion_blur_kernel_px")
    cam = it.details["cameras"][0]
    assert cam["n_episodes_used"] == 1, (
        f"只应有 1 个 episode 参与（另一个缺本地文件），实际 "
        f"{cam['n_episodes_used']} —— 出现即意味着拿错了文件")
    pool = it.details["episode_pool"]["observation.images.cam0"]
    assert pool["pool_size"] == 1 and pool["sampled"] == [0]
    # 抽样池受限这件事必须写进 note，不得默默缩小抽样范围
    assert "抽样池说明" in "".join(it.details["notes"])


def test_no_video_features_is_not_applicable(tmp_path: Path):
    ds = write_dataset(tmp_path / "j")
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# 4. per-feature 覆盖：不看 videos/ 目录在不在
# ---------------------------------------------------------------------------


def test_per_feature_partial_coverage_is_disclosed(tmp_path: Path):
    """一路相机有 payload、一路没有 → 照常裁定有的一路，并把缺的写在明面上。"""
    ds = dataset_with_real_video(tmp_path / "k", SHIFT_SMALL,
                                 cameras=("cam0", "cam1"))
    # cam1 的载荷删掉（videos/ 目录仍然存在，且 cam0 的文件还在）
    (ds / "videos" / "observation.images.cam1" / "chunk-000" / "file-000.mp4").unlink()
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    # videos/ 目录存在 ≠ 视频齐全：判定不得因为目录在就宣称覆盖
    assert (ds / "videos").exists()
    assert it.verdict == Verdict.PASS
    assert "observation.images.cam1" in it.details["features_without_payload"]
    cov = {f["feature"]: f["file_count"] for f in it.details["feature_coverage"]}
    assert cov["observation.images.cam0"] >= 1
    assert cov["observation.images.cam1"] == 0
    assert "未参与本次判定" in it.note


# ---------------------------------------------------------------------------
# 5. 附录 A 观测项
# ---------------------------------------------------------------------------


def test_appendix_a_optical_flow_is_measured_and_disclosed(tmp_path: Path):
    ds = dataset_with_real_video(tmp_path / "l", SHIFT_SMALL,
                                 cameras=("cam0", "cam2"))
    result = run_conformance(ds)
    flow = result.observations["appendix_a_optical_flow"]
    assert flow["status"] == "MEASURED"
    assert flow["binding"] == "—"
    assert flow["mode"] == "sample"
    # 每路相机各自的统计值 + 有效点/总点 + 帧对数
    for cam in flow["cameras"]:
        assert cam["displacement_px"]["p50"] is not None
        assert cam["displacement_px"]["p95"] is not None
        assert cam["displacement_px"]["max"] is not None
        assert cam["n_valid_points"] > 0
        assert cam["n_total_points"] >= cam["n_valid_points"]
        assert cam["n_pairs_used"] > 0
        assert cam["fps"] == pytest.approx(30.0)
    # 抽样规模必须写出来
    sp = flow["sampling"]
    assert len(sp["episodes_sampled"]) == 2
    assert sp["video_frame_pairs"] == 8
    assert sp["seed"] == 0
    # 临临文件：明确写着没有产生
    assert "未产生任何临时文件" in flow["temp_files"]
    # 观测项不产四态：不得出现在判定表里
    assert flow["clause"] == "附录 A.1"


def test_appendix_a_spectral_flux_not_applicable_without_audio(tmp_path: Path):
    """无音频特征 → NOT_APPLICABLE，且必须说明 A.2 原文讲的是音频信号。"""
    ds = dataset_with_real_video(tmp_path / "m", SHIFT_SMALL)
    sf = run_conformance(ds).observations["appendix_a_spectral_flux"]
    assert sf["status"] == "NOT_APPLICABLE"
    assert sf["clause"] == "附录 A.2"
    assert "音频信号" in sf["note"]
    assert "频谱流量量化了频谱在时间上的变化" in sf["note"]
    # 严禁把 2D FFT 拿糊上去：MEASURED 只能出现在有音频列的时候
    assert sf["feature"] is None


def test_appendix_a_pointcloud_not_applicable_without_pointcloud(tmp_path: Path):
    ds = dataset_with_real_video(tmp_path / "n", SHIFT_SMALL)
    pc = run_conformance(ds).observations["appendix_a_pointcloud_dynamics"]
    assert pc["status"] == "NOT_APPLICABLE"
    assert pc["clause"] == "附录 A.3"
    assert pc["feature"] is None


# ---------------------------------------------------------------------------
# 6. 报告与 CLI
# ---------------------------------------------------------------------------


def test_report_has_appendix_a_chapter_and_renumbered_sections(tmp_path: Path):
    ds = dataset_with_real_video(tmp_path / "o", SHIFT_SMALL)
    md = render_conformance_markdown(run_conformance(ds))
    # 新增章节按任务书编号为第 5 章，附录 B 仍是第 4 章，原 5/6/7 顺延为 6/7/8
    assert "## 4. 观测项（附录 B" in md
    assert "## 5. 观测项（附录 A" in md
    assert "### 附录 A.1" in md and "### 附录 A.2" in md and "### 附录 A.3" in md
    assert "## 6. 本次判定生效的阈值" in md
    assert "## 7. 数据集事实（用于复核）" in md
    assert "## 8. 门禁状态与遗留项" in md
    assert "## 5. 本次判定生效的阈值" not in md  # 旧编号不得残留
    # 每路相机的统计 + 抽样披露 + 临时文件声明都要进正文
    assert "observation.images.cam0" in md
    assert "临时文件" in md
    assert "motion_blur_kernel_px" in md


def _cli_node(tmp_path: Path, ds: Path, *extra: str) -> dict:
    """Run ``rda conform`` and return the parsed JSON node."""
    out = tmp_path / "reports"
    runner = CliRunner()
    args = ["conform", str(ds), "--format", "json", "--out", str(out), *extra]
    r = runner.invoke(cli, args)
    assert r.exit_code in (0, 1), r.output
    report = out / "conformance_report_gbz218.1-2026.json"
    assert report.exists(), f"报告未写出：{sorted(p.name for p in out.glob('*'))}"
    return json.loads(report.read_text(encoding="utf-8"))


def _node_item(node: dict, metric: str) -> dict:
    for it in node["items"]:
        if it["metric"] == metric:
            return it
    raise AssertionError(f"metric {metric!r} not in node")


def test_report_flags_max_as_single_point_and_gives_real_threshold(tmp_path: Path):
    """max 必须被标成单点极值，剔除规则必须写出**实际 px 门限**。

    只写"对角线 × 0.10"读者看不出这条线有多松；只给 max 不给提示，第三方
    会把 15 万点里的 1 个野值读成"存在 75 px 的帧间位移"。
    """
    ds = dataset_with_real_video(tmp_path / "t", SHIFT_SMALL)
    result = run_conformance(ds)
    cam = result.observations["appendix_a_optical_flow"]["cameras"][0]
    # --- 结构化证据（JSON 节点）-------------------------------------------
    assert cam["displacement_px"]["p99_9"] is not None
    # n_above_p99 与其余分位数同在 displacement_px 段里：曾因放在顶层而让报告
    # 恒为 0（真实数据上应为 153/15262），故一并钉住键的位置与类型。
    assert isinstance(cam["displacement_px"]["n_above_p99"], int)
    assert isinstance(cam["displacement_px"]["p99"], float)
    assert cam["track_err"]["n"] > 0            # LK 残差被接出来了，没丢
    assert cam["track_err"]["p50"] is not None
    assert cam["frame_shape"] == "320×240"
    # 320×240 → 对角线 400 px → 门限 40.00 px
    assert cam["reject_limit_px"] == pytest.approx(40.0)
    assert cam["diagonal_px"] == pytest.approx(400.0)

    # --- 报告正文 ---------------------------------------------------------
    md = render_conformance_markdown(result)
    assert "单点极值" in md                      # max 列脚注
    assert "p99.9" in md
    assert "max (px) ⚠" in md
    assert "40.00 px" in md                      # 剔除门限的实际数值
    assert "不是'跟得准'的门槛" in md or "不是「跟得准」的门槛" in md
    assert "LK 残差" in md
    # 逐 episode 明细同样要给 err@max 与 p99.9
    assert "err@max" in md


def test_cli_video_off_leaves_the_item_undetermined(tmp_path: Path):
    ds = dataset_with_real_video(tmp_path / "p", SHIFT_SMALL)
    node = _cli_node(tmp_path / "off", ds, "--video=off")
    assert _node_item(node, "motion_blur_kernel_px")["verdict"] == "UNDETERMINED"
    assert node["observations"]["appendix_a_optical_flow"]["status"] == "OFF"


def test_cli_default_is_sample(tmp_path: Path):
    """默认档位必须是 sample：不写 --video 也应该真的去读了像素。"""
    ds = dataset_with_real_video(tmp_path / "q", SHIFT_SMALL)
    node = _cli_node(tmp_path / "sample", ds)
    assert _node_item(node, "motion_blur_kernel_px")["verdict"] == "PASS"
    flow = node["observations"]["appendix_a_optical_flow"]
    assert flow["mode"] == "sample"
    assert flow["cameras"], "sample 档位下应当真的解出了帧对"


def test_cli_full_mode_walks_every_episode(tmp_path: Path):
    """full = 全部 episode（帧对数仍受 video_frame_pairs 限制）。"""
    ds = dataset_with_real_video(tmp_path / "r", SHIFT_SMALL, n_episodes=3)
    node = _cli_node(tmp_path / "full", ds, "--video=full")
    flow = node["observations"]["appendix_a_optical_flow"]
    assert flow["mode"] == "full"
    assert len(flow["sampling"]["episodes_sampled"]) == 3
    for cam in flow["cameras"]:
        assert cam["n_episodes_used"] == 3
