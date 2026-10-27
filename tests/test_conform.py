"""rda conform 的判定语义测试。

测试重点（按红线排序）：
  1. 四态各一例；
  2. 阈值边界值（恰好等于阈值时 comparator 的行为）；
  3. 字段缺失（含"列根本不存在"）；
  4. 空 episode / 空数据集 → 必须是 UNDETERMINED，绝不能是 PASS；
  5. 自定义 profile 覆盖生效，且不得丢掉 clause / binding。

这里的 fixture 是**真实写盘的最小 LeRobot v3.0 数据集**（真 parquet、
真 info.json），不是 mock —— 缺列、缺视频目录这些判定都建立在真实文件
结构上，用 mock 会测不到。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from rda.cli.main import cli
from rda.conform import (
    ConformOptions,
    ConformanceItem,
    Verdict,
    compare,
    default_profile,
    load_profile,
    run_conformance,
    summarize,
)

STATE_DIM = 4

CORE_FIELDS = ("timestamp", "frame_index", "action", "observation.state")


# ---------------------------------------------------------------------------
# fixture：写一个最小 LeRobot v3.0 数据集
# ---------------------------------------------------------------------------


def write_dataset(
    root: Path,
    *,
    n_episodes: int = 1,
    frames_per_ep: int = 10,
    fps: int = 30,
    nulls: Optional[Dict[str, int]] = None,
    drop_columns: Sequence[str] = (),
    n_tasks: int = 1,
    cameras: Sequence[str] = (),
    video_from: Optional[Dict[str, Dict[int, float]]] = None,
    video_duration_delta: Optional[Dict[str, Dict[int, float]]] = None,
    videos_dir: bool = False,
    videos_with_files: bool = False,
    readme: Optional[str] = None,
    info_extra: Optional[dict] = None,
    state_value=None,
) -> Path:
    """Write a minimal LeRobot v3.0 dataset to ``root`` and return it.

    Args:
        nulls: ``{column: n}`` — 该列末尾 n 行置为 null。
        drop_columns: 完全不写这些列（模拟"字段根本不存在"）。
        cameras: 声明的视觉流 key；会写入 info.features 与 episodes meta
            的 videos/<key>/{from,to}_timestamp。
        video_from: ``{camera: {episode_index: from_timestamp}}``。模拟 LeRobot
            v3 的真实语义——**文件内相对偏移**：各相机分块不同，同一 episode
            在不同相机里偏移不同是常态，偏移不等 ≠ 时间基准不同。
        video_duration_delta: ``{camera: {episode_index: delta}}``，加到该相机
            该 episode 的时长上。只有**时长**才是跨视角可比量，时长不一致才是
            真正的时间基准问题。
        videos_dir: 是否创建空的 ``videos/`` 目录。
        videos_with_files: 是否在每个 ``videos/<feature>/`` 下写真实文件
            （模拟视频真的下载下来了）。
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    nulls = nulls or {}
    total = n_episodes * frames_per_ep
    ep_col: List[int] = []
    ts_col: List[float] = []
    fi_col: List[int] = []
    act_col: List[Optional[List[float]]] = []
    st_col: List[Optional[List[float]]] = []
    ti_col: List[int] = []
    for e in range(n_episodes):
        for f in range(frames_per_ep):
            ep_col.append(e)
            ts_col.append(round(f / fps, 9))
            fi_col.append(f)
            base = 0.0 if state_value is None else float(state_value(e))
            act_col.append([base + 0.1 * f] * STATE_DIM)
            st_col.append([base + 0.1 * f] * STATE_DIM)
            ti_col.append(e % max(n_tasks, 1))

    def with_nulls(name: str, values: list):
        n = int(nulls.get(name, 0))
        if n:
            values = values[:-n] + [None] * n
        return pa.array(values)

    columns = {
        "episode_index": pa.array(ep_col),
        "timestamp": with_nulls("timestamp", ts_col),
        "frame_index": with_nulls("frame_index", fi_col),
        "action": with_nulls("action", act_col),
        "observation.state": with_nulls("observation.state", st_col),
        "index": with_nulls("index", list(range(total))),
        "task_index": with_nulls("task_index", ti_col),
    }
    table = pa.table({k: v for k, v in columns.items()
                      if k not in set(drop_columns)})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    # --- episodes meta ---
    ep_rows: Dict[str, list] = {
        "episode_index": list(range(n_episodes)),
        "length": [frames_per_ep] * n_episodes,
        "data/chunk_index": [0] * n_episodes,
        "data/file_index": [0] * n_episodes,
        "dataset_from_index": [e * frames_per_ep for e in range(n_episodes)],
        "dataset_to_index": [(e + 1) * frames_per_ep for e in range(n_episodes)],
    }
    for cam in cameras:
        dur = frames_per_ep / fps
        ep_rows[f"videos/observation.images.{cam}/chunk_index"] = [0] * n_episodes
        ep_rows[f"videos/observation.images.{cam}/file_index"] = [0] * n_episodes
        fr_vals = []
        to_vals = []
        for e in range(n_episodes):
            fr = 0.0
            if video_from and cam in video_from and e in video_from[cam]:
                fr = float(video_from[cam][e])
            delta = 0.0
            if video_duration_delta and cam in video_duration_delta \
                    and e in video_duration_delta[cam]:
                delta = float(video_duration_delta[cam][e])
            fr_vals.append(fr)
            to_vals.append(fr + dur + delta)
        ep_rows[f"videos/observation.images.{cam}/from_timestamp"] = fr_vals
        ep_rows[f"videos/observation.images.{cam}/to_timestamp"] = to_vals
    pq.write_table(pa.table(ep_rows),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # --- tasks ---
    pq.write_table(pa.table({"task_index": list(range(max(n_tasks, 1)))}),
                   root / "meta" / "tasks.parquet")

    # --- info.json ---
    features: Dict[str, dict] = {
        "action": {"dtype": "float32", "shape": [STATE_DIM]},
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_episodes": n_episodes,
        "total_frames": total,
        "total_tasks": n_tasks,
        "fps": fps,
        "features": features,
    }
    if info_extra:
        info.update(info_extra)
    (root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if videos_dir or videos_with_files:
        (root / "videos").mkdir(exist_ok=True)
    if videos_with_files:
        for cam in cameras:
            d = root / "videos" / f"observation.images.{cam}" / "chunk-000"
            d.mkdir(parents=True, exist_ok=True)
            (d / "file-000.mp4").write_bytes(b"\x00" * 16)  # 占位，不校验可解码性
    if readme is not None:
        (root / "README.md").write_text(readme, encoding="utf-8")
    return root


def item(result, metric: str) -> ConformanceItem:
    for it in result.items:
        if it.metric == metric:
            return it
    raise AssertionError(f"metric {metric!r} not found in "
                         f"{[i.metric for i in result.items]}")


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """单 episode、10 帧、无相机、无 README 的最小数据集。"""
    return write_dataset(tmp_path / "ds")


# ---------------------------------------------------------------------------
# 1. 四态各一例
# ---------------------------------------------------------------------------


def test_four_states_all_appear_in_one_run(tmp_path: Path):
    """一次运行里必须同时出现 PASS / FAIL / NOT_APPLICABLE / UNDETERMINED。"""
    ds = write_dataset(
        tmp_path / "ds",
        n_episodes=2, frames_per_ep=12, fps=30,
        n_tasks=1,                       # → 6.2.3.2 FAIL（宜）
        cameras=("cam0", "cam1"),
        # 两个相机的**时长**相差 2 s → 6.1.1 f FAIL（应）。
        # 注意：不能用 video_from 造 FAIL —— from/to 是文件内相对偏移，
        # 偏移不等只是分块差异，不是时间基准问题。
        video_duration_delta={"cam1": {0: 2.0, 1: 2.0}},
    )
    result = run_conformance(ds)
    states = {it.verdict for it in result.items}
    assert states == {Verdict.PASS, Verdict.FAIL,
                      Verdict.NOT_APPLICABLE, Verdict.UNDETERMINED}

    # 逐态定位到具体条款，避免"恰好凑出四态"
    assert item(result, "core_field_missing_rate").verdict == Verdict.PASS
    assert item(result, "task_type_kinds").verdict == Verdict.FAIL
    assert item(result, "multimodal_time_base_unified").verdict == Verdict.FAIL
    # 未声明可行解空间 → 必须 UNDETERMINED，不许拿 Hausdorff 绝对值凑阈值
    h = item(result, "trajectory_hausdorff_ratio")
    assert h.verdict == Verdict.UNDETERMINED
    assert h.value is None
    assert "可行解空间" in h.note
    # 视频层已实现（v0.11.0），但这份本地副本没有 videos/ 载荷：必须判
    # UNDETERMINED —— "没查"不许伪装成 PASS，也不许记成数据集的 FAIL。
    mb = item(result, "motion_blur_kernel_px")
    assert mb.verdict == Verdict.UNDETERMINED
    assert "本地副本不完整" in mb.note


def test_pass_state_is_reachable(tmp_path: Path):
    """字段齐全 + 多任务 + 无相机 + 说明文档齐备 → 应类条款全 PASS。"""
    ds = write_dataset(
        tmp_path / "ds", n_episodes=3, frames_per_ep=12, n_tasks=3,
        # 6.2.2.1 c 要求"应提供超阈值数据处理方式说明"：必须真的写了才算
        readme="## 数据说明\n异常值与超阈值数据采用裁剪（clipping）处理。\n",
        info_extra={"doi": "10.1234/rda.test.dataset"},
    )
    result = run_conformance(ds)
    assert item(result, "core_field_missing_rate").verdict == Verdict.PASS
    assert item(result, "core_field_missing_rate").value == 0.0
    assert item(result, "task_type_kinds").verdict == Verdict.PASS
    assert item(result, "task_type_kinds").value == 3
    assert item(result, "out_of_threshold_handling_declared").verdict == Verdict.PASS
    assert item(result, "doi_uri_present").verdict == Verdict.PASS
    assert result.summary["mandatory_fail"] == 0


def test_motion_blur_not_applicable_without_video_features(tmp_path: Path):
    """数据集根本没声明视频流特征 → NOT_APPLICABLE（条款无从适用）。"""
    ds = write_dataset(tmp_path / "ds")
    it = item(run_conformance(ds), "motion_blur_kernel_px")
    assert it.verdict == Verdict.NOT_APPLICABLE
    assert "未声明" in it.note


def test_undetermined_states_are_not_counted_as_pass(tmp_path: Path):
    """UNDETERMINED 不得进入 PASS 计数。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12)
    result = run_conformance(ds)
    summary = result.summary
    undet = [it for it in result.items if it.verdict == Verdict.UNDETERMINED]
    assert undet
    assert summary["undetermined"] == len(undet)
    assert all(it.metric in summary["undetermined_metrics"] for it in undet)
    assert not any(it.verdict == Verdict.PASS for it in undet)


# ---------------------------------------------------------------------------
# 2. 阈值边界值
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comparator,value,threshold,expected",
    [
        ("<", 0.05, 0.05, False),     # 恰好等于 → "<" 不满足
        ("<", 0.0499, 0.05, True),
        ("<=", 0.05, 0.05, True),     # 恰好等于 → "<=" 满足
        ("<=", 0.0501, 0.05, False),
        (">=", 2, 2, True),           # 恰好等于 → ">=" 满足
        (">=", 1, 2, False),
        (">", 2, 2, False),           # 恰好等于 → ">" 不满足
        ("==", 1, 1, True),
        ("is_true", True, True, True),
        ("is_true", False, True, False),
    ],
)
def test_compare_at_exact_threshold(comparator, value, threshold, expected):
    assert compare(value, threshold, comparator) is expected


def test_compare_returns_none_when_value_missing():
    """value 为 None 必须返回 None（调用方据此走 UNDETERMINED）。"""
    assert compare(None, 0.05, "<") is None
    assert compare(None, True, "is_true") is None


def test_compare_returns_none_on_unknown_comparator():
    assert compare(1.0, 1.0, "≈") is None


def test_core_missing_rate_exactly_at_threshold_fails(tmp_path: Path):
    """核心字段缺失率恰好 5%：comparator 是 "<"，恰好等于即不满足。"""
    # 20 行 × 4 个核心字段 = 80 个单元；4 个 null → 恰好 5%
    ds = write_dataset(tmp_path / "ds", n_episodes=1, frames_per_ep=20,
                       nulls={"frame_index": 2, "action": 2})
    it = item(run_conformance(ds), "core_field_missing_rate")
    assert it.value == pytest.approx(0.05)
    assert it.verdict == Verdict.FAIL
    assert it.binding == "应"


def test_core_missing_rate_one_cell_below_threshold_passes(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=1, frames_per_ep=20,
                       nulls={"frame_index": 2, "action": 1})
    it = item(run_conformance(ds), "core_field_missing_rate")
    assert it.value == pytest.approx(0.0375)
    assert it.verdict == Verdict.PASS


def test_frame_loss_rate_exactly_at_threshold_passes(tmp_path: Path):
    """丢帧率 comparator 是 "<="，恰好 1% 即满足。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=1, frames_per_ep=101, fps=30)
    # 手工造出 100 个间隔，其中 1 个超阈值 → 恰好 1%
    ts = np.arange(101, dtype=np.float64) / 30.0
    ts[50] = ts[49] + 1.0  # 一个巨大的间隔 → 1 个丢帧
    _rewrite_timestamps(ds, ts)
    it = item(run_conformance(ds), "consecutive_frame_loss_rate")
    assert it.value == pytest.approx(0.01)
    assert it.verdict == Verdict.PASS
    assert it.binding == "宜"


def test_task_kinds_exactly_at_threshold_passes(tmp_path: Path):
    """任务种类数 comparator 是 ">="，恰好 2 种即满足。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=4, frames_per_ep=8, n_tasks=2)
    it = item(run_conformance(ds), "task_type_kinds")
    assert it.value == 2
    assert it.verdict == Verdict.PASS


def test_task_kinds_one_below_threshold_fails_as_advisory(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=4, frames_per_ep=8, n_tasks=1)
    it = item(run_conformance(ds), "task_type_kinds")
    assert it.verdict == Verdict.FAIL
    assert it.binding == "宜"
    assert "宜类条款（建议性）" in it.note


def _rewrite_timestamps(root: Path, ts: np.ndarray) -> None:
    """把 data parquet 的 timestamp 列整体替换（用于构造精确边界）。"""
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    cols = {name: table.column(name) for name in table.schema.names}
    cols["timestamp"] = pa.array(ts.tolist())
    pq.write_table(pa.table(cols), path)


# ---------------------------------------------------------------------------
# 3. 字段缺失
# ---------------------------------------------------------------------------


def test_missing_core_column_counts_as_fully_missing(tmp_path: Path):
    """列根本不存在 → 该核心字段在所有帧上缺失 → 单元级缺失率 25% → FAIL。"""
    ds = write_dataset(tmp_path / "ds", frames_per_ep=10,
                       drop_columns=["timestamp"])
    it = item(run_conformance(ds), "core_field_missing_rate")
    # 4 个核心字段 × 10 帧 = 40 个单元，timestamp 整列缺失 → 10/40 = 25%
    assert it.value == pytest.approx(0.25)
    assert it.verdict == Verdict.FAIL
    assert "列不存在" in it.details["per_field"]["timestamp"].get("note", "")
    assert it.details["per_field"]["timestamp"]["rate"] == 1.0


def test_undownloaded_video_payload_is_undetermined_not_fail(tmp_path: Path):
    """视频层没下载 → 不许判 FAIL，必须 UNDETERMINED 并写明原因。

    这是本模块最要紧的一条红线：本地副本缺 videos/ 是我们自己的下载策略
    造成的，不是数据集的字段缺失。把它算成"全帧缺失"去违反 6.2.2.1 b 的
    "应"类条款，等于拿"我没拿到数据"去指控"你的数据不合格"。
    """
    ds = write_dataset(tmp_path / "ds", cameras=("cam0",))
    it = item(run_conformance(ds), "general_field_missing_rate")
    assert it.verdict == Verdict.UNDETERMINED
    # 不下结论就绝不能给出判定的数值，否则会被读成"缺失率 = X"
    assert it.value is None
    assert "视频层未下载" in it.note
    assert "不是数据集本身的字段缺失" in it.note
    assert "请提供含 videos/ 的完整数据集后复测" in it.note
    assert it.details["undownloaded_payload_features"] == ["observation.images.cam0"]
    # 可核验部分的对照值要一并给出，但只能作参考
    assert it.details["rate_parquet_columns_only"] == pytest.approx(0.0)


def test_empty_videos_dir_is_still_undetermined(tmp_path: Path):
    """videos/ 目录存在但里面没有文件 → 同样是不可核验，不是"零缺失"。"""
    ds = write_dataset(tmp_path / "ds", cameras=("cam0",), videos_dir=True)
    it = item(run_conformance(ds), "general_field_missing_rate")
    assert it.verdict == Verdict.UNDETERMINED
    assert it.details["videos_dir_present"] is True
    assert it.details["videos_dir_file_count"] == 0


def test_declared_video_feature_with_payload_is_not_missing(tmp_path: Path):
    """视频文件确实在 → 该特征按已采集计入，缺失率 0，可正常判定。"""
    ds = write_dataset(tmp_path / "ds", cameras=("cam0",), videos_with_files=True)
    it = item(run_conformance(ds), "general_field_missing_rate")
    assert it.value == pytest.approx(0.0)
    assert it.verdict == Verdict.PASS
    assert it.details["undownloaded_payload_features"] == []


# ---------------------------------------------------------------------------
# 3b. 6.1.1 f 多模态统一时间基准：可比量是时长，不是 from/to 偏移
# ---------------------------------------------------------------------------


def test_video_offset_alone_is_not_a_time_base_violation(tmp_path: Path):
    """各相机 from/to 偏移不同（分块差异）→ 不得判 6.1.1 f FAIL。

    LeRobot v3 的 from/to 是**视频文件内相对偏移**：每个文件的首个 episode
    from 恒为 0，同文件内相邻 episode 首尾相接。各相机分块数不同时，同一
    episode 的偏移必然不同。旧实现直接跨视角相减，得到过 1033 s 的伪偏差，
    把一个时间基准完全一致的数据集判成了违反"应"类条款。
    """
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12,
                       cameras=("cam0", "cam1"),
                       video_from={"cam1": {0: 500.0, 1: 500.0}})
    it = item(run_conformance(ds), "multimodal_time_base_unified")
    assert it.verdict == Verdict.PASS, it.note
    # 伪值必须仍然作为参考信息暴露出来，但不得参与判定
    assert it.details["cross_camera_offset_max_deviation_s_NOT_A_METRIC"] == \
        pytest.approx(500.0)
    assert it.details["cross_camera_max_duration_deviation_s"] == pytest.approx(0.0)


def test_video_duration_mismatch_is_a_time_base_violation(tmp_path: Path):
    """各相机**时长**不一致 → 这才是真正的时间基准问题，判 FAIL。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12,
                       cameras=("cam0", "cam1"),
                       video_duration_delta={"cam1": {0: 2.0, 1: 2.0}})
    it = item(run_conformance(ds), "multimodal_time_base_unified")
    assert it.verdict == Verdict.FAIL
    assert it.details["cross_camera_max_duration_deviation_s"] == pytest.approx(2.0)


def test_video_layer_vs_frame_layer_duration_is_compared_with_one_period(tmp_path: Path):
    """视频时长 = 帧层跨度 + 1 个标称周期；只差这 1 个周期不算偏差。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12, fps=30,
                       cameras=("cam0",), video_from={"cam0": {0: 7.0, 1: 7.0}})
    it = item(run_conformance(ds), "multimodal_time_base_unified")
    assert it.verdict == Verdict.PASS, it.note
    # 12 帧 → 帧层跨度 11/30 s；视频时长 12/30 s；补 1 个周期后应为 0
    # （残留的是 fixture 里 round(f/fps, 9) 的浮点噪声，量级 1e-10 s）
    assert it.details["frame_vs_video_max_duration_deviation_s"] == \
        pytest.approx(0.0, abs=1e-6)


def test_general_field_nulls_are_counted(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", frames_per_ep=20, nulls={"index": 10})
    it = item(run_conformance(ds), "general_field_missing_rate")
    # 一般字段 = episode_index / index / task_index → 10/(20*3) ≈ 16.7% < 20%
    assert it.value == pytest.approx(10 / 60)
    assert it.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# 4. 空 episode / 空数据集
# ---------------------------------------------------------------------------


def test_empty_dataset_never_passes(tmp_path: Path):
    """0 帧数据集：任何判定项都不许给出 PASS。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=1, frames_per_ep=0)
    result = run_conformance(ds)
    assert result.summary["pass"] == 0
    assert item(result, "core_field_missing_rate").verdict == Verdict.UNDETERMINED
    assert item(result, "timestamp_jitter_std").verdict == Verdict.UNDETERMINED
    assert item(result, "consecutive_frame_loss_rate").verdict == Verdict.UNDETERMINED
    for it in result.items:
        if it.verdict == Verdict.UNDETERMINED:
            assert it.note  # 必须写明缺什么


def test_single_frame_episode_has_no_jitter_to_judge(tmp_path: Path):
    """1 帧 episode 没有帧间隔 → 抖动/丢帧判不了，必须 UNDETERMINED。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=1, frames_per_ep=1)
    result = run_conformance(ds)
    assert item(result, "timestamp_jitter_std").verdict == Verdict.UNDETERMINED
    assert item(result, "consecutive_frame_loss_rate").verdict == Verdict.UNDETERMINED
    assert result.summary["pass"] <= 2  # 只有不依赖帧间隔的项可以 PASS


def test_missing_fps_makes_jitter_undetermined(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", frames_per_ep=10)
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["fps"]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    result = run_conformance(ds)
    it = item(result, "timestamp_jitter_std")
    assert it.verdict == Verdict.UNDETERMINED
    assert "fps" in it.note


def test_no_readme_and_no_info_scalars_is_undetermined(tmp_path: Path):
    """连说明载体都没有 → 6.2.2.1 c 判 UNDETERMINED，不许默认 PASS。"""
    ds = write_dataset(tmp_path / "ds", frames_per_ep=10)
    (ds / "meta" / "info.json").write_text("{}", encoding="utf-8")
    result = run_conformance(ds)
    assert item(result, "out_of_threshold_handling_declared").verdict == (
        Verdict.UNDETERMINED)


# ---------------------------------------------------------------------------
# 5. 自定义 profile 覆盖
# ---------------------------------------------------------------------------


def test_profile_override_changes_verdict(tmp_path: Path):
    """同一份 2% 缺失率的数据：默认阈值 PASS，收紧到 1% 后 FAIL。"""
    # 100 行 × 4 核心字段 = 400 单元；8 个 null → 恰好 2%
    ds = write_dataset(tmp_path / "ds", frames_per_ep=100,
                       nulls={"frame_index": 4, "action": 4})

    loose = run_conformance(ds)
    assert item(loose, "core_field_missing_rate").value == pytest.approx(0.02)
    assert item(loose, "core_field_missing_rate").verdict == Verdict.PASS

    profile_path = tmp_path / "strict.json"
    profile_path.write_text(json.dumps({
        "thresholds": {"core_field_missing_rate": {"value": 0.01}},
    }), encoding="utf-8")
    strict = run_conformance(ds, profile=load_profile(profile_path))
    assert item(strict, "core_field_missing_rate").verdict == Verdict.FAIL


def test_profile_override_keeps_clause_and_binding():
    """只覆盖 value 时，clause / binding / comparator 必须沿用标准原文。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.json"
        p.write_text(json.dumps({
            "thresholds": {"core_field_missing_rate": {"value": 0.02}},
        }), encoding="utf-8")
        profile = load_profile(p)
    th = profile.threshold("core_field_missing_rate")
    assert th.value == 0.02
    assert th.clause == "6.2.2.1 a"
    assert th.binding == "应"
    assert th.comparator == "<"
    assert profile.source == str(p)


def test_profile_override_sampling_is_respected(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=4, frames_per_ep=20)
    profile = default_profile()
    profile.sampling.dtw_episodes = 2
    profile.sampling.dtw_max_pairs = 1
    profile.sampling.hausdorff_episodes = 2
    result = run_conformance(ds, profile=profile)
    obs = result.observations
    assert obs["dtw"]["n_pairs"] == 1
    assert obs["trajectory_hausdorff"]["n_pairs"] == 1


def test_load_profile_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nope.json")


def test_load_profile_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_profile(p)


def test_load_profile_unknown_field(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"nonsense_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown profile field"):
        load_profile(p)


def test_load_profile_bad_binding(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({
        "thresholds": {"core_field_missing_rate": {"value": 0.1,
                                                   "binding": "必须"}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        load_profile(p)


def test_default_profile_uses_standard_values():
    p = default_profile()
    assert p.standard == "gbz218.1-2026"
    assert p.threshold("core_field_missing_rate").value == 0.05
    assert p.threshold("general_field_missing_rate").value == 0.20
    assert p.threshold("timestamp_jitter_std_ratio").value == 0.10
    assert p.threshold("consecutive_frame_loss_rate").value == 0.01
    assert p.threshold("sensor_alignment_error_s").value == 0.001
    assert p.threshold("task_type_min_kinds").value == 2
    assert p.threshold("trajectory_hausdorff_ratio").value == 0.30
    assert p.source is None


def test_every_threshold_has_clause_and_binding():
    """常量表里不允许出现没有条款出处的裸数字。"""
    from rda.conform.standards.gbz218_1 import all_thresholds

    table = all_thresholds()
    assert table
    for key, th in table.items():
        assert th.clause, f"{key} 缺 clause"
        assert th.binding in ("应", "宜", "—"), f"{key} binding 非法: {th.binding}"


# ---------------------------------------------------------------------------
# 6. 6.2.3.4 a 的可行解空间声明
# ---------------------------------------------------------------------------


def test_hausdorff_undetermined_without_workspace_declaration(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=3, frames_per_ep=20)
    result = run_conformance(ds)
    it = item(result, "trajectory_hausdorff_ratio")
    assert it.verdict == Verdict.UNDETERMINED
    assert it.value is None
    # 观测值仍要给出，供人工复核，但不参与判定
    assert result.observations["trajectory_hausdorff"]["mean"] is not None


def test_hausdorff_judged_when_workspace_declared(tmp_path: Path):
    """声明可行解空间后必须真的下结论（不能永远 UNDETERMINED）。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=20,
                       state_value=lambda e: 0.0 if e == 0 else 50.0)

    far = run_conformance(ds, options=ConformOptions(workspace_max_distance=100.0))
    it = item(far, "trajectory_hausdorff_ratio")
    assert it.verdict == Verdict.PASS
    assert it.value >= 0.30

    near = run_conformance(ds, options=ConformOptions(workspace_max_distance=10000.0))
    it2 = item(near, "trajectory_hausdorff_ratio")
    assert it2.verdict == Verdict.FAIL
    assert it2.value < 0.30
    assert "宜类条款（建议性）" in it2.note


def test_workspace_bounds_box_diagonal(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=20,
                       state_value=lambda e: 0.0 if e == 0 else 1.0)
    opts = ConformOptions(workspace_bounds=[0, 1, 0, 1, 0, 1])
    assert opts.resolved_max_distance() == pytest.approx(float(np.sqrt(3.0)))
    result = run_conformance(ds, options=opts)
    assert item(result, "trajectory_hausdorff_ratio").verdict in (
        Verdict.PASS, Verdict.FAIL)


def test_workspace_bounds_bad_arity_is_none():
    assert ConformOptions(workspace_bounds=[0, 1, 2]).resolved_max_distance() is None
    assert ConformOptions().resolved_max_distance() is None


# ---------------------------------------------------------------------------
# 7. 汇总与严重性
# ---------------------------------------------------------------------------


def test_summarize_separates_advisory_from_mandatory():
    items = [
        ConformanceItem("6.2.2.1 a", "a", 1, "t", "应", Verdict.FAIL),
        ConformanceItem("6.2.3.2", "b", 1, "t", "宜", Verdict.FAIL),
        ConformanceItem("6.2.2.3", "c", 1, "t", "宜", Verdict.PASS),
        ConformanceItem("6.2.3.5", "d", 1, "t", "—", Verdict.NOT_APPLICABLE),
        ConformanceItem("6.2.3.4 a", "e", None, "t", "宜", Verdict.UNDETERMINED),
    ]
    s = summarize(items)
    assert s["total"] == 5
    assert s["fail"] == 2
    assert s["mandatory_fail"] == 1
    assert s["advisory_fail"] == 1
    assert s["informational"] == 1
    assert s["undetermined"] == 1
    assert s["undetermined_metrics"] == ["e"]
    assert s["coverage"] == pytest.approx(3 / 5)


def test_overall_verdict_precedence():
    from rda.conform.verdict import overall_verdict

    p = ConformanceItem("c", "m", 1, "t", "应", Verdict.PASS)
    u = ConformanceItem("c", "m2", None, "t", "应", Verdict.UNDETERMINED)
    f = ConformanceItem("c", "m3", 0, "t", "应", Verdict.FAIL)
    assert overall_verdict([p, u]) == Verdict.UNDETERMINED  # 没查实 ≠ 符合
    assert overall_verdict([p]) == Verdict.PASS
    assert overall_verdict([f, u]) == Verdict.FAIL  # 已查实的不符合优先
    assert overall_verdict([]) == Verdict.UNDETERMINED


def test_conformance_item_to_dict_is_json_safe():
    it = ConformanceItem("6.2.2.1 a", "m", np.float64(0.5), "t", "应",
                         Verdict.FAIL, details={"n": np.int64(3)})
    d = json.dumps(it.to_dict(), default=str)
    assert "FAIL" in d


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def test_cli_conform_exit_1_on_mandatory_fail(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12,
                       n_tasks=1, cameras=("cam0", "cam1"),
                       video_from={"cam1": {0: 500.0, 1: 500.0}})
    out = tmp_path / "reports"
    r = CliRunner().invoke(cli, ["conform", str(ds), "--out", str(out)])
    assert r.exit_code == 1, r.output
    assert (out / "conformance_report_gbz218.1-2026.md").exists()


def test_cli_conform_exit_2_on_missing_path(tmp_path: Path):
    r = CliRunner().invoke(cli, ["conform", str(tmp_path / "nope")])
    assert r.exit_code == 2


def test_cli_conform_exit_2_on_bad_profile(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    r = CliRunner().invoke(cli, ["conform", str(ds), "--profile", str(bad)])
    assert r.exit_code == 2
    assert "invalid profile" in r.output


def test_cli_conform_exit_2_on_bad_workspace_bounds(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds")
    r = CliRunner().invoke(cli, ["conform", str(ds), "--workspace-bounds", "1,2"])
    assert r.exit_code == 2
    assert "workspace-bounds" in r.output


def test_cli_conform_json_output(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12)
    out = tmp_path / "reports"
    r = CliRunner().invoke(cli, ["conform", str(ds), "--format", "json",
                                 "--out", str(out)])
    payload = json.loads((out / "conformance_report_gbz218.1-2026.json")
                         .read_text(encoding="utf-8"))
    assert payload["standard"] == "gbz218.1-2026"
    assert "summary" in payload and "items" in payload
    assert payload["coverage_gate"]["status"] == "TODO_NOT_IMPLEMENTED"
    assert r.exit_code in (0, 1)


def test_cli_conform_strict_coverage_is_stubbed(tmp_path: Path):
    """--strict-coverage 本轮不得假装生效（不许返回 3）。"""
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12)
    r = CliRunner().invoke(cli, ["conform", str(ds), "--strict-coverage",
                                 "--out", str(tmp_path / "o")])
    assert r.exit_code != 3
    assert "stub" in r.output


def test_cli_conform_accepts_profile_and_bounds(tmp_path: Path):
    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=20,
                       state_value=lambda e: 0.0 if e == 0 else 50.0)
    profile = tmp_path / "p.json"
    profile.write_text(json.dumps({
        "thresholds": {"trajectory_hausdorff_ratio": {"value": 0.99}},
    }), encoding="utf-8")
    r = CliRunner().invoke(cli, [
        "conform", str(ds), "--profile", str(profile),
        "--workspace-bounds", "0,1,0,1,0,1", "--format", "json",
        "--out", str(tmp_path / "o"),
    ])
    payload = json.loads((tmp_path / "o" / "conformance_report_gbz218.1-2026.json")
                         .read_text(encoding="utf-8"))
    th = payload["profile"]["thresholds"]["trajectory_hausdorff_ratio"]
    assert th["value"] == 0.99
    assert th["clause"] == "6.2.3.4 a"


def test_cli_audit_still_registered():
    """conform 的注册不得挤掉既有命令。"""
    names = set(cli.commands.keys())
    assert {"audit", "ui", "example", "recommend", "conform"} <= names


# ---------------------------------------------------------------------------
# 9. 报告
# ---------------------------------------------------------------------------


def test_markdown_report_discloses_sampling_size(tmp_path: Path):
    """报告必须写明实际抽样规模，不许含糊成"全量扫描"。"""
    from rda.conform import render_conformance_markdown

    ds = write_dataset(tmp_path / "ds", n_episodes=4, frames_per_ep=20)
    profile = default_profile()
    profile.sampling.hausdorff_episodes = 3
    profile.sampling.dtw_episodes = 3
    profile.sampling.dtw_max_pairs = 3
    result = run_conformance(ds, profile=profile)
    md = render_conformance_markdown(result)
    assert "3 个 episode" in md
    assert "TODO" in md or "未实现" in md
    assert "UNKNOWN" not in md


def test_conformance_node_shape(tmp_path: Path):
    from rda.conform import build_conformance_node

    ds = write_dataset(tmp_path / "ds", n_episodes=2, frames_per_ep=12)
    node = build_conformance_node(run_conformance(ds))
    json.dumps(node, default=str)  # 必须可序列化
    assert node["conformance_node_version"] == "1.0"
    for key in ("standard", "dataset", "overall_verdict", "summary", "items",
                "observations", "profile", "dataset_info", "coverage_gate"):
        assert key in node
