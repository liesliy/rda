"""conform 编排层：数据集视图 + 判定流水线。

本模块做两件事：

1. :class:`DatasetView` —— 把 LeRobot v3.0 数据集目录抽象成 conform 需要的
   "数据集级"视图。现有 :class:`~rda.io.schema.EpisodeData` 是逐 episode 的，
   而这里是数据集级的（列级缺失率、跨 episode 的时间轴、声明但不存在的文件），
   两者互为补充，不重复。
2. :func:`run_conformance` —— 加载 → 跑适用判定项 → 汇总 → 返回 conformance dict。

依赖纪律
--------
pyarrow 是**延迟导入**的。P0 不引入重依赖，pyarrow 只用于读 parquet 元数据
与列投影；它缺失时相关判定项走 UNDETERMINED 并在 note 里写明缺什么，
而不是崩溃或静默 PASS。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rda.conform.profile import ConformProfile, default_profile
from rda.conform.verdict import ConformanceItem, Verdict, overall_verdict, summarize

_MISSING_DEPS_MSG = (
    "pyarrow is required to read LeRobot parquet datasets. "
    'Install it with: pip install "robot-data-audit[parquet]" '
    "(or: pip install pyarrow)"
)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class ConformOptions:
    """判定需要的外部声明。标准不给的值必须由数据提供方声明，不能猜。"""

    # 可行解空间的笛卡尔边界 (xmin,xmax,ymin,ymax,zmin,zmax)，6 个数
    workspace_bounds: Optional[Sequence[float]] = None
    # 或直接声明"可行解空间最大距离"（6.2.3.4 a 的分母）
    workspace_max_distance: Optional[float] = None
    # coverage 门禁。P0 只留桩，见 run_conformance 中的 TODO。
    strict_coverage: bool = False

    def resolved_max_distance(self) -> Optional[float]:
        """Return the declared 'feasible-space max distance', or None.

        只接受两种输入：显式声明的距离，或 6 个数的笛卡尔边界（取其空间
        对角线）。两者都没给就返回 None —— 调用方必须判 UNDETERMINED，
        不得拿实测 Hausdorff 值反推。
        """
        if self.workspace_max_distance is not None:
            return float(self.workspace_max_distance)
        if self.workspace_bounds is not None:
            b = [float(v) for v in self.workspace_bounds]
            if len(b) == 6:
                span = [abs(b[1] - b[0]), abs(b[3] - b[2]), abs(b[5] - b[4])]
                return float(np.linalg.norm(span))
        return None


# ---------------------------------------------------------------------------
# DatasetView
# ---------------------------------------------------------------------------


class DatasetView:
    """LeRobot 数据集目录的只读视图（数据集级，非逐 episode）。

    所有读取都是惰性的：只有真正被某个判定项用到的部分才会去读盘，
    结果缓存在实例上。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._info: Optional[dict] = None
        self._episodes_meta: Optional[List[dict]] = None
        self._tasks: Optional[Dict[int, str]] = None
        self._null_counts: Optional[Dict[str, Tuple[int, int]]] = None
        self._columns: Optional[List[str]] = None
        self._ts_by_ep: Optional[Dict[int, np.ndarray]] = None
        self._task_indices: Optional[np.ndarray] = None
        self._arrays_cache: Dict[Tuple[str, ...], Dict[int, Dict[str, np.ndarray]]] = {}
        self.pyarrow_available: bool = True

    # --- 基础元信息 --------------------------------------------------------

    @property
    def info(self) -> dict:
        """``meta/info.json`` 的内容。文件缺失时返回空 dict。"""
        if self._info is None:
            p = self.root / "meta" / "info.json"
            if p.exists():
                try:
                    self._info = json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._info = {}
            else:
                self._info = {}
        return self._info

    @property
    def fps(self) -> Optional[float]:
        """Declared frames per second, or ``None`` when not declared."""
        v = self.info.get("fps")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def num_episodes(self) -> Optional[int]:
        """Declared episode count (``meta/info.json`` → ``total_episodes``)."""
        v = self.info.get("total_episodes")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def feature_specs(self) -> Dict[str, dict]:
        """``meta/info.json`` → ``features``，键为特征名。"""
        feats = self.info.get("features")
        if isinstance(feats, dict):
            return {k: (v if isinstance(v, dict) else {}) for k, v in feats.items()}
        return {}

    def feature_names(self) -> List[str]:
        """All feature names declared in ``meta/info.json``."""
        return list(self.feature_specs().keys())

    @property
    def data_files(self) -> List[Path]:
        """All ``data/chunk-*/file-*.parquet`` paths, sorted."""
        return sorted((self.root / "data").glob("chunk-*/file-*.parquet"))

    @property
    def videos_dir(self) -> Path:
        """Path of the ``videos/`` directory (may not exist)."""
        return self.root / "videos"

    def video_payload_files(self, feature: str) -> List[Path]:
        """Files under ``videos/<feature>/**`` (LeRobot v3 video layout).

        只判断**存在性**，不校验可解码性——本版本不引入 opencv/av，
        一个文件在不在是可以确定的，能不能解码不是。
        """
        base = self.videos_dir / feature
        if not base.exists():
            return []
        try:
            return sorted(p for p in base.rglob("*") if p.is_file())
        except OSError:
            return []

    def videos_dir_file_count(self) -> int:
        """Total files under ``videos/``; 0 when the directory is absent.

        用于区分「压根没下载视频」与「下载了但某个视角缺文件」——两者都是
        本地副本不完整，都不许记成数据集的字段缺失。
        """
        if not self.videos_dir.exists():
            return 0
        try:
            return sum(1 for p in self.videos_dir.rglob("*") if p.is_file())
        except OSError:
            return 0

    def _require_pyarrow(self):
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError:
            self.pyarrow_available = False
            raise ImportError(_MISSING_DEPS_MSG) from None
        return pq

    # --- 列结构 / 缺失率 ---------------------------------------------------

    @property
    def parquet_columns(self) -> List[str]:
        """Column names actually present in the data parquet files.

        优先用 row group statistics 里的 ``null_count``（几乎零成本）；
        statistics 不可用时退回物化读列。
        """
        if self._columns is not None:
            return self._columns
        pq = self._require_pyarrow()
        cols: List[str] = []
        for f in self.data_files:
            try:
                pf = pq.ParquetFile(f)
            except (OSError, ValueError):
                continue
            cols = list(pf.schema_arrow.names)
            break
        self._columns = cols
        return cols

    def column_null_counts(self) -> Dict[str, Tuple[int, int]]:
        """Per-column ``(null_count, total_rows)`` over the whole dataset.

        这是 6.2.2.1 a/b 缺失率的分子分母来源：全量、不抽样。
        """
        if self._null_counts is not None:
            return self._null_counts
        pq = self._require_pyarrow()
        totals: Dict[str, List[int]] = {}
        for f in self.data_files:
            try:
                pf = pq.ParquetFile(f)
            except (OSError, ValueError):
                continue
            md = pf.metadata
            names = list(pf.schema_arrow.names)
            for i, name in enumerate(names):
                null_count: Optional[int] = 0
                for r in range(md.num_row_groups):
                    st = md.row_group(r).column(i).statistics
                    if st is None or st.null_count is None:
                        null_count = None
                        break
                    null_count += int(st.null_count)
                if null_count is None:
                    # statistics 不可用 → 物化这一列来数
                    tbl = pf.read(columns=[name])
                    null_count = int(tbl.column(name).null_count)
                acc = totals.setdefault(name, [0, 0])
                acc[0] += int(null_count)
                acc[1] += int(md.num_rows)
        self._null_counts = {k: (v[0], v[1]) for k, v in totals.items()}
        return self._null_counts

    # --- 时间戳 ------------------------------------------------------------

    def timestamps_by_episode(self) -> Dict[int, np.ndarray]:
        """Per-episode timestamp arrays (float64), keyed by episode index."""
        if self._ts_by_ep is not None:
            return self._ts_by_ep
        pq = self._require_pyarrow()
        ep_all: List[np.ndarray] = []
        ts_all: List[np.ndarray] = []
        for f in self.data_files:
            if "timestamp" not in self.parquet_columns:
                break
            tbl = pq.read_table(f, columns=["episode_index", "timestamp"])
            ep_all.append(tbl.column("episode_index").to_numpy(zero_copy_only=False))
            ts_all.append(
                tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.float64)
            )
        if not ep_all:
            self._ts_by_ep = {}
            return self._ts_by_ep
        ep = np.concatenate(ep_all)
        ts = np.concatenate(ts_all)
        order = np.argsort(ep, kind="stable")
        ep, ts = ep[order], ts[order]
        out: Dict[int, np.ndarray] = {}
        for ue in np.unique(ep):
            seg = ts[ep == ue]
            out[int(ue)] = np.sort(seg)
        self._ts_by_ep = out
        return out

    def episode_ids(self) -> List[int]:
        """All episode indices present in the data parquet files."""
        return sorted(self.timestamps_by_episode().keys())

    def task_indices(self) -> Optional[np.ndarray]:
        """Per-frame ``task_index`` array, or ``None`` when absent."""
        if self._task_indices is not None:
            return self._task_indices
        if "task_index" not in self.parquet_columns:
            return None
        pq = self._require_pyarrow()
        parts = []
        for f in self.data_files:
            tbl = pq.read_table(f, columns=["task_index"])
            parts.append(tbl.column("task_index").to_numpy(zero_copy_only=False))
        self._task_indices = np.concatenate(parts) if parts else None
        return self._task_indices

    # --- episode 级元信息 --------------------------------------------------

    @property
    def episodes_meta(self) -> List[dict]:
        """Rows of ``meta/episodes/**.parquet`` as plain dicts (one per episode)."""
        if self._episodes_meta is not None:
            return self._episodes_meta
        pq = self._require_pyarrow()
        rows: List[dict] = []
        for p in sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            try:
                tbl = pq.read_table(p)
            except (OSError, ValueError):
                continue
            rows.extend(tbl.to_pylist())
        self._episodes_meta = rows
        return rows

    def video_windows(self) -> Dict[str, Dict[int, Tuple[float, float, Optional[int]]]]:
        """Per-camera video time windows from the episode metadata.

        Returns:
            ``{camera_key: {episode_index: (from_ts, to_ts, file_index)}}``
            ``camera_key`` 取自列名 ``videos/<video_key>/from_timestamp`` 的
            中间段。<video_key> 不存在时返回空 dict（不是 UNDETERMINED 的
            判定依据——调用方据此决定）。
        """
        rows = self.episodes_meta
        if not rows:
            return {}
        out: Dict[str, Dict[int, Tuple[float, float, Optional[int]]]] = {}
        pat = re.compile(r"^videos/(?P<key>.+)/(from_timestamp|to_timestamp|file_index)$")
        for row in rows:
            try:
                ep_idx = int(row.get("episode_index"))
            except (TypeError, ValueError):
                continue
            for col, val in row.items():
                m = pat.match(str(col))
                if not m:
                    continue
                key = m.group("key")
                kind = m.group(2)
                slot = out.setdefault(key, {})
                cur = slot.get(ep_idx, (None, None, None))
                if kind == "from_timestamp":
                    slot[ep_idx] = (_f(val), cur[1], cur[2])
                elif kind == "to_timestamp":
                    slot[ep_idx] = (cur[0], _f(val), cur[2])
                else:
                    slot[ep_idx] = (cur[0], cur[1], _i(val))
        return out

    # --- 抽样数组 ----------------------------------------------------------

    def load_episode_arrays(
        self, columns: Sequence[str], episode_ids: Sequence[int]
    ) -> Dict[int, Dict[str, np.ndarray]]:  # noqa: D401
        """Load ``columns`` for the selected episodes only.

        Args:
            columns: 需要读取的列名（不含 ``episode_index``，会自动带上）。
            episode_ids: 目标 episode 下标列表。

        Returns:
            ``{episode_index: {column: ndarray}}``，数组按 timestamp 升序排列。
        """
        key = (tuple(sorted(columns)), tuple(sorted(episode_ids)))
        if key in self._arrays_cache:
            return self._arrays_cache[key]
        pq = self._require_pyarrow()
        available = set(self.parquet_columns)
        cols = [c for c in columns if c in available]
        if not cols:
            self._arrays_cache[key] = {}
            return {}
        wanted = set(int(e) for e in episode_ids)
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for f in self.data_files:
            tbl = pq.read_table(f, columns=["episode_index"] + cols)
            ep = tbl.column("episode_index").to_numpy(zero_copy_only=False)
            mask = np.isin(ep, list(wanted))
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            ee = ep[idx]
            mats: Dict[str, np.ndarray] = {}
            for c in cols:
                mats[c] = _column_to_matrix(tbl.column(c))[idx]
            order_col = mats.get("timestamp")
            for ue in np.unique(ee):
                # m 是"idx 子集"内的位置，不是全表位置——mats 里的数组
                # 已经按 idx 取过子集，两者必须同域。
                m = np.flatnonzero(ee == ue)
                if order_col is not None and m.size > 1:
                    m = m[np.argsort(order_col[m], kind="stable")]
                out[int(ue)] = {c: mats[c][m] for c in cols}
        self._arrays_cache[key] = out
        return out

    # --- 说明文档 / 标识 ---------------------------------------------------

    def dataset_card_text(self) -> Optional[str]:
        """Concatenated text of root-level README / dataset-card markdown files.

        只取根目录的 ``README*.md`` / ``dataset_card*.md`` —— 更深层的文件
        不属于"数据集级说明"。
        """
        parts: List[str] = []
        for name in sorted(p.name for p in self.root.glob("*.md")):
            low = name.lower()
            if low.startswith("readme") or low.startswith("dataset_card") or low.startswith("card"):
                try:
                    parts.append((self.root / name).read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
        return "\n".join(parts) if parts else None

    def root_file_names(self) -> List[str]:
        """Names of files directly under the dataset root."""
        try:
            return sorted(p.name for p in self.root.iterdir() if p.is_file())
        except OSError:
            return []


def _column_to_matrix(col) -> np.ndarray:
    """把一个 pyarrow 列转成 numpy 数组：list 列 → ``(n, d)``，标量列 → ``(n,)``。

    含 null 的 list 列必须走逐行补齐路径：``np.vstack`` 遇到 ``None`` 会直接抛
    ``ValueError``，而含缺失值的向量列在真实数据集里并不罕见。null 位置填 NaN，
    由下游算法自行决定 NaN 语义——这里不替它"当成 0"，那是编数据。
    """
    import pyarrow as pa  # noqa: PLC0415 — 与读 parquet 同域的延迟依赖

    t = col.type
    if pa.types.is_list(t) or pa.types.is_fixed_size_list(t) or pa.types.is_large_list(t):
        if col.null_count == 0:
            try:
                return np.asarray(np.vstack(col.to_numpy(zero_copy_only=False)),
                                  dtype=np.float64)
            except ValueError:
                pass  # 落到逐行补齐路径
        values = col.to_pylist()
        width = 1
        for v in values:
            if v is not None:
                width = max(width, len(v))
        out = np.full((len(values), width), np.nan, dtype=np.float64)
        for i, v in enumerate(values):
            if v is not None:
                out[i, : len(v)] = v
        return out
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class ConformanceResult:
    """Outcome of one conform run."""

    standard: str
    dataset: str
    items: List[ConformanceItem]
    summary: Dict[str, Any]
    observations: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    coverage_gate: Dict[str, Any] = field(default_factory=dict)
    overall: str = Verdict.UNDETERMINED.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the ``conformance`` node of the JSON report."""
        return {
            "standard": self.standard,
            "dataset": self.dataset,
            "overall_verdict": self.overall,
            "summary": self.summary,
            "items": [it.to_dict() for it in self.items],
            "observations": self.observations,
            "profile": self.profile,
            "dataset_info": self.dataset_info,
            "coverage_gate": self.coverage_gate,
        }


def run_conformance(
    dataset_path: str | Path,
    profile: Optional[ConformProfile] = None,
    options: Optional[ConformOptions] = None,
) -> ConformanceResult:
    """Run every applicable conformance check against a dataset.

    Args:
        dataset_path: LeRobot 数据集目录。
        profile: 阈值配置；``None`` 用内置默认（GB/Z 218.1-2026 原文）。
        options: 外部声明（可行解空间等）与门禁开关。

    Returns:
        :class:`ConformanceResult`。
    """
    from rda.conform import checks as _checks  # 局部导入，避免模块级循环

    profile = profile or default_profile()
    options = options or ConformOptions()
    view = DatasetView(dataset_path)

    items: List[ConformanceItem] = []
    for check in _checks.ALL_CHECKS:
        try:
            items.append(check(view, profile, options))
        except ImportError as e:
            # 依赖缺失：不能静默跳过，也不能记成 PASS —— 记成 UNDETERMINED
            # 并把缺的依赖写进 note。
            items.append(
                ConformanceItem(
                    clause="—",
                    metric=check.__name__,
                    value=None,
                    threshold=None,
                    binding="—",
                    verdict=Verdict.UNDETERMINED,
                    method="未执行：依赖缺失。",
                    note=f"缺失输入：{e}",
                )
            )
        except Exception as e:  # noqa: BLE001 — 单项失败不得拖垮整份报告
            items.append(
                ConformanceItem(
                    clause="—",
                    metric=check.__name__,
                    value=None,
                    threshold=None,
                    binding="—",
                    verdict=Verdict.UNDETERMINED,
                    method="未执行：判定项内部异常。",
                    note=f"判定项 {check.__name__} 执行失败（{type(e).__name__}: {e}），"
                         f"该项结论未定，不得视为通过。",
                )
            )

    summary = summarize(items)
    observations = _checks.compute_observations(view, profile, options)

    # --- coverage 门禁（P0 留桩）-----------------------------------------
    # TODO(v0.10.x): 定义"可裁定覆盖率"的达标阈值与门禁语义（按条款重要性
    # 加权？还是按应/宜分层？），并与 --strict-coverage 联动触发退出码 3。
    # 本轮只把覆盖率算出来写进报告，不做门禁判定——硬编码一个阈值然后说
    # "已实现"比留空更糟。
    coverage_gate = {
        "enabled": False,
        "requested": bool(options.strict_coverage),
        "coverage": summary["coverage"],
        "status": "TODO_NOT_IMPLEMENTED",
        "note": (
            "coverage 门禁本轮未实现：达标阈值与门禁语义尚未定义，"
            "--strict-coverage 当前只做记录，不触发退出码 3。"
        ),
    }

    dataset_info = {
        "path": str(Path(dataset_path)),
        "fps": view.fps,
        "num_episodes": view.num_episodes,
        "declared_features": view.feature_names(),
        "parquet_columns": view.parquet_columns,
        "videos_dir_present": view.videos_dir.exists(),
    }

    return ConformanceResult(
        standard=profile.standard,
        dataset=str(Path(dataset_path)),
        items=items,
        summary=summary,
        observations=observations,
        profile=_profile_dict(profile),
        dataset_info=dataset_info,
        coverage_gate=coverage_gate,
        overall=overall_verdict(items).value,
    )


def _profile_dict(profile: ConformProfile) -> Dict[str, Any]:
    from rda.conform.profile import profile_to_dict  # 局部导入避免循环

    return profile_to_dict(profile)
