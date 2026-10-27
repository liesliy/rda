"""视频像素层的解码与光流底座（GB/Z 218.1-2026 附录 A / 6.2.5.2 a) 1)）。

为什么单列一个模块
------------------
``rda/metrics/visual_integrity.py:99`` 的 ``_decode_span_gray`` 会把帧缩放到
64×64 再算像素差——那个尺寸足够判冻结/抖动，**不足以判别 5 px 尺度的运动
模糊核长度**（缩放本身就抹掉了 5 px 的信息）。本模块为运动模糊与附录 A 的
动态性测量提供**原分辨率**的邻近帧解码。

依赖纪律
--------
``av``（解码）与 ``cv2``（光流）都是**延迟导入**：它们缺失时 :class:`VideoProbe`
的相应可用性标志置 False 并抛 :class:`ImportError`，由上层判
``UNDETERMINED``。模式照抄 ``engine.py:_require_pyarrow``（不适合 *1 处）。
注意措辞纪律：依赖缺失是**本地环境**问题，报告里必须写成"本地环境未安装"，
不得暗示是数据集缺陷。

不产生临时文件
--------------
全程 ``av.open`` + ``seek`` + ``decode`` 在**内存中**完成，不把 mp4 的任
何片段落盘，也不做预提取。90 GB 级数据集的完整视频副本因此不会被复制、
也不需要额外的磁盘余量。

方法依据（写在代码里，因为要进报告）
------------------------------------
* 附录 A.1：标准明文允许"卢卡斯-卡纳薇(Lucas-Kanade)方法或金字塔光流法"。
  本实现用 ``cv2.goodFeaturesToTrack`` 取点 + ``cv2.calcOpticalFlowPyrLK``
  （金字塔 Lucas-Kanade）求位移。
* ``L = α · d`` 的换算见 :func:`displacement_is_blur_upper_bound` 的注释，
  出处：David Korcak & Jiri Matas, *Video shutter angle estimation using
  optical flow and linear blur*, arXiv:2303.10247（CVWW 2024）。
* fps **只从容器元数据读**（``average_rate`` → ``base_rate`` →
  ``guessed_rate``），读不到就返回 ``None``，让上层判 ``UNDETERMINED``。
  绝不假定 30 或 60 —— 假定一个 fps 就等于把伪造的量纲塞进物理换算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AV_MISSING_MSG = (
    "av is required to decode video pixels. "
    'Install it with: pip install "robot-data-audit[video]" '
    "(or: pip install av)"
)
CV2_MISSING_MSG = (
    "opencv is required to compute optical flow. "
    'Install it with: pip install "robot-data-audit[video]" '
    "(or: pip install 'opencv-python-headless>=4.8,<5')"
)

# --- 光流参数（**工程口径**，标准只推荐了方法族，没给这些数值） -------------
DEFAULT_MAX_CORNERS = 200
DEFAULT_QUALITY_LEVEL = 0.01
DEFAULT_MIN_DISTANCE = 7.0
LK_WIN_SIZE = (15, 15)
LK_MAX_LEVEL = 3
LK_CRITERIA = (3, 30, 0.01)  # (TYPE=EPS|COUNT, maxCount, epsilon)
# 可靠性判据：单次跟踪位移超过图像对角线的这个比例即视为跟踪失败并剔除。
# 0.10 是工程口径（相邻帧之间真实世界位移通常远小于画幅对角线），必须在
# 报告中连同"剔除了多少点"一起披露，不许静默丢弃。
DIAG_REJECT_RATIO = 0.10
# 一个帧对至少要有这么多有效跟踪点才计入统计，否则整对剔除（并计数）。
MIN_VALID_POINTS_PER_PAIR = 5
# 相邻帧必须简单按呈现序取前 n_pairs+1 帧，多采这么多个是为了容纳 B 帧乱序。
PAIR_BUFFER = 8


def displacement_is_blur_upper_bound() -> str:
    """Return the stated justification for using optical flow as an upper bound.

    帧间光流位移 d 是运动模糊核长度 L 的**几何上界**：

        L = α · d ,   α = 曝光时间 / 帧周期 ≤ 1

    因为模糊是在曝光时间内物体划过的距离，而 d 是整个帧周期内的位移，两者
    方向相同、时间包含关系明确。出处：David Korcak & Jiri Matas, *Video
    shutter angle estimation using optical flow and linear blur*,
    arXiv:2303.10247（CVWW 2024）。

    由此可知两条不能越过的边界：
      * 已知 α 时，L = α·d 可与 5 px 阈值双边比较；
      * **未知 α 时**，只有单边结论成立——d ≤ 5 px **蕴含** L ≤ 5 px（可判
        PASS）；d > 5 px 什么也不蕴含（可能 L 仍 ≤ 5 px），只能 UNDETERMINED。
    """
    return (
        "光流上界法（optical-flow displacement as an **upper bound** of the "
        "motion-blur kernel length）：帧间光流位移 d 是运动模糊核长度 L 的"
        "几何上界，L = α·d，α = 曝光时间/帧周期 ≤ 1"
        "（David Korcak & Jiri Matas, Video shutter angle estimation "
        "using optical flow and linear blur, arXiv:2303.10247, CVWW 2024）。"
    )


# ---------------------------------------------------------------------------
# 数据载体
# ---------------------------------------------------------------------------


@dataclass
class FlowPairResult:
    """一个相邻帧对的 Lucas-Kanade 跟踪结果。

    Attributes:
        n_features: 参考帧上 ``goodFeaturesToTrack`` 给出的初始点数。
        n_valid: 通过全部可靠性判据的跟踪点数。
        n_rejected_status: ``calcOpticalFlowPyrLK`` 的 ``status != 1`` 的点数。
        n_rejected_displacement: status 通过但位移 ≥ 对角线的
            ``DIAG_REJECT_RATIO`` 倍、或位移非有限值的点数。
        displacements: ``n_valid`` 个有效点的位移范数（px）。
        track_errors: 与 ``displacements`` **逐点对齐**的 LK 跟踪残差
            （``calcOpticalFlowPyrLK`` 的第三个返回值，即两帧 patch 的 SSD
            残差）。它是**质量指示量，不是剔除依据**：我们没有依据给 err 划
            一条"多大算坏"的线——凭空设一个阈值就是把猜测写成判据。取而代之
            把它如实报出来，第三方才能看到"那个孤立的 max 点自己也知道跟不
            住"。OpenCV 未返回该值时为全 NaN（统计时按缺失处理）。
        diagonal_px: 图像对角线长度（px），即可靠性判据的分母。
        image_shape: ``(height, width)``。
        note: 形状不匹配等整帧退化的说明。
    """

    n_features: int = 0
    n_valid: int = 0
    n_rejected_status: int = 0
    n_rejected_displacement: int = 0
    displacements: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64))
    track_errors: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64))
    diagonal_px: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    note: str = ""

    @property
    def n_rejected(self) -> int:
        """被剔除的点数（两类相加，两者互斥且之和即为总剔除数）。"""
        return int(self.n_rejected_status + self.n_rejected_displacement)


@dataclass
class FramePairSet:
    """从一个容器里解出来的相邻帧对集合。

    Attributes:
        pairs: ``[(g0, g1), (g1, g2), ...]``，相邻帧两两成对；帧对象共享，
            不额外占内存。
        fps: 容器元数据读到的帧率；``None`` 表示读不到（上层须判 UNDETERMINED）。
        n_frames_decoded: 实际解码出来的帧数。
        n_frames_requested: 期望帧数（``n_pairs + 1``）。
        start_sec: 实际开始解码的时间戳（由抽样种子决定）。
        decode_error: 解码失败的原因（``None`` 表示成功）。不能解码是**本地
            文件或环境问题**，措辞不得指向数据集属性。
    """

    pairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    fps: Optional[float] = None
    n_frames_decoded: int = 0
    n_frames_requested: int = 0
    start_sec: Optional[float] = None
    decode_error: Optional[str] = None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


@dataclass
class CameraMotionStats:
    """一路相机（一个 video 特征）在一个抽样计划下的位移统计。"""

    feature: str
    file_count: int = 0
    files_resolved: List[str] = field(default_factory=list)
    fps: Optional[float] = None
    n_episodes_requested: int = 0
    n_episodes_used: int = 0
    n_pairs_requested: int = 0
    n_pairs_used: int = 0
    n_pairs_rejected: int = 0
    n_frames_decoded: int = 0
    n_valid_points: int = 0
    n_total_points: int = 0
    mean: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None
    # --- 稳健上界与质量指示量（v0.11.0 修订：max 是单点极值，不可单独引用）--
    p99: Optional[float] = None
    p99_9: Optional[float] = None
    n_above_p99: int = 0
    track_err_p50: Optional[float] = None
    track_err_p95: Optional[float] = None
    track_err_at_max: Optional[float] = None
    track_err_n: int = 0
    # 剔除判据的**实际门限**（px）与分辨率：光有比例看不出这条线有多松。
    frame_shape: str = ""
    diagonal_px: Optional[float] = None
    reject_limit_px: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    decode_errors: List[str] = field(default_factory=list)
    # 逐 episode 明细：第三方复核要能定位到"第几个 episode 的多少帧"，
    # 只有一路相机一个聚合值的话，抽到哪几个 episode 就成了黑盒。
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (numbers already plain floats)."""
        return {
            "feature": self.feature,
            "file_count": self.file_count,
            "files_resolved": list(self.files_resolved),
            "fps": self.fps,
            "n_episodes_requested": self.n_episodes_requested,
            "n_episodes_used": self.n_episodes_used,
            "n_pairs_requested": self.n_pairs_requested,
            "n_pairs_used": self.n_pairs_used,
            "n_pairs_rejected": self.n_pairs_rejected,
            "n_frames_decoded": self.n_frames_decoded,
            "n_valid_points": self.n_valid_points,
            "n_total_points": self.n_total_points,
            "displacement_px": {
                "mean": self.mean, "p50": self.p50,
                "p95": self.p95, "max": self.max,
                "p99": self.p99, "p99_9": self.p99_9,
                "n_above_p99": self.n_above_p99,
            },
            # 质量指示量，不是剔除依据：没有依据给 err 划阈值，只披露。
            "track_err": {
                "n": self.track_err_n,
                "p50": self.track_err_p50,
                "p95": self.track_err_p95,
                "at_max": self.track_err_at_max,
                "meaning": (
                    "calcOpticalFlowPyrLK 返回的每点 SSD 残差；数值越大说明该点"
                    "的局部 patch 在两帧间越不匹配（可能跟丢）。本工具不据此"
                    "剔除任何点，仅供第三方判断尾部点是否可信。"
                ),
            },
            "frame_shape": self.frame_shape,
            "diagonal_px": self.diagonal_px,
            "reject_limit_px": self.reject_limit_px,
            "notes": list(self.notes),
            "decode_errors": list(self.decode_errors),
            "episodes": list(self.episodes),
        }


# ---------------------------------------------------------------------------
# 延迟依赖
# ---------------------------------------------------------------------------


def require_av():
    """Import and return PyAV, raising ImportError with install hint."""
    try:
        import av  # noqa: PLC0415
    except ImportError:
        raise ImportError(AV_MISSING_MSG) from None
    return av


def require_cv2():
    """Import and return cv2, raising ImportError with install hint."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        raise ImportError(CV2_MISSING_MSG) from None
    return cv2


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------


def stream_fps(stream) -> Optional[float]:
    """Read the frame rate from container metadata only.

    优先级 ``average_rate`` → ``base_rate`` → ``guessed_rate``。三者都拿不到
    就返回 ``None`` —— 调用方必须据此走 UNDETERMINED，不许回落成 30/60。
    """
    for attr in ("average_rate", "base_rate", "guessed_rate"):
        rate = getattr(stream, attr, None)
        if rate is None:
            continue
        try:
            value = float(rate)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return None


def _gray_frame(frame, blur_ksize: int = 0) -> np.ndarray:
    """Convert one decoded frame to a 2-D uint8 grayscale array.

    保持**原分辨率**——降采样会把 5 px 尺度的核长度信息抹掉。``blur_ksize``
    为 0 时不做任何平滑；为正且为偶数时向上取到最近的奇数（高斯核必须为奇数）。
    """
    arr = frame.reformat(format="gray").to_ndarray()
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.ascontiguousarray(arr)
    if blur_ksize and blur_ksize > 1:
        cv2 = require_cv2()
        k = int(blur_ksize) | 1
        if k < arr.shape[0] and k < arr.shape[1]:
            arr = cv2.GaussianBlur(arr, (k, k), 0)
    return arr


def _pick_start(from_ts: float, to_ts: float, fps: Optional[float],
                n_frames: int, rng: np.random.Generator) -> float:
    """Pick a start timestamp for the consecutive-frame window.

    窗口内随机取起点：整段 episode 只有一个 GOP 的话固定起点会重复抽样，而
    固定（比如总取中段）又会漏掉首尾运动。``fps`` 未知时无法换算"需要多长
    时间"，就退回 ``from_ts`` 并在 note 里说明。
    """
    from_ts = float(max(from_ts or 0.0, 0.0))
    to_ts = float(to_ts or 0.0)
    if not fps or fps <= 0 or to_ts <= from_ts:
        return from_ts
    need_sec = n_frames / float(fps)
    slack = (to_ts - from_ts) - need_sec
    if slack <= 0:
        return from_ts
    return from_ts + float(rng.uniform(0.0, slack))


class VideoProbe:
    """Video pixel reader with graceful degradation on missing deps.

    用法与 ``DatasetView._require_pyarrow`` 同款：构造时假定可用，真正要用
    时才导入；导入失败把 ``*_available`` 置 False 并抛出，让调用方把它变成
    一条 UNDETERMINED 而不是崩溃。
    """

    def __init__(self) -> None:
        self.av_available: bool = True
        self.cv2_available: bool = True

    def _require_av(self):
        """Import av; set ``av_available = False`` and raise when missing."""
        try:
            return require_av()
        except ImportError:
            self.av_available = False
            raise

    def _require_cv2(self):
        """Import cv2; set ``cv2_available = False`` and raise when missing."""
        try:
            return require_cv2()
        except ImportError:
            self.cv2_available = False
            raise

    def probe_fps(self, path: Path) -> Optional[float]:
        """Read the container frame rate of ``path``; ``None`` when unreadable.

        只在拿不到容器元数据（或文件打不开）时为 None。这不是 30/60 的猜测值。
        """
        av = self._require_av()
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    return None
                return stream_fps(container.streams.video[0])
        except Exception:  # noqa: BLE001 — 坏文件要降级成 None，不能炸
            return None

    def read_consecutive_pairs(
        self,
        path: Path,
        from_ts: float,
        to_ts: float,
        *,
        n_pairs: int,
        rng: Optional[np.random.Generator] = None,
        blur_ksize: int = 0,
    ) -> FramePairSet:
        """Decode ``n_pairs`` adjacent frame pairs from ``[from_ts, to_ts)``.

        全程内存内 seek + decode：**不落盘任何临时文件**。为避免把一个
        几万帧的文件整段读进内存，只解 ``n_pairs + 1`` 个**相邻**帧。

        Args:
            path: mp4 路径。
            from_ts / to_ts: episode 在该文件内的时间窗口（秒）。
            n_pairs: 需要的相邻帧对数。
            rng: 抽样随机源（默认 ``default_rng(0)``），保证起点可复算。
            blur_ksize: >1 时做一次轻度高斯平滑（光流前处理），不改变分辨率。

        Returns:
            :class:`FramePairSet`。解不出来时不抛异常，把原因写进
            ``decode_error``——因为"文件不可解码"是判别的前提条件，调用方需要
            它来写 note，而不是一个 traceback。
        """
        av = self._require_av()
        need = int(n_pairs) + 1
        rng = rng if rng is not None else np.random.default_rng(0)
        empty = FramePairSet(n_frames_requested=need)
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    empty.decode_error = "容器内没有视频轨"
                    return empty
                stream = container.streams.video[0]
                fps = stream_fps(stream)
                empty.fps = fps
                if fps is None:
                    empty.decode_error = (
                        "容器元数据未提供帧率（average_rate / base_rate / "
                        "guessed_rate 均不可用），无法定位帧窗口"
                    )
                    # 帧率不可知时窗口失去意义：本实现不假设帧率。
                    return empty
                start = _pick_start(from_ts, to_ts, fps, need, rng)
                empty.start_sec = start
                tb = float(stream.time_base)
                target = int(start / tb) if tb > 0 else int(start)
                self._seek(container, stream, target)
                tol = 0.5 / fps
                # seek 之后 PTS 未必单调递增（有 B 帧时解码序 ≠ 呈现序），所以先
                # 按 pts 收集再排序取前 need 帧，而不是"读到哪个算哪个"。
                # 两个预算都要有上限：跳过的帧太多说明 seek 落点远得离谱（长
                # GOP），采集太多会把整段 episode 读进内存。
                skip_budget = max(int(5 * fps), 5 * need)
                harvested: Dict[float, np.ndarray] = {}
                skipped = 0
                for frame in container.decode(stream):
                    if frame.pts is None:
                        skipped += 1
                        if skipped > skip_budget:
                            break
                        continue
                    pts = float(frame.pts * tb)
                    if pts < start - tol:
                        skipped += 1
                        if skipped > skip_budget:
                            break
                        continue
                    harvested[pts] = _gray_frame(frame, blur_ksize)
                    if len(harvested) >= need + PAIR_BUFFER:
                        break
                frames = [harvested[k] for k in sorted(harvested)[:need]]
                pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
                return FramePairSet(
                    pairs=pairs, fps=fps, n_frames_decoded=len(frames),
                    n_frames_requested=need, start_sec=start,
                )
        except ImportError:
            # 依赖缺失不是"文件不可解码"，不能被降级成一条普通的解码错误——
            # 它必须原样往上抛，让上层写成"本地环境未安装，非数据集缺陷"。
            raise
        except Exception as e:  # noqa: BLE001 — 坏文件/坏编码都必须降级
            empty.decode_error = f"{type(e).__name__}: {e}"
            return empty

    @staticmethod
    def _seek(container, stream, target: int) -> None:
        """Seek ``stream`` to ``target`` (stream time base units), best effort."""
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
            return
        except (OSError, ValueError):
            pass
        try:
            container.seek(target, stream=stream)
        except (OSError, ValueError):
            pass

    def track_displacements(
        self, pair: Tuple[np.ndarray, np.ndarray], **kwargs
    ) -> FlowPairResult:
        """Track one adjacent frame pair (thin wrapper over :func:`lk_displacement`)."""
        self._require_cv2()
        return lk_displacement(pair[0], pair[1], **kwargs)


def lk_displacement(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    *,
    max_corners: int = DEFAULT_MAX_CORNERS,
    quality_level: float = DEFAULT_QUALITY_LEVEL,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    diag_ratio: float = DIAG_REJECT_RATIO,
) -> FlowPairResult:
    """Lucas-Kanade pyramidal displacement between two grayscale frames.

    附录 A.1 明文允许的方法族（"卢卡斯-卡纳薇(Lucas-Kanade)方法或金字塔光流
    法"）。这里用 Shi-Tomasi 角点 + ``calcOpticalFlowPyrLK``。

    可靠性判据（**先记 status，再记位移幅值**，两者互斥且之和 = 总剔除数）：
      1. ``calcOpticalFlowPyrLK`` 返回的 ``status == 1``；
      2. 位移范数有限且 < ``diag_ratio`` × 图像对角线。

    必须清楚这条判据的**松紧程度**：它是一条"跟踪彻底跑飞"的兜底线，不是
    "跟得准"的门槛——640×480 的门限是 800 × 0.10 = 80 px，一个跑了 75 px 的
    单点野值会安然落在保留区。残量较小的野值不靠这条线拦，而是靠
    :func:`summarize_displacements` 给出的 p99.9 / max 与逐点残差**披露**
    出来：**判定用 p95，复核看 p99.9，不建议读 max**。

    Returns:
        :class:`FlowPairResult`。剔除数一律回填到结果里，供报告披露。
    """
    cv2 = require_cv2()
    h, w = int(gray_a.shape[0]), int(gray_a.shape[1])
    diagonal = float(np.hypot(h, w))
    base = FlowPairResult(diagonal_px=diagonal, image_shape=(h, w))
    if gray_a.shape != gray_b.shape:
        base.note = "相邻帧分辨率不一致，帧对整体剔除"
        return base
    features = cv2.goodFeaturesToTrack(
        gray_a, maxCorners=int(max_corners), qualityLevel=float(quality_level),
        minDistance=float(min_distance), blockSize=7,
    )
    if features is None or len(features) == 0:
        base.note = "参考帧无可跟踪角点，帧对整体剔除"
        return base
    pts0 = features.reshape(-1, 2).astype(np.float32)
    base.n_features = int(pts0.shape[0])
    pts1, status, track_err = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, features, None,
        winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )
    if pts1 is None or status is None:
        base.note = "calcOpticalFlowPyrLK 未返回结果，帧对整体剔除"
        return base
    disp = np.linalg.norm(pts1.reshape(-1, 2) - pts0, axis=1)
    ok_status = status.reshape(-1) == 1
    within = np.isfinite(disp) & (disp < diag_ratio * diagonal)
    valid = ok_status & within
    base.n_valid = int(valid.sum())
    base.n_rejected_status = int((~ok_status).sum())
    base.n_rejected_displacement = int((ok_status & ~within).sum())
    base.displacements = np.asarray(disp[valid], dtype=np.float64)
    # 残差与原位移同序，这里按 valid 掩码取同一批点。取不到（后端没返回）时
    # 填 NaN——"拿不到"必须表现为缺失，不许用 0 冒充"跟得完美"。
    if track_err is None:
        errs = np.full(disp.shape, np.nan, dtype=np.float64)
    else:
        flat = np.asarray(track_err, dtype=np.float64).reshape(-1)
        errs = flat if flat.shape[0] == disp.shape[0] else \
            np.full(disp.shape, np.nan, dtype=np.float64)
    base.track_errors = errs[valid]
    return base


def _robust_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Statistics of a non-empty finite sample (helper, not part of the API)."""
    p99 = float(np.percentile(arr, 99, method="linear"))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50, method="linear")),
        "p95": float(np.percentile(arr, 95, method="linear")),
        "p99": p99,
        # 稳健上界：p99.9 对应的点数在 200 点/帧对的默认规模下约为 0.2 个点，
        # 因此它几乎不受单点野值影响，比 max 更能代表"尾部到哪为止"。
        "p99_9": float(np.percentile(arr, 99.9, method="linear")),
        "n_above_p99": int(np.count_nonzero(arr > p99)),
        "max": float(arr.max()),
    }


_EMPTY_STATS: Dict[str, Any] = {
    "n": 0, "mean": None, "p50": None, "p95": None, "p99": None,
    "p99_9": None, "n_above_p99": 0, "max": None,
    "track_err_p50": None, "track_err_p95": None, "track_err_at_max": None,
    "track_err_n": 0,
}


def summarize_displacements(
    values: Optional[Sequence[float] | np.ndarray],
    track_errors: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, Any]:
    """Aggregate displacement norms into count + mean/p50/p95/p99.9/max.

    为什么要有 p99.9：``max`` 是**全部有效点的单一最大值**，一个跟丢了的点
    就能把它抬到整帧真实运动的好几倍（实测：640×480 下 lerobot_folding 报出
    max=75.04 px，而同帧其余 199 个点最大仅 10.02 px、LK 自己给该点的残差
    err=66.74）。把这样的数裸着放进报告，第三方复核时会读成"存在 75 px 的帧
    间位移"，那是错的。所以：
      * **判定仍用 p95**（保守口径，与 v0.11.0 一致，未变）；
      * **p99.9 作为稳健上界**一并给出；
      * **max 保留但必须标注为单点极值**（标注在报告那份 markdown 里，见
        ``rda/conform/report.py`` 的脚注），不允许单独引用。

    Args:
        values: 位移范数样本（px）。
        track_errors: 与 ``values`` **逐点对齐**的 LK 跟踪残差；给出时额外
            返回 ``track_err_p50`` / ``track_err_p95`` /
            ``track_err_at_max``（产生 max 的那个点的残差——复核单个野值时
            最直接的一条证据）与 ``track_err_n``（残差可用的点数）。

    百分位用 **线性插值**（``numpy.percentile`` 默认 ``method="linear"``），
    报告里要把插值方法写出来，否则第三方无法复算同一个数。空样本返回全 None
    且 ``n == 0``——调用方据此判 UNDETERMINED，不许当成 0 去凑阈值。
    """
    out = dict(_EMPTY_STATS)
    if values is None:
        return out
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size and track_errors is not None:
        terr = np.asarray(track_errors, dtype=np.float64).reshape(-1)
        if terr.shape[0] == arr.shape[0]:
            ok = np.isfinite(arr) & np.isfinite(terr)
            aligned_err = terr[ok]
            aligned_disp = arr[ok]
        else:
            aligned_err = None  # 长度对不上 → 残差不可用，按缺失处理
            ok = np.isfinite(arr)
            aligned_disp = arr[ok]
    else:
        aligned_err = None
        ok = np.isfinite(arr)
        aligned_disp = arr[ok]
    arr = aligned_disp
    if arr.size == 0:
        return out
    out.update(_robust_stats(arr))
    if aligned_err is not None and aligned_err.size:
        out["track_err_n"] = int(aligned_err.size)
        out["track_err_p50"] = float(np.percentile(aligned_err, 50,
                                                   method="linear"))
        out["track_err_p95"] = float(np.percentile(aligned_err, 95,
                                                   method="linear"))
        idx = int(np.argmax(aligned_disp))
        out["track_err_at_max"] = float(aligned_err[idx])
    return out
