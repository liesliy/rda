"""conform 的阈值配置：pydantic 模型 + 加载 + 内置默认 profile。

为什么用 JSON 而不是 YAML
--------------------------
P0 的依赖约束是"核心依赖仅 click / pydantic / numpy / requests"。YAML 需要
额外装 pyyaml，JSON 不需要。既然 profile 是机器读写的，JSON 完全够用，
就没有理由为它新增一个依赖。

覆盖语义
--------
``load_profile(path)`` 以内置默认 profile 为底，用文件中的字段**逐项覆盖**
（不是整体替换）。这样一份 profile 可以只改一个阈值而不必抄全表，也避免
了"用户 profile 缺字段 → 静默退回默认值"这类说不清的行为。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rda.conform.standards.gbz218_1 import (
    BINDING_ADVISORY,
    BINDING_INFORMATIONAL,
    BINDING_MANDATORY,
    CORE_FIELDS,
    FRAME_LOSS_FACTOR,
    all_thresholds,
)

PROFILE_FORMAT_VERSION = "1.0"

_BINDING_CHOICES = (BINDING_MANDATORY, BINDING_ADVISORY, BINDING_INFORMATIONAL)


class ThresholdSpec(BaseModel):
    """单条阈值。字段与 :class:`~rda.conform.standards.gbz218_1.Threshold` 一致。

    ``clause`` / ``binding`` 允许为空：用户 profile 只想改数值时不必重复
    条款号，加载时会继承内置值。
    """

    clause: Optional[str] = None
    binding: Optional[str] = None
    value: Any = None
    comparator: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    note: Optional[str] = None


class SamplingSpec(BaseModel):
    """重计算项（Hausdorff / DTW）的抽样与算法参数。

    这些参数**不影响标准符合性结论的口径**，只影响"在多少个样本上算"。
    默认值刻意偏小，因为 DTW 是 O(n·m)；报告里会把实际抽样规模原样写出来，
    不得声称"全量扫描"。
    """

    hausdorff_episodes: int = Field(
        default=100, ge=2,
        description="Hausdorff 均值抽样的 episode 数（默认 100）。",
    )
    hausdorff_resample_T: int = Field(
        default=64, ge=2,
        description="Hausdorff 计算前把每条轨迹重采样到的点数（默认 64）。",
    )
    dtw_episodes: int = Field(
        default=100, ge=2,
        description=(
            "DTW 抽样的 episode 数（默认 100，与 hausdorff_episodes 一致）。"
            "默认口径与 Hausdorff 对齐，两个附录 B 观测项才可比。"
        ),
    )
    dtw_max_pairs: int = Field(
        default=4950, ge=1,
        description=(
            "DTW 最多计算的 episode 配对数，超出则随机截断。"
            "默认 4950 = C(100,2)，即 100 个 episode 的全配对，不截断。"
        ),
    )
    dtw_resample_T: Optional[int] = Field(
        default=64, ge=2,
        description=(
            "DTW 计算前把每条序列线性重采样到的点数（默认 64）。"
            "必须与 Hausdorff 的重采样点数同口径，否则同一份报告里两个"
            "附录 B 观测项不可比；设为 null 表示不重采样直接用原始帧序列"
            "（此时 DTW/L 会显著偏小，且计算量随帧数平方增长）。"
        ),
    )
    acf_episodes: int = Field(
        default=5, ge=1,
        description="自相关观测抽样的 episode 数（默认 5）。",
    )
    acf_max_lag: int = Field(
        default=200, ge=2,
        description="自相关（含速度层过零搜索）的最大滞后阶。",
    )
    seed: int = Field(default=0, description="抽样随机种子，保证可复算。")
    # --- v0.11.0 视频层抽样 -------------------------------------------------
    # 视频解码比读 parquet 贵几个数量级（单帧 640×480 灰度 ≈ 300 KB，一次
    # seek+decode 区间包含上千帧），默认值刻意取小。
    video_episodes: int = Field(
        default=2, ge=1,
        description=(
            "视频层（6.2.5.2 a) 1) 运动模糊 / 附录 A）抽样的 episode 数"
            "（默认 2）。总数不足时按实际存在的 episode 数。"
        ),
    )
    video_frame_pairs: int = Field(
        default=8, ge=1,
        description=(
            "每个 episode 每路相机抽取的**相邻帧对**数（默认 8）。"
            "每对跑一次 Lucas-Kanade 金字塔光流。"
        ),
    )


class ConformProfile(BaseModel):
    """一份完整的符合性测评配置。"""

    standard: str = "gbz218.1-2026"
    profile_format_version: str = PROFILE_FORMAT_VERSION
    core_fields: List[str] = Field(default_factory=lambda: list(CORE_FIELDS))
    frame_loss_factor: float = Field(
        default=FRAME_LOSS_FACTOR, gt=1.0,
        description=(
            "帧间隔超过 frame_loss_factor × 标称采样周期即计为丢帧。"
            "标准未规定该系数，属工程口径，可配置。"
        ),
    )
    exposure_fraction: Optional[float] = Field(
        default=None, gt=0.0, le=1.0,
        description=(
            "曝光时间占帧周期的比例 α = 曝光时间 / 帧周期（曝光分数），"
            "口径为 ORIGIN_DECLARED（委托方声明值）。标准把全快门角度/曝光"
            "时间交给数据提供方，工具不得代为取值。声明后运动模糊核长度由"
            "光流上界 d 按下式换算：L = α·d；未声明则只能做单边判定"
            "（d ≤ 5 px 判 PASS，d > 5 px 判 UNDETERMINED，绝不判 FAIL）。"
        ),
    )
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    thresholds: Dict[str, ThresholdSpec] = Field(default_factory=dict)
    source: Optional[str] = Field(
        default=None,
        description="该 profile 的来源路径；内置默认 profile 为 None。",
    )

    # --- 阈值读取 ---------------------------------------------------------

    def threshold(self, key: str):
        """按 key 取阈值，返回 :class:`Threshold`（含 clause/binding/comparator）。

        内置表提供 clause/binding/comparator 的默认值，用户 profile 只覆盖
        显式给出的字段。取不到 comparator 时抛出异常——没有比较运算符就
        无法判定，静默回落会造出假结论。
        """
        from rda.conform.standards.gbz218_1 import Threshold

        base = all_thresholds().get(key)
        override = self.thresholds.get(key)
        if base is None and override is None:
            raise KeyError(f"unknown threshold key: {key!r}")
        if base is None:
            base_dict: Dict[str, Any] = {}
        else:
            base_dict = base.to_dict()
        if override is not None:
            for field_name, field_value in override.model_dump().items():
                if field_value is not None:
                    base_dict[field_name] = field_value
        # "—" 类（信息性输出，如 6.2.3.5 的覆盖矩阵）标准未设阈值，本来就没有
        # comparator；其余阈值缺 comparator 就无法判定，必须报错而不是静默通过。
        if not base_dict.get("comparator") and base_dict.get("binding") != BINDING_INFORMATIONAL:
            raise ValueError(
                f"threshold {key!r} has no comparator — cannot be evaluated"
            )
        return Threshold(**base_dict)

    def binding(self, key: str) -> str:
        """Return the binding ("应"/"宜"/"—") of a threshold key."""
        return self.threshold(key).binding


def default_profile() -> ConformProfile:
    """Build the built-in profile from the GB/Z 218.1-2026 threshold table.

    内置 profile 的 thresholds 故意留空：默认即标准原文，不需要也不应该
    把标准数值复制一份进来（复制就会有分叉）。覆盖只发生在用户 profile 里。
    """
    return ConformProfile()


def _coerce_threshold_payload(raw: Dict[str, Any]) -> Dict[str, ThresholdSpec]:
    """Validate the ``thresholds`` section of a profile file."""
    out: Dict[str, ThresholdSpec] = {}
    for key, payload in (raw or {}).items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"thresholds.{key} must be an object, got {type(payload).__name__}"
            )
        spec = ThresholdSpec(**payload)
        if spec.binding is not None and spec.binding not in _BINDING_CHOICES:
            raise ValueError(
                f"thresholds.{key}.binding must be one of {_BINDING_CHOICES}, "
                f"got {spec.binding!r}"
            )
        out[key] = spec
    return out


def load_profile(path: Optional[str | Path] = None) -> ConformProfile:
    """Load a profile, layering it on top of the built-in defaults.

    Args:
        path: JSON profile 路径。``None`` 表示只用内置默认 profile。

    Returns:
        :class:`ConformProfile`，``source`` 记录了文件路径。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: JSON 非法、顶层不是对象、阈值段结构不对或 binding 非法。
    """
    if path is None:
        return default_profile()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"profile is not valid JSON ({p}): {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"profile must be a JSON object, got {type(raw).__name__}")

    payload: Dict[str, Any] = dict(raw)
    payload.pop("source", None)
    thresholds_raw = payload.pop("thresholds", {}) or {}
    sampling_raw = payload.pop("sampling", None)

    profile = ConformProfile(thresholds=_coerce_threshold_payload(thresholds_raw))
    if sampling_raw is not None:
        if not isinstance(sampling_raw, dict):
            raise ValueError("profile field 'sampling' must be an object")
        profile.sampling = SamplingSpec(**sampling_raw)
    for field_name, field_value in payload.items():
        if field_name not in ConformProfile.model_fields:
            raise ValueError(f"unknown profile field: {field_name!r}")
        setattr(profile, field_name, field_value)
    profile.source = str(p)
    return profile


def profile_to_dict(profile: ConformProfile) -> Dict[str, Any]:
    """Serialize a profile (resolved thresholds included) for the report.

    输出的是**生效后**的阈值表（内置默认 + 用户覆盖合并结果），这样报告
    的读者不必再去翻代码才知道判定用的是哪个数。
    """
    resolved: Dict[str, Any] = {}
    for key in sorted(set(all_thresholds()) | set(profile.thresholds)):
        try:
            resolved[key] = profile.threshold(key).to_dict()
        except (KeyError, ValueError):
            continue
    return {
        "standard": profile.standard,
        "profile_format_version": profile.profile_format_version,
        "source": profile.source,
        "core_fields": profile.core_fields,
        "frame_loss_factor": profile.frame_loss_factor,
        "sampling": profile.sampling.model_dump(),
        "exposure_fraction": profile.exposure_fraction,
        "thresholds": resolved,
    }
