"""GB/Z 218.1-2026 内置阈值常量表。

本模块是 conform 子包唯一的"标准数值来源"。**每一条常量都必须携带
``clause``（条款号）与 ``binding``（"应"/"宜"/"—"）**，不允许出现没有
条款来源的裸数字——这是能被第三方复核的前提。

binding 语义（GB/T 1.1 用词惯例）：
  ``应``（mandatory）    标准要求的强制性条款，违反即不符合。
  ``宜``（advisory）     标准建议的推荐性条款，违反仍判 FAIL，但在汇总中
                         单列 ``advisory_fail``，不与强制性 FAIL 混计。
  ``—``（informational）标准只要求输出/声明，未设符合性阈值，不构成判定。

数字来源纪律：
  以下数值均直接取自 GB/Z 218.1-2026 正式版正文。任何修改都必须同步更新
  ``clause`` 与 ``note``，并说明依据。未查到条款号的内容一律标 ``—``，
  不得用"看起来合理"的数字填充。

口径来源（origin）纪律
----------------------
一份要被第三方采信的报告，必须让读者一眼分辨出每个数是**哪儿来的**：

  ``standard``    GB/Z 218.1-2026 正文给出的阈值，照抄。
  ``engineering`` 标准没给数，本实现为可判定而自行选定的工程口径
                  （如借用他条阈值、或设定识别系数）。**必须在报告里标明**，
                  不许伪装成标准值。
  ``declared``    标准把值交给数据提供方声明（如可行解空间最大距离），
                  本实现不得代为取值。

三者混写是这份报告最容易被击穿的地方：读者无法区分"这是国标要求的"和
"这是工具自己定的"。故 origin 是 Threshold 的必填字段。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# --- binding 取值 ---------------------------------------------------------
BINDING_MANDATORY = "应"
BINDING_ADVISORY = "宜"
BINDING_INFORMATIONAL = "—"

# --- origin（口径来源）取值 -----------------------------------------------
ORIGIN_STANDARD = "standard"
ORIGIN_ENGINEERING = "engineering"
ORIGIN_DECLARED = "declared"

ORIGIN_LABEL = {
    ORIGIN_STANDARD: "标准值",
    ORIGIN_ENGINEERING: "工程口径（非标准值）",
    ORIGIN_DECLARED: "委托方声明值",
}

STANDARD_ID = "gbz218.1-2026"
STANDARD_TITLE = "机器人 数据集质量 第1部分：评测指南（GB/Z 218.1-2026）"


@dataclass(frozen=True)
class Threshold:
    """一条带条款出处与 binding 的标准阈值。

    Attributes:
        clause: 标准条款号，例如 ``"6.2.2.1 a"``。没有条款号时填 ``"—"``，
            并在 ``note`` 中说明为什么没有（严禁回填猜测的条款号）。
        binding: ``"应"`` / ``"宜"`` / ``"—"``。
        value: 阈值本体。可以是数值、布尔（"应提供"类条款）或列表
            （"输出矩阵"类条款）。
        comparator: 比较运算符，决定**恰好等于阈值**时的行为：
            ``"<"`` 恰好等于即不满足；``"<="`` 恰好等于即满足；
            ``">="`` 恰好等于即满足；``"=="`` 相等才满足。
            布尔型条款用 ``"is_true"``。
        quantity: 被测量的量名，用于报告可读性。
        unit: 单位。比值类用 ``"ratio"``，无量纲计数用 ``"count"``。
        note: 口径说明/来源备注。
        origin: 口径来源，``standard`` / ``engineering`` / ``declared``
            之一。见模块 docstring 的"口径来源纪律"。
    """

    clause: str
    binding: str
    value: Any
    comparator: str = ""
    quantity: str = ""
    unit: str = ""
    note: str = ""
    origin: str = ORIGIN_STANDARD

    def describe(self) -> str:
        """Return a human-readable 'quantity comparator value unit' string."""
        if self.comparator == "is_true":
            return f"{self.quantity} 应成立"
        if self.unit == "ratio":
            val = f"{float(self.value) * 100:g}%"
        elif self.unit == "count":
            val = f"{self.value}"
        elif self.unit == "s":
            val = f"{self.value} s"
        elif self.unit == "px":
            val = f"{self.value} px"
        else:
            val = str(self.value)
        return f"{self.quantity} {self.comparator} {val}".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (profile files use the same shape)."""
        return {
            "clause": self.clause,
            "binding": self.binding,
            "value": self.value,
            "comparator": self.comparator,
            "quantity": self.quantity,
            "unit": self.unit,
            "note": self.note,
            "origin": self.origin,
        }


# --- 6.2.2.1 数据完整性：字段缺失率 ---------------------------------------

CORE_FIELD_MISSING_RATE = Threshold(
    clause="6.2.2.1 a",
    binding=BINDING_MANDATORY,
    value=0.05,
    comparator="<",
    quantity="核心字段缺失率",
    unit="ratio",
    note="核心字段指 timestamp / frame_index / action / observation.state。",
    origin=ORIGIN_STANDARD,
)

# 6.2.2.1 a 定义的核心字段名。它们是"缺了就无法定位一帧"的最小集合。
CORE_FIELDS = ("timestamp", "frame_index", "action", "observation.state")

GENERAL_FIELD_MISSING_RATE = Threshold(
    clause="6.2.2.1 b",
    binding=BINDING_MANDATORY,
    value=0.20,
    comparator="<",
    quantity="一般字段缺失率",
    unit="ratio",
    note="一般字段 = 数据集声明的全部特征中，不属于 6.2.2.1 a 核心字段的那些。",
    origin=ORIGIN_STANDARD,
)

OUT_OF_THRESHOLD_HANDLING_DECLARED = Threshold(
    clause="6.2.2.1 c",
    binding=BINDING_MANDATORY,
    value=True,
    comparator="is_true",
    quantity="超阈值数据的处理方式说明",
    unit="boolean",
    note="数据集应提供超阈值（out-of-range）数据的处理方式说明。",
    origin=ORIGIN_STANDARD,
)

# --- 6.2.2.3 时间一致性 ---------------------------------------------------

TIMESTAMP_JITTER_STD_RATIO = Threshold(
    clause="6.2.2.3",
    binding=BINDING_ADVISORY,
    value=0.10,
    comparator="<=",
    quantity="时间戳抖动标准差",
    unit="ratio",
    note="限值为「采样周期 × 10%」，即 std(diff(timestamp)) ≤ 0.1 / fps。",
    origin=ORIGIN_STANDARD,
)

CONSECUTIVE_FRAME_LOSS_RATE = Threshold(
    clause="6.2.2.3",
    binding=BINDING_ADVISORY,
    value=0.01,
    comparator="<=",
    quantity="连续丢帧率",
    unit="ratio",
    note=(
        "标准给出 1% 限值，但未规定「丢帧」的识别系数；本实现把帧间隔 "
        "超过 frame_loss_factor × 标称采样周期的间隔计为丢帧，该系数可由 "
        "profile 覆盖（默认 1.5）。"
        "注意：1% 是标准值，1.5 倍识别系数是工程口径，两者不可混为一谈。"
    ),
    origin=ORIGIN_STANDARD,
)

# 标准未规定；工程口径，可配置。
FRAME_LOSS_FACTOR = 1.5
FRAME_LOSS_FACTOR_ORIGIN = ORIGIN_ENGINEERING

# --- 6.1.1 f / 6.2.5.2 多模态时间基准 -------------------------------------

MULTIMODAL_TIME_BASE_UNIFIED = Threshold(
    clause="6.1.1 f",
    binding=BINDING_MANDATORY,
    value=True,
    comparator="is_true",
    quantity="多模态统一时间基准",
    unit="boolean",
    note="帧层与视频层须同源，且跨视角可比。",
    origin=ORIGIN_STANDARD,
)

# 6.1.1 f 只说"同源/可比"，未给数值容差。本实现复用 6.2.5.2 c) 4) 的 1 ms
# 对齐误差作为容差，避免另造一个标准里不存在的数字；可由 profile 覆盖。
TIME_BASE_TOLERANCE_S = Threshold(
    clause="6.2.5.2 c) 4)",
    binding=BINDING_ADVISORY,
    value=0.001,
    comparator="<=",
    quantity="跨视角时间基准容差",
    unit="s",
    note=(
        "【工程口径，非标准值】6.1.1 f 只要求「同源/可比」，未给任何容差数值；"
        "此处沿用 6.2.5.2 c) 4) 的多传感器时间戳对齐误差限值 1 ms 作为可量化"
        "口径。该 1 ms 不是 6.1.1 f 的标准要求值，可由 profile 覆盖。"
        "另注：本判定的可比量是各视频流的时长（to − from），不是 from/to 的"
        "绝对值——LeRobot v3 的 from/to 是视频文件内相对偏移，跨视角直接相减"
        "得到的只是分块差异，不是时间基准差异。"
    ),
    origin=ORIGIN_ENGINEERING,
)

SENSOR_ALIGNMENT_ERROR_S = Threshold(
    clause="6.2.5.2 c) 4)",
    binding=BINDING_ADVISORY,
    value=0.001,
    comparator="<=",
    quantity="多传感器时间戳对齐误差",
    unit="s",
    note="多传感器时间戳对齐误差应 ≤ 1 ms。",
    origin=ORIGIN_STANDARD,
)

# --- 6.2.3 内容多样性 -----------------------------------------------------

TASK_TYPE_MIN_KINDS = Threshold(
    clause="6.2.3.2",
    binding=BINDING_ADVISORY,
    value=2,
    comparator=">=",
    quantity="任务/技能类型种类数",
    unit="count",
    note="任务/技能类型应不少于 2 种。",
    origin=ORIGIN_STANDARD,
)

MODALITY_CATEGORIES = Threshold(
    clause="6.2.3.5",
    binding=BINDING_INFORMATIONAL,
    value=("视觉", "声学", "力学", "惯性", "三维感知", "本体感知"),
    comparator="",
    quantity="传感器模态覆盖矩阵",
    unit="matrix",
    note="标准要求输出六类传感器模态的覆盖矩阵，未设符合性阈值。",
    origin=ORIGIN_STANDARD,
)

TRAJECTORY_HAUSDORFF_RATIO = Threshold(
    clause="6.2.3.4 a",
    binding=BINDING_ADVISORY,
    value=0.30,
    comparator=">=",
    quantity="轨迹 Hausdorff 均值 / 可行解空间最大距离",
    unit="ratio",
    note=(
        "标准要求轨迹 Hausdorff 均值 ≥ 可行解空间最大距离的 30%。"
        "「可行解空间最大距离」必须由数据提供方声明（--workspace-bounds）；"
        "未声明时不得用绝对距离反推，一律判 UNDETERMINED。"
    ),
    origin=ORIGIN_DECLARED,
)

# --- 6.2.7 标识 -----------------------------------------------------------

DOI_URI_PRESENT = Threshold(
    clause="6.2.7 f",
    binding=BINDING_MANDATORY,
    value=True,
    comparator="is_true",
    quantity="DOI/URI 全链路唯一标识",
    unit="boolean",
    note="数据集应提供 DOI 或 URI 形式的全链路唯一标识。",
    origin=ORIGIN_STANDARD,
)

# --- 视频像素层判定（本版本未实现） ---------------------------------------
# 这些判定需要解码视频像素，P0 不引入 opencv/av/torch，故一律 NOT_APPLICABLE。
# 条款号已按 GB/Z 218.1-2026 正式版补录：6.2.5.2 a) 1) 与附录 A.1~A.3。
# 仍未查到条款号的项（曝光异常、帧间分辨率一致性）继续留 ``—`` 并标"待补录"，
# 不编造条款号。
MOTION_BLUR_KERNEL_PX = Threshold(
    clause="6.2.5.2 a) 1)",
    binding=BINDING_ADVISORY,
    value=5,
    comparator="<=",
    quantity="运动模糊核长度",
    unit="px",
    note="运动模糊核长度宜 ≤ 5 像素。",
    origin=ORIGIN_STANDARD,
)

# 附录 A：标准给出的"推荐方法/公式"，不是符合性判定项，故 binding 为 "—"。
APPENDIX_A_RECOMMENDED = (
    Threshold(
        clause="附录 A.1",
        binding=BINDING_INFORMATIONAL,
        value="Lucas-Kanade 光流 或 金字塔光流",
        comparator="",
        quantity="光流计算方法",
        unit="method",
        note="标准推荐 Lucas-Kanade 或金字塔光流计算帧间光流，未设阈值。",
        origin=ORIGIN_STANDARD,
    ),
    Threshold(
        clause="附录 A.2",
        binding=BINDING_INFORMATIONAL,
        value="SF(t) = Σ_k ( |X_k(t)| − |X_k(t−1)| )",
        comparator="",
        quantity="频谱流量 SF(t)",
        unit="formula",
        note="频谱流量：相邻帧频谱幅度差之和，用于刻画画面变化量，未设阈值。",
        origin=ORIGIN_STANDARD,
    ),
    Threshold(
        clause="附录 A.3",
        binding=BINDING_INFORMATIONAL,
        value="D(t)",
        comparator="",
        quantity="点云运动动态性 D(t)",
        unit="formula",
        note="点云运动动态性，用于刻画三维点云随时间的变化强度，未设阈值。",
        origin=ORIGIN_STANDARD,
    ),
)

VIDEO_PIXEL_LAYER_CHECKS: tuple = (
    f"{MOTION_BLUR_KERNEL_PX.clause} {MOTION_BLUR_KERNEL_PX.describe()}"
    f"（{MOTION_BLUR_KERNEL_PX.binding}）",
    "曝光异常（过曝/欠曝）——条款号待补录",
    "帧间分辨率一致性——条款号待补录",
)
VIDEO_PIXEL_LAYER_CLAUSES: tuple = (
    MOTION_BLUR_KERNEL_PX.clause,
    "—",
    "—",
)
VIDEO_PIXEL_LAYER_NOTE = "需要视频像素层，本版本未实现"


def all_thresholds() -> Dict[str, Threshold]:
    """Return the default threshold table keyed by machine-readable name.

    The keys are the same names used in profile JSON files, so a profile
    can override any single entry without redefining the whole table.
    """
    return {
        "core_field_missing_rate": CORE_FIELD_MISSING_RATE,
        "general_field_missing_rate": GENERAL_FIELD_MISSING_RATE,
        "out_of_threshold_handling_declared": OUT_OF_THRESHOLD_HANDLING_DECLARED,
        "timestamp_jitter_std_ratio": TIMESTAMP_JITTER_STD_RATIO,
        "consecutive_frame_loss_rate": CONSECUTIVE_FRAME_LOSS_RATE,
        "multimodal_time_base_unified": MULTIMODAL_TIME_BASE_UNIFIED,
        "time_base_tolerance_s": TIME_BASE_TOLERANCE_S,
        "sensor_alignment_error_s": SENSOR_ALIGNMENT_ERROR_S,
        "task_type_min_kinds": TASK_TYPE_MIN_KINDS,
        "modality_coverage": MODALITY_CATEGORIES,
        "trajectory_hausdorff_ratio": TRAJECTORY_HAUSDORFF_RATIO,
        "doi_uri_present": DOI_URI_PRESENT,
    }
