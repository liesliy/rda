# video_freeze

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录 · **D-27（2026-11-03）判据修订**


# 算法说明 — video_freeze

VA-A 视觉流完整性：检测"手臂在动而视频冻结"的相机掉线签名。将 episode 对应的视频段解码为 64×64 灰度帧，计算相邻帧平均绝对差分（下称"帧差"）。

## 判据（D-27 起）

相机掉线时相机**重发上一帧**，因此相邻帧的差分解码后精确为 0（容 codec 抖动）。这与"运动幅度小"是**种类差别，不是程度差别**。判据据此改为：

```
upper         = max(floor, min(0.5 × p25, 0.01))        # 绝不切进正常运动
duplicate_eps = min(max(floor, 0.25 × p10), upper)       # 跟随该段自身静息水平
frozen        = 帧差 ≤ duplicate_eps
```

`floor` 固定为 0.002（容 codec 抖动），**不随 profile 变**——两帧是否相同是关于文件的物理事实，不是判罚尺度；profile 只调"多少段/多长算 conclusive"（strict 2 段 / default 3 段 / lenient 5 段）。上界 `0.5 × p25` 是关键约束：它保证阈值永远落在该相机自身分布的低端，而不是落进中段。

结论分级不变：短停帧（≥0.5s 且手臂在动）记 REVIEW；多次停帧（≥3 次）或单次 ≥3s 或冻结占比 ≥30% 判 EXCLUDE。

顺带修正：`min_freeze_duration_sec` 此前**只在 profile 表里写着、代码没读**（strict 声明 0.3s、lenient 声明 1.0s，实际都按模块常量 0.5s 跑）。D-27 起最小段长取自 profile，`details.params.min_freeze_seconds` 也随之反映实际值。

## 为什么改（原来的阈值错在哪）

原判据 `max(0.10, 0.25 × p10)` 是**绝对阈值**，在 libero_10 上校准（正常帧差 ~1.0–3.0，慢动作 0.3–1.0，真停帧 < ~0.1）。问题在于 0.10 这个下限与**图像尺度无关**：

- 它只在"正常运动 ≥1.0"的视角下成立；
- 固定鸟瞰机位里机械臂只占画面几个百分点，正常帧差本身就只有 **0.12–0.17**，与 0.10 同量级；
- 于是阈值切在分布中段，**正常慢动作被整段判成卡死**。

FineART（Scale AI × HF LeRobot，40,543 集）实测（6 集 × 4 相机；数字取自 `rda audit` 报告里的 `details.stream_motion_profile`，可直接复现）：

| 相机 | p10 | p50 | p90 | 旧阈值 0.10 下判为 frozen 的帧 |
|---|---|---|---|---|
| cam_high（固定鸟瞰） | 0.028 | 0.156 | 0.645 | **37.9%（中位）/ 43.2%（最大）** |
| cam_low | 0.046 | 0.477 | 1.655 | 21.2% |
| cam_left_wrist | 0.255 | 2.429 | 12.136 | 2.8% |
| cam_right_wrist | 0.170 | 2.461 | 11.421 | 5.8% |

同一份数据里，**帧差精确为 0 的帧对占比为 0.00%**（24 段全为 0），即不存在任何真正的重发帧。误判产生的 36 个"冻结区"全在全分辨率复测中显示画面在动（区段均值 0.10–0.15，同集全片中位 0.29–0.87）。

`--freeze-motion-source state` 曾把结果从 5 EXCLUDE 降为 6 REVIEW，但那是绕过而非修复：它只是用 state 静止把误判段降级，判据本身仍会命中正常数据。

## 输出

measurement 含 checked_features / freeze_region_count；`details.freeze_regions` 逐段列出 feature、视频/parquet 帧区间、时长、moving_ratio、**mean_diff_in_span**、**duplicate_eps**（可直接看出该段到底是"0 变化"还是"小变化"）。`details.stream_motion_profile` 按相机给出 median_diff / p90_diff / duplicate_eps / duplicate_ratio，用于解释"这个视角本来就静"，避免把安静读成损坏。
