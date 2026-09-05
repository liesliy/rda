# velocity_acceleration

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — velocity_acceleration

Finite-difference kinematics on state/action channels: velocity = diff(state)/dt, acceleration = diff(velocity)/dt. Reports median/p95/p99/max velocity and extreme acceleration spike counts. Spikes are flag-only (RISK_SIGNAL), never auto-deleted (audit positioning).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

