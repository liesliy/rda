# schema_consistency

> **Layer**: Layer 1 - Data Integrity (hard check)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — schema_consistency

Dtype/shape consistency check across frames and features within an episode: every feature must keep its declared dtype and first-dim (= num_frames) across the whole episode; mismatches are listed per frame. Maps to reason code REPAIRABLE in the recommendation gate (design doc v1.1).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

