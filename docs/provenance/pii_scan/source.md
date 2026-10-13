# pii_scan

> **Layer**: Layer 4 - Security & Privacy (WARNING-level by default)
> **Status**: 自研实现（无第三方代码） · RDA v0.9.22 provenance 记录

# 来源依据 — pii_scan

思想来源与公开先例（仅参考思想/规范，未复制任何代码）：

- **GB 11643-1999《公民身份号码》** — mod-11-2 校验位规则，作为
  `cn_resident_id` 的语义校验器。
- **IETF RFC 1918** — 私有 IPv4 地址空间划分（`10/8`、`172.16/12`、`192.168/16`），
  另含 loopback `127/8` 与 link-local `169.254/16`，作为 `ipv4` 检测器区分
  "真实地理暴露"与"仅网络拓扑暴露"的依据。
- **YD/T 6771-2026 具身智能数据集质量规范**（2026-11-01 实施）— 八项质量维度中的
  **安全性**为唯一 P0 准入项，本指标为该维度的部分技术覆盖。
- Internal MVP Spec v0.2.0, Layer 定义。

## 语义边界声明（同设计文档 §1.1）

- RDA 的指标是 **evidence-oriented measurement framework**，不是"数据质量标准"。
- **本指标不构成隐私合规证明**：不判定是否取得知情同意、不审查授权链、
  不做 OCR（车牌/门牌/工牌）、不做语音生物特征识别、不对每一帧做人脸检测。
  产出物是**待人工复核项**，不是验收结论。
- 涉及的政策/法域判定一律标注为 needs-human-review，不称"标准定义"。
