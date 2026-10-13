# pii_scan

> **Layer**: Layer 4 - Security & Privacy (WARNING-level by default)
> **Status**: 自研实现（无第三方代码，人脸检测部分调用 opencv 的公开 API） · RDA v0.9.22 provenance 记录

# 实现出处 — pii_scan

- **实现位置**: `rda/metrics/pii_scan.py`（注册表见 `rda/metrics/__init__.py`，
  层常量 `LAYER4_SECURITY_PRIVACY`）
- **关联修改**: `rda/audit/rules.py`（`critical_with()`）、
  `rda/audit/episode_audit.py` / `dataset_audit.py`（单次运行 CRITICAL 名单透传）、
  `rda/cli/main.py`（`--pii-video` / `--pii-strict`）、
  `rda/report/json_report.py`（`PRV-01` 与报告侧脱敏）
- **作者**: RDA 项目（NiuSu）原创实现，2026 年
- **第三方代码**: 无。全部为本仓库原创代码。
- **第三方运行时依赖（可选）**: `opencv-python-headless>=4.8,<5`，仅用于
  `cv2.CascadeClassifier` 与 `cv2.data.haarcascades` 的调用，未复制其任何代码。
  算法与环境兼容坑参见 `algorithm.md`。
- **参考方式**: 仅依据下列**公开标准/规范所述算法**自行编写实现：
  GB 11643-1999（公民身份号码校验位）、IETF RFC 1918（私有地址空间）、
  YD/T 6771-2026（质量维度定义）。标准文本本身不属于可执行代码对象，
  未复制任何第三方的源代码或配置文件。
