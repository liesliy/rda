# pii_scan

> **Layer**: Layer 4 - Security & Privacy (WARNING-level by default)
> **Status**: 自研实现（无第三方代码） · RDA v0.9.22 provenance 记录

# 许可 — pii_scan

- **本指标实现**: 随 RDA 主仓库以 MIT License 发布（见仓库根 `LICENSE`）。
- **可选运行时依赖**: `opencv-python-headless`（Apache-2.0）。RDA 仅**调用**
  其 API（`cv2.CascadeClassifier` 检测、`cv2.data.haarcascades` 定位级联文件），
  未 vendored、未复制其源代码。启用方式为用户显式安装可选依赖组
  `pip install robot-data-audit[pii]`，属于运行期动态链接场景，不触发源码
  复制许可义务。
- **上游参考来源许可**:

  - GB 11643-1999 / RFC 1918 / YD/T 6771-2026：规范性文献，非代码对象；
    仅按其所述算法自行实现。
  - 无其他上游代码来源。

- **合规结论**: 仅参考思想/规范不受版权保护（参见 US Copyright Circular 31,
  *Copyright Does Not Protect Ideas*）；本实现不包含任何上游源代码，不触发上游
  许可证的代码复制义务。provenance 文件即为该结论的书面证据链。
