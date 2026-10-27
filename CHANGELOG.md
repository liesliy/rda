# Changelog

## 0.11.0 - 2026-10-27

新增**视频像素层**（6.2.5.2 a) 1) 运动模糊核长度 + 附录 A 动态性测量）。0.10.x 里这一层是一个三合一的 `NOT_APPLICABLE` 占位项，本版把它换成一条真判定和三条例行观测。

### Added
- **新模块 `rda/conform/video.py`** —— 解码与光流底座：
  - `av` / `cv2` 双延迟导入，缺失时按 `DatasetView._require_pyarrow` 同款模式降级（`av_available` / `cv2_available` 置 False + `ImportError`），note 明写"本地环境未安装 av，非数据集缺陷"。
  - **原分辨率**区间解码：`rda/metrics/visual_integrity.py:99` 的 `_decode_span_gray` 会缩到 64×64，那个尺寸足以判冻结/抖动，但**抹掉了 5 px 尺度的核长度信息**，因此新解码器不降采样（可选轻度高斯平滑只做前处理，尺寸不变）。
  - fps **只从容器元数据读**（`average_rate` → `base_rate` → `guessed_rate`），读不到即 `None` → 上层 `UNDETERMINED`。**不假定 30 或 60**。
  - Lucas-Kanade 金字塔光流（`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`，附录 A.1 明文允许）；可靠性判据为 `status==1` 且位移 < 图像对角线 × 10%，**被剔除的点与帧一律计数并进报告，不静默丢弃**。
  - **全程内存内 seek + decode，不落盘任何临时文件**（比"下载后删除"更干净，也避开 90 GB 数据集的磁盘问题）。
- **`motion_blur_kernel_px` 判定项**（条款 6.2.5.2 a) 1)，宜，限值 5 px）。标准给限值不给方法，本实现的口径是**光流上界法**：帧间光流位移 d 是模糊核长度 L 的几何上界，`L = α·d`，`α = 曝光时间/帧周期 ≤ 1`，出处 David Korcak & Jiri Matas, *Video shutter angle estimation using optical flow and linear blur*, arXiv:2303.10247（CVWW 2024）。三条支路：
  1. 视频不可得 / `--video=off` / av+cv2 缺失 → `UNDETERMINED`（措辞沿用"本地副本不完整，非数据集本身缺陷"）；
  2. α 已由委托方声明 → `L = α·d`，双边判 PASS / FAIL；
  3. α 未声明 → `d ≤ 5 px` 判 PASS（`value` 留 `None`，因为没测出核长度，只测出上界）；`d > 5 px` **判 `UNDETERMINED`，绝不判 FAIL**（没有 α 就排除不掉 L 仍 ≤ 5 px）。
  - 红线已写进 method：**不使用** `visual_quality.py:64` 的 Laplacian 方差等清晰度代理指标——那类指标分不清散焦与运动拖尾，拿来比 5 px 就是方法不符。
- **附录 A 三项观测**（报告第 4 章，binding `—`，**不产 PASS/FAIL**）：`appendix_a_optical_flow`（A.1，p50/p95/max + 有效点/总点 + 剔除帧数）、`appendix_a_spectral_flux`（A.2 公式 A.2）、`appendix_a_pointcloud_dynamics`（A.3 公式 A.3）。理由：7.2.6 a) 是对**评价方法**的要求，不是数据属性，因此不构成符合性判定。
- **CLI**：`rda conform --video=sample|full|off`，**默认 sample**。sample = `sampling.video_episodes` 个 episode × 每路相机 × `sampling.video_frame_pairs` 帧对；full = 全部 episode（帧对数仍受限）；off = 不读像素，相关项全部 `UNDETERMINED`。
- **profile**：`sampling.video_episodes=2`、`sampling.video_frame_pairs=8`（沿用 seed=0 的 `Sampler`）；新增顶层 `exposure_fraction`（口径 `ORIGIN_DECLARED`，标准把曝光时间交给数据提供方，工具不得代为取值）。JSON profile 仍不引 pyyaml。
- **`motion_blur_kernel_px` 补进 `all_thresholds()`**：该阈值自 0.10.0 起存在却没注册，profile 覆盖不到它——它是唯一一个有条款、有限值却不受 profile 控制的判定项。
- **报告**：新增 `## 5. 观测项（附录 A 动态性测量）`（含逐相机表格 + **逐 episode 明细** + 按特征级的载荷覆盖表 + 剔除规则 + 百分位插值方法 + 临时文件声明），原 5/6/7 章顺延为 6/7/8。
- **`video` 可选依赖组**扩展为 `["av>=12.0", "opencv-python-headless>=4.8,<5"]`（opencv 上界沿用 `pii` 组的实测依据：4.14.0 自带 Haar 级联 XML，5.0.0 一个都不带）。

### Changed
- **0.10.0 中"本版本不得声称符合附录 A"的表述已作废**，按实现状况如实改为：**A.1 已实现（LK 金字塔）；A.2 仅适用于音频模态；A.3 仅适用于含点云的数据集**。
- **抽样池按 episode 的 `file_index` 严格找文件**：`videos/<feature>/chunk-*/file-<idx>.mp4` 找不到就跳过该 episode 并披露，**不许拿同特征的另一个 mp4 顶替**——同一个文件里多个 episode 的 `from/to` 偏移不同，顶替会把别人的画面记成这个 episode 的位移，报告上还看不出来。抽样只在"本地可核验的那部分 episode"里做，并在报告里写明池大小与总 episode 数。
- **视频层按特征级判断载荷覆盖**：`videos/` 目录在不在不算数，按 `videos/<feature>/**` 的文件数逐个特征判定（本次 `lerobot_folding` 副本只有 base 有文件，正好是一个天然负样本）。

### Removed
- 三合一占位项 `check_video_pixel_layer` / metric `video_pixel_layer_checks`，以及 `VIDEO_PIXEL_LAYER_CHECKS` / `VIDEO_PIXEL_LAYER_CLAUSES` / `VIDEO_PIXEL_LAYER_NOTE` 三个常量。
- **占位项里的"曝光异常（过曝/欠曝）"与"帧间分辨率一致性"**：经逐字核对 GB/Z 218.1-2026 全文，**标准中没有这两条要求**，所谓"条款号待补录"永远补不上——留着等于把凑不出条款号的自选指标挂在国标名下，故移除。

### Tests
- `tests/test_conform_video.py`（11 项）：**合成真 mp4 验证光流数值精度**——每帧平移 3 px 的方块视频，解码跑光流，libx264 与 mpeg4 两种编码器下恢复位移均为 **3.0000 px（真值 3.0，误差 0.0000 px < 0.5 px 容差）**；另覆盖依赖缺失降级、假 mp4 / 无视频轨 / 帧率不可知 / 不产生临时文件。
- `tests/test_conform_video_checks.py`（20 项）：三条支路开关锁（含"d > 5 且 α 未声明 → UNDETERMINED 而非 FAIL"、α=0.2/0.9 的 PASS/FAIL 换算、profile 覆盖阈值生效）、per-feature 部分载荷、缺文件不借用、`--video=sample|full|off` 三档 CLI、报告章节编号与逐 episode 明细。
- 全量：**407 passed, 5 skipped, 0 failed**（基线 376 passed, 5 skipped）。

### 真实数据集回归（`lerobot_folding`，1200 episode / 30 fps）
- `rda conform --video=sample` 端到端 **15.2 s**（其中视频层约 4.1 s，其余为附录 B 的 Hausdorff/DTW）；`--video=full` 端到端 **15.3 s**（10 个可核验 episode 全部跑完，视频层 4.4 s）。
- 只有 `observation.images.base` 有本地载荷文件（1 个 file-000.mp4，容器元数据 fps **30.0**，含 episode 0–9），`left_wrist` / `right_wrist` payload 文件数 **0** → 未参与判定，报告按特征级披露。
- `--video=full` 实测（10 episode × 8 帧对 × 每帧最多 200 个跟踪点）：base 聚合 p50 **0.5421** / p95 **7.6327** / max **75.0426** px，有效跟踪点 15262/15281，剔除帧对 0。逐 episode：ep0 p50 **2.5529** / p95 **10.9739** / max **16.2708**；ep1 p50 **0.6579** / p95 **6.5637** / max **19.2846**。
- 判定结论：`UNDETERMINED` —— p95 = 7.6327 px（full）/ 14.6533 px（sample 抽到 ep6、ep7）超过 5 px，而委托方未声明曝光分数 α，**不许判 FAIL**。附上 `--profile` 声明 α 即可转为可裁定结论。
- 抽样：sample 档 seed=0 抽到 episode [6, 7]（抽样池 10 / 总 1200）。

### 已知边界（不掩饰）
1. **附录 A 三项不是判定项**：A.1 已实现；**A.2 仅适用于音频模态**（标准 A.2 原文通篇讲"音频信号"，本工具不把视频帧的 2D FFT 拿去算 SF(t) 冒充）；**A.3 仅适用于含三维点云特征的数据集**。无相应模态时记 `NOT_APPLICABLE`。
2. **以下参数是工程口径（非标准值）**，报告已标注：对角线的 10% 剔除门限、单帧对最少 5 个有效点、每帧最多 200 个跟踪点、"取各相机 p95 的最大值"的聚合方式、STFT 窗（Hann，nperseg=512，hop=256）。
3. **上界法的假设与失效场景**（已写进 method）：假设全局快门、帧内运动近似匀速、曝光时间 ≤ 帧周期；滚动快门（逐行有效 α 不同）、剧烈非匀速运动、遮挡边界粗大运动、相机自运动与被跟踪物体运动不可分离时，d 不再是该点模糊核长度的上界。
4. **本次结论只对抽到的 episode 成立**：本地副本只有 10 个 episode 有视频载荷（总 1200），不得外推到整个数据集。
5. **A.2 / A.3 在本数据集上没有任何实测数字**（无音频、无点云），只有"不适用"——没有数字不等于测过了。
6. **coverage 门禁仍是桩**：`--strict-coverage` 只记录，不触发退出码 3。
7. 本工具不声称数据集合规。GB/Z 218.1-2026 是**国家标准化指导性技术文件**，本身不具有强制性（《国家标准化指导性技术文件管理规定》第九条）；conform 产出的是**可复核的判定依据**，不是认证证书。

## 0.10.1 - 2026-10-27

纯修复版，不动任何判定阈值、指标集合或 CLI 行为。合规结论与 0.10.0 完全一致。

### Fixed
- **装了默认包却读不了 LeRobot 数据集**。`pip install robot-data-audit` 之后直接 `rda conform <lerobot_dataset>` 会硬停在 `pyarrow is required` —— 因为 pyarrow 此前只是 `lerobot>=0.6` 的传递依赖，而 base 依赖只有 click / pydantic / numpy / requests。而 `rda conform` 的主要对象恰恰就是 LeRobot parquet 数据集，**默认场景第一步就撞墙**。新增可选依赖组 `parquet = ["pandas>=2.0", "pyarrow>=14.0"]`，两处报错文案改为推荐 `pip install "robot-data-audit[parquet]"`（`rda/conform/engine.py`、`rda/io/lerobot_loader.py`）。`rda/conform/engine.py` 的文案保留 "pandas and pyarrow" 子串，因为 `rda/cli/main.py` 靠它区分"缺 parquet 依赖"与"缺 lerobot"两种提示。
- **`all` 组漏了 `pii`**：`all` 此前包含 `lerobot/video/ui` 三组却不含 `opencv-python-headless`，装 `all` 不等于装全部。已补齐，并加入 `parquet` 组。

### Fixed (CI)
- **main 上的 CI 从 0.9.22 起连续三次推送全红**（`cdfe9fa` / `cdd0883` / `75f96ec`）。根因是 `tests/test_pii_scan.py::TestGracefulDegradation::test_opencv_installed_without_cascade_is_recorded`：该用例 monkeypatch 了 `_resolve_cascade`，但 `_opencv_missing()` 先 `import cv2`，CI 没装 opencv 就直接返回 `opencv_missing`，断言 `== "haar_cascade_missing"` 必失败。**测试随环境腐化，不是产品缺陷**。
  - 用例改为注入一个 `data/haarcascades` 为空的假 cv2 模块（正是 opencv 5.x 的形状），于是无论真实 opencv 在不在，这条"opencv 在但级联缺失"的路径都被覆盖 —— 比加 skip 更好，加 skip 会让最关键的环境反而没有覆盖。
- 本地镜像 CI 环境（`-e ".[ui,video,pii,parquet]"`）全量复跑：**376 passed, 5 skipped, 0 failed**。
- 遗留：CI 工作流本身尚未改成安装 `pii`/`parquet` 两组（`.github/workflows/ci.yml` 的改动需要带 `workflow` 权限的 token 才能推送，当前 token 只有 `repo`）。在那之前 CI 仍以"注入假 cv2"的方式覆盖该路径，结论不受影响。

## 0.10.0 - 2026-10-27

### Added
- **`rda conform` 子命令 —— GB/Z 218.1-2026 符合性测评模式**。`rda audit` 回答"这批数据好不好用"，`rda conform` 回答"这批数据过不过标准"。后者每一条判定都绑定到一个**编号条款**和一个**效力**（`应` 强制性 / `宜` 建议性），结论是四态之一：`PASS` / `FAIL` / `NOT_APPLICABLE` / `UNDETERMINED`。
- **新子包 `rda/conform/`**（8 个模块）：
  - `standards/gbz218_1.py`——内置阈值表。每个常量都带 `clause`（条款号）和 `binding`（应/宜）两个字段，**不允许存在没有条款来源的裸数字**。
  - `verdict.py`——四态枚举 + `ConformanceItem`（clause / metric / value / threshold / binding / verdict / method / note）+ 汇总。
  - `checks.py`——12 项判定：字段缺失率两档（6.2.2.1 a/b）、超阈值处理说明（6.2.2.1 c）、时间戳抖动（6.2.2.3）、连续丢帧率（6.2.2.3）、统一时间基准（6.1.1 f）、多传感器对齐（6.2.5.2 c4）、任务种类数（6.2.3.2）、模态覆盖矩阵（6.2.3.5）、DOI/URI（6.2.7 f）、轨迹 Hausdorff（6.2.3.4 a）等。
  - `timeseries.py`——附录 B 三项：自相关 B.1、傅里叶 B.2、DTW B.3。
  - `engine.py` / `profile.py` / `report.py`——编排、JSON profile（刻意不引 pyyaml，保持核心依赖只有 click/pydantic/numpy/requests）、markdown 报告。
- **报告输出三类口径的显式区分**：`标准值`（标准正文给出，照抄）、`工程口径（非标准值）`（标准没给数字、本工具为可判定而自选，复核时须单独审视）、`委托方声明值`（标准把该值交给数据提供方，工具不得代取）。三者混为一谈的符合性报告没有可信度，因此这个区分落在报告正文里，不放在注释里。
- **CLI**：`rda conform <ds> [--standard gbz218.1-2026] [--profile FILE] [--workspace-bounds ...] [--format md|json] [--out DIR]`。退出码 0/1/2 三态。

### Changed
- **时间同步判据改为 `duration = to − from`**（本版本最重要的方法学修正）。此前跨视角直接相减 `from_timestamp` / `to_timestamp` 得到 1,033 秒的"偏差"，并据此判定"统一时间基准不成立"。这是**误读**：LeRobot v3 的 from/to 是**各自 mp4 文件内的相对偏移**，其数值取决于该 episode 在其所属 mp4 中的分块位置（base 相机打包为 240 个 mp4，wrist 相机分块方式不同），跨视角相减没有物理意义。改用时长比对后，三路相机 duration 均值**完全相同**（90.394333 s），跨视角最大偏差 1.14e-13 s，1,200/1,200 episode 一致。**结论由"不符合"翻转为"合格"。** 该 1,033 秒重新定性为**使用陷阱**（误用会导致跨视角样本错位），而非数据缺陷。
- **判定红线的三条硬约束**，写进代码而不只是文档：
  1. 视频层未下载时判 `UNDETERMINED`，**不把"本地副本不完整"当成"数据集缺字段"**。初版曾因此把 6.2.2.1 b 判成 50% > 20% 的应类 FAIL——因为自己没下数据（完整仓库 90 GB 中视频占 89.93 GB）就判被测对象违反国标，这种误判会让整份报告失去可信度。
  2. 可行解空间未声明时判 `UNDETERMINED`，**不许用实测 Hausdorff 值反推阈值**。
  3. `UNDETERMINED` 的输出路径与 `PASS` 完全隔离，任何一条都不能被静默写成"合格"。

### Tests
- `tests/test_conform.py`（55 项）：四态各有用例、阈值边界值（恰好等于阈值时的行为）、字段缺失、空 episode、自定义 profile 覆盖加载生效。
- `tests/test_conform_timeseries.py`（23 项）：附录 B 三项对已知解析解的对照，其中自相关 B.1 钉住**分母取全长度 T 项求和**（不是 T−k 项）——这是本标准的特殊口径，写错即方法不符。
- 真实数据集回归：`lerobot/high_quality_folding`（1,200 episode / 3,254,196 帧 / 30.131 h）全量扫描，实测核心字段缺失率 0.0、时间戳抖动 4.618657e-06 s（限值 3.333333e-03 s，裕度 721.7×）、连续丢帧率 0/3,252,996、任务种类 1、跨视角 duration 偏差 1.14e-13 s——与独立测算基线逐项一致。

### 已知边界（不掩饰）
1. **视频像素层未实现**：附录 A 的光流、频谱流量、点云动态性，以及 6.2.5.2 a) 1) 运动模糊核长度（≤5 px），本版本一律 `NOT_APPLICABLE`。**因此本版本不得声称"符合 GB/Z 218.1-2026 附录 A"**——附录 A 是规范性附录，方法未实现则结果不可引用。
2. **附录 A 的算法选型尚未落实**：标准明文推荐 Lucas-Kanade 或金字塔光流，而此前验证脚本用的是 Farneback；A.2 的公式是频谱流量 `SF(t)=Σ(|X_k(t)|−|X_k(t−1)|)`，不是功率谱统计。这两项在落进 conform 前必须先改口径。
3. **coverage 门禁是桩**：`--strict-coverage` 只记录请求，不触发退出码 3。达标阈值与门禁语义尚未定义，不把不存在的功能说成已实现。
4. **两处工程口径不是标准值**：丢帧识别系数 1.5×、跨视角时间基准容差 1 ms。标准未给，可 profile 覆盖，报告中已标注为"工程口径"。
5. **DTW 仍是 O(n·m)**：默认抽样 100 episode / 4,950 对约 55 s，1,200 episode 全量不可行。抽样规模在报告中如实写出，不声称全量。
6. **本工具不声称数据集合规**。GB/Z 218.1-2026 是**国家标准化指导性技术文件**，本身不具有强制性（《国家标准化指导性技术文件管理规定》第九条）。conform 产出的是**可复核的判定依据**，不是认证证书。

## 0.9.22 - 2026-10-13

### Added
- **`pii_scan` 指标 —— 安全性维度首次覆盖**：YD/T 6771-2026 八项质量维度里唯一的 P0 准入项是安全性，此前 RDA 覆盖为零。新指标检测三类**客观可复现的暴露证据**：
  1. **文本面**（episode meta、task 文本、特征名）中的直接标识符：中国手机号、**中国居民身份证号（按 GB 11643-1999 mod-11-2 校验位验证）**、邮箱、MAC 地址、IPv4（按 RFC1918 / loopback / link-local 区分公有与私有，仅公网地址记为暴露）。
  2. **数值特征中的精确地理坐标**：lat / lon / GNSS / UTM 需成对出现且 95% 采样落在合法区间才记 finding；列名匹配由子串前缀改为分隔符整词匹配（`gps_lat`、`robot.lon` 现在能识别，同时 `translation` / `plate` / `relation` / `isolation` 内部含 "lat" 不再误报）。
  3. **视频帧人脸**：`--pii-video` 开启时对每个相机采样帧跑 Haar 级联检测（需 `opencv-python-headless>=4.8,<5`）。
- **判定策略：默认 WARNING，绝不默认的闸门。** 与 `numeric_sanity` 同一惯例：有发现时 `has_finding=True`、`assessment.status="pass"`，**判定不受影响**。一次误报的人脸检测不应该静默丢掉一段可用的示教数据。`--pii-strict` 才把 `pii_scan` 提升为本次运行的 CRITICAL 闸门（ HELP 文案已写明默认值）。实现上新增 `critical_with()`（`rda/audit/rules.py`）做**单次运行**的名单扩展，而刻意**不**写入 `CRITICAL_METRICS` 常量——那是全局默认值，一旦被污染，启发式检测器就握有了静默丢弃数据的权力。
- **CLI**：`--pii-video`、`--pii-strict`。`--no-video` 下人脸扫描强制关闭（该档位已声明"未读视频"，报告不得暗示已检查帧）。
- **可选依赖组** `pii`：`opencv-python-headless>=4.8,<5`（上界有实测依据：4.14.0 自带 19 个 Haar 级联 XML，5.0.0 一个都不带）。
- JSON 报告 issue code `PRV-01`；新增 **Layer 4（安全/隐私）** 与 `LAYER4_SECURITY_PRIVACY`，`ALL_METRICS` 22 → 23。刻意新建层而非并入 L1——`json_report.py` 用 `LAYER1_INTEGRITY` 做完整性分组，混入会改变既有报告的分组语义。

### Fixed
- **报告自身就是泄漏源（本版本最重要）**：JSON 报告此前**原样回显** `task_description`（v0.9.20 任务标识功能引入）和 `dataset.tasks`，而这两处恰是整个数据集中最"有人味"的表面——采集者姓名、手机号、"采集于某住址"都写在里面。结果是：一次审计报出了 3 个标识符，同一份报告在下一层把 3 个标识符原文完整地又印了一遍。现在报告层所有自由文本回显先过 `redact_text()`，命中处替换为 `[REDACTED:<type>]`（任务语义保留：`"pick up the red block"` 照旧可读），并新增布尔字段 `task_description_redacted` 标记该字段被改写。**该函数与指标共用同一张检测器表和同一套校验器**，因此"报告隐藏了什么"与"报告报了什么"不可能分歧——测试 `test_metric_and_report_disagree_on_nothing` 钉住这一点。超出扫描预算（10 万字符）的文本不再回显，而是显式标注截断：扫描止于此，其后即未知，不能当作"已检查且干净"。
- **`--blind` 名不副实**：该模式文档承诺"可外发共享"，却从未处理任何自由文本。检测式脱敏识别不了任意人名和住址，因此 blind 模式不依赖它——所有任务文本替换为稳定的 SHA-256 短哈希桩 `redacted:<8位>`，相同文本仍可跨 episode 归并。其 docstring 此前断言"报告中不存在原始文本"，该断言在 task 文本面前本就不成立，已连同去重一起修正。
- **JSON 序列化崩溃**：`rda/report/json_report.py` 的 `_NumpySafeEncoder` 使用了 `np.` 却从未 `import numpy`，任何携带 numpy 标量/数组的报告在序列化时直接 `NameError`（实测触发源：`temporal_structure` 的 `idle_prefix_frames: int64`）。该缺陷在 HEAD 已存在，与本次新增功能无关——用去掉 `pii_scan` 的对照可以复现。已修复并补回归测试。
- 邮箱正则漏检单级域名（`name@company.com`）：域名尾部锚组要求至少两段点分，已改为 `*`。

### Changed
- `report_schema_version` `1.3` → `1.4`（新增 `task_description_redacted`，且 `task_description` 内容语义变更）。

### Tests
- `tests/test_pii_scan.py`（51 项）：验证器精度 / 检测召回与假阳性对照 / **不泄漏**（原始标识符既不得进入 `measurement`、`details`、`message`，也不得出现在 JSON 报告的任何节点）/ 判定策略（默认不改判定、`--pii-strict` 才 EXCLUDE）／优雅降级（opencv 缺失、opencv 5.x 无级联、`RDA_HAAR_CASCADE` 覆盖、视频文件缺失逐相机上报）／执行档位交互（`--no-video` 下文本扫描仍须执行；真实机械臂视频零人脸误报且必须证明确实解码成功，避免"没读到所以通过"的空过）。
- `tests/test_invariant_guards.py` 的 INV-006 随 schema 版本同步更新至 `1.4`。

### 已知边界（不掩饰）
本指标**不声称**数据集合规、不声称已获知情同意、不声称找全了每一帧的每一张人脸：无 OCR（车牌/门牌/工牌），无语音生物特征，无授权链审查；人脸检测只对采样帧（默认 5 帧/相机）。它产出的是**待人工复核项**，不是隐私合规证明。

---

## 0.9.21 - 2026-10-13

### Fixed
- **P0-1 零内容 episode 静默 PASS**：一个有帧、但 loader 未能解析出任何 `action` / `observation` / 视频特征的 episode，此前绝大多数指标转为 N/A，`classify_episode()` 因设计上跳过非 available 结果而返回 PASS（实测 22 项指标仅 6 项 computable、0 项 finding）。现已新增 `_zero_content_guard`：该场景强制判 EXCLUDE 并给出明确原因。
  守卫按执行档位判定"什么算可审计"：`--no-video` 下视频特征**不**计入（否则对纯视频数据集用 `--no-video` 会审计 0 项却仍报 PASS，而 `--no-video` 正是"视频未下载"场景的推荐规避手段）；`--video-only` 下机器人特征不计入。新增 `has_auditable_content()`（`rda/audit/episode_audit.py`），`EpisodeAuditResult` 新增 `has_auditable_content` 字段。
- **P0-2 缺少「未有效审计」退出码**：此前 CLI 仅在存在 EXCLUDE 时退出 2，数据集 0 episode、或每个 episode 都无可审计内容这两种情况都返回 0，接进 CI 会被当成通过。现新增退出码 **3 = NO_EFFECTIVE_AUDIT**（检查顺序在 exit 2 之前，避免被"数据不好"掩盖"审计没跑"），并在 JSON 报告顶层新增 `audit_effective` 字段。`report_schema_version` 由 `1.2` 升为 `1.3`。修复位置 `rda/cli/main.py`、`rda/report/json_report.py`。

### Removed
- **P0-3 删除零引用死代码**：`rda/audit/rules_current.py`（319 行）与 `rda/metrics/visual_integrity_old.py`（629 行）全库无任何 import。前者 CRITICAL 名单第 7 项引用的是 `video_stream_sync`——该指标在 v0.9 已拆分为 presence / span_consistency / temporal_offset / temporal_drift 四项，名字已不存在；一旦被误引入，「整路相机丢失」闸门会静默失效。

### Changed
- **P0-4 CHANGELOG 版本日期修正**：0.9.18 原记 `2026-10-14`，晚于 0.9.19（10-02）与 0.9.20（10-08），且为未来日期。本仓库无 git 历史可查，暂按版本顺序推定为 `2026-09-30`，**待以实际发布记录核实**。

### Added
- `tests/test_zero_content_guard.py`：20 项测试钉住 P0-1 / P0-2 行为——零内容 episode 必为 EXCLUDE、正常 episode 不受影响、纯视频数据集不被误伤、0 帧守卫保持优先、档位感知（`--no-video` / `--video-only`）、`audit_effective` 与 exit 3。**任一失败即代表静默放行回归已复发。**

---

## 0.9.20 - 2026-10-08

### Added
- **EXCLUDE verdict 原因标注**：审计结果现在会在 EXCLUDE 判定旁标注触发的具体 metric 名称和层级（如 `EXCLUDE (L1: video_stream_presence)`），用户无需再猜测排除原因。新增 `get_exclude_reasons()` 函数（`rda/audit/rules.py`），`EpisodeAuditResult` 新增 `exclude_reasons` 字段，文本报告和 JSON 报告均已同步更新。
- **Action 字段别名兼容**：支持非标准 action 列名，包括 `actions`（复数形式，如 robomme 数据集）、`end_pose`/`start_pos`/`gripper_width`（如 umi_cup_in_the_wild 数据集）。新增 `_ACTION_FIELD_ALIASES`、`_TOP_LEVEL_ACTION_ALIASES` 映射表及 `_is_action_key()`、`_to_action_key()` 辅助函数（`rda/io/lerobot_loader.py`），`_get_feature_keys()`、`_extract_episode_from_dataframe()`、`_read_parquet_columns()`、`_infer_action_unit()`、`_primary_action_array()`（`rda/metrics/motion.py`）及 HF dataset 迭代逻辑均已同步适配。

---

## 0.9.19 - 2026-10-02

### Changed
- **Renamed `physical_plausibility` → `numeric_sanity`**: the metric's name now better reflects its actual purpose — a lightweight numeric range sanity check that catches obviously implausible values (e.g. degree/radian confusion, extreme force spikes) without claiming rigorous physical validation. Remains a Layer 1 WARNING-level metric that does not affect verdict.

---

## 0.9.18 - 2026-09-30

> 日期说明：原记录为 `2026-10-14`，与版本顺序矛盾（0.9.19 = 2026-10-02、0.9.20 = 2026-10-08），
> 且为未来日期。本仓库无 git 历史，真实发布日无法核实，此处按版本顺序推定，
> 待以实际发布记录为准。

### Added
- **Physical plausibility check** (`physical_plausibility`): new Layer 1 WARNING-level metric that checks numeric features for physically implausible values. Detects joint angles exceeding ±π (configurable), force/torque spikes, and velocity anomalies. Always returns PASS (does not affect verdict) but attaches warnings in `details["warnings"]` for diagnostic visibility. Thresholds are configurable via constructor kwargs (`angle_limit`, `force_limit`, `torque_limit`, `velocity_limit`).

### Changed
- **Technical debt cleanup**: all 37 bare `except Exception:` handlers in `rda/` have been replaced with specific exception types or properly bound with `as e` and logging. Video decode operations now catch `(OSError, ValueError)`; parquet reads catch `(OSError, ValueError)`; lerobot API fallbacks catch `(TypeError, ValueError, AttributeError)`; metric computation catch-alls retain `Exception as e` with proper logging.
- **Preflight metric errors** now log at DEBUG level instead of silently swallowing exceptions.

---

## 0.9.17 - 2026-09-28

### Added
- **Frozen episode detection** (`check_frozen_episode`): restores regression from v0.9.7 where frozen episodes (effective_motion_ratio ≈ 0) were no longer caught. New `FROZEN_EPISODE_EMR_THRESHOLD = 0.02` triggers PASS → REVIEW upgrade. Calibrated on ArmnetBench data: blind test frozen episodes (EMR=0) are caught; real data minimum EMR=0.023.
- **Calibration script** (`scripts/calibrate_thresholds.py`): data-driven threshold optimization from ArmnetBench raw data using Youden's J statistic. Computes ROC-AUC, Cliff's delta, and optimal cutpoints for all key metrics.
- **Data-driven severity brackets** for `action_discontinuity` in `compute_behavior_severity()`: thresholds updated from arbitrary (20/50/100) to ArmnetBench-calibrated (16/27/37) based on success/failure distribution analysis (ROC-AUC=0.86, N=2499).

### Fixed
- **Frozen episode regression**: idle_ratio was moved to DIAGNOSTIC_METRICS in v0.9, losing its verdict influence. Frozen episodes (0% motion) in blind tests were no longer caught. Fixed via dedicated `check_frozen_episode()` in the verdict pipeline, always-on.
- Provenance docs updated for action_discontinuity and idle_ratio with calibration data sources and threshold rationale.

### Changed
- `compute_behavior_severity()` action_discontinuity brackets: 20→16, 50→27, 100→37 (ArmnetBench Youden optimal + distribution quantiles).

---

## 0.9.16 - 2026-09-24

### Added
- **video_freeze 阈值溯源输出**（D-25 / T-16）：当 `motion_source` 为 `state` 或 `auto`（且 state 可用）时，`details` 中新增 `state_motion_median`、`state_motion_mad`、`state_motion_threshold` 三个字段，便于外部复现阈值判定逻辑。
- 新增 2 个单元测试覆盖阈值输出的存在性与条件排除。

---

## 0.9.15 - 2026-09-21

### Fixed
- **DEV-01 (P0)**: `dataset_audit.py` fixed `ep_result.results` → `ep_result.metrics.values()`, dataset_summary now correctly populated.
- **DEV-02 (P1)**: `lerobot_loader.py` added top-level `info["joint_limits"]` extraction with per-feature fallback.
- **DEV-03 (P1)**: Implemented profile system (strict/default/lenient) for `video_freeze`, `video_timestamp_alignment`, `video_frame_integrity` — 6 configurable parameters now realized.
- **DEV-04 (P2)**: `_TS_SOFT_TOLERANCE` corrected from 0.02 to 0.05, aligned with spec.
- **DEV-05 (P2)**: `missing_dropout` now returns `make_na(reason="empty_episode")` for empty episodes instead of PASS.
- **DEV-06 (P2)**: `invalid_values` now returns `make_na(reason="no_numeric_features")` when no numeric features exist.
- **DEV-07 (P2)**: `timestamp_validity` now returns `make_na(reason="insufficient_timestamps")` when < 2 timestamps.
- **DEV-08 (P3)**: `joint_limit` EXCLUDE path measurement output enriched via profile implementation.
- **DEV-09 (P3)**: Removed unreachable `state_array_invalid_shape` dead code in `motion.py`.
- **DEV-10 (P3)**: Spec §2 metric count documentation synchronized.
- **DEV-11 (P2)**: `visual_quality.py` added v2.1 video path fallback (`videos/chunk-XXX/<feature>/episode_XXXXXX.mp4`).
- **DEV-12 (P3)**: `video_frame_integrity` soft-level mismatches now return REVIEW instead of EXCLUDE.
- **DEV-13 (P2)**: `json_report.py` added `_NumpySafeEncoder` for proper numpy type serialization.
- **DEV-14 (P2)**: `lerobot_loader.py` v2.1 fps injection moved outside video-conditional block.

### Changed
- Governance docs updated: DECISIONS.md D-24 marked all fixed; 核心参数规范 version bumped to 0.9.15 with deviation annotations resolved.
- Report/UI layer verified: no breaking changes needed — N/A verdict handling, verifiability, and badge rendering all compatible.

---

## 0.9.14 - 2026-09-20

### Added
- **Action unit inference** (`dataset_summary.action_unit`): loader now
  infers whether action values are `degrees`, `radians`, `normalized`,
  `raw_steps`, `mixed`, or `unknown` by combining info.json explicit
  fields, feature name suffixes (`.pos`/`.rad`/`.norm`/`.step`) and
  value-range heuristics. Label is surfaced in JSON report and text
  summary.
- **Per-joint spike breakdown at top-level measurement**:
  `action_discontinuity` now exposes `by_joint` in `measurement`
  (sorted by `spike_count` descending), plus a new `top_k_joints`
  constructor parameter (default `0` = all). `details.by_joint` is
  preserved for backward compatibility.
- **video_freeze state cross-validation** (`--freeze-motion-source`):
  new CLI flag with modes `action` (default, pure vision, unchanged),
  `state` (cross-validate against `observation.state` motion signal
  using median + 3×MAD threshold), and `auto` (use state when
  available, fall back to action otherwise). Reports
  `freeze_motion_source` and `state_cross_validated_segments` for
  transparent decision paths. EXCLUDE downgrades to REVIEW only when
  state confirms whole-machine stall.
- **INV-011 scale invariance guard test**: new
  `test_inv011_scale_invariance` verifies that action/state linear
  transforms (degrees ↔ radians ↔ raw_steps ↔ normalized) leave
  `spike_count`, `median_idle_ratio` and per-joint `spike_count`
  invariant.

## 0.9.13 - 2026-09-18

### Fixed
- **LeRobot v2.1 video format support**: RDA now correctly locates and audits
  video files in v2.1 datasets where the directory structure is
  `videos/chunk-XXX/<feature>/episode_XXXXXX.mp4` (previously only supported
  v3.0 layout `videos/<feature>/chunk-XXX/file-XXX.mp4`).
- Added `_extract_video_features_for_v21()` helper to reconstruct video feature
  metadata from info.json for v2.1 datasets.
- Added `dataset_root` to v2.1 episode metadata so video-dependent metrics
  (stream presence, frame integrity, freeze detection, timestamp alignment)
  can resolve file paths.
- Video frame integrity, freeze detection, and timestamp alignment metrics
  now fall back to v2.1 path layout when v3.0 paths are not found.

## 0.9.12 - 2026-09-16

### Fixed
- Ship `video_integrity.py` and `visual_quality.py` in the published
  package (present in the repository since 0.9.9 but omitted from the
  0.9.11 build artifact).

## 0.9.11 - 2026-09-16

### Added
- **Timestamp gap detection** (`timestamp_validity`): flag inter-frame
  intervals exceeding `2.5 × median dt` and emit `gap_count`,
  `gap_positions` and `gap_total_duration` (ms). The 2.5x multiplier is
  an internal constant, not user-facing. Gap information is reported but
  does not change the verdict by default.
- **Missing-frame Path B** (`missing_dropout`): when `frame_index` is
  unavailable (e.g. Parquet), infer dropped frames from timestamp gaps
  (`round(dt / median_dt) - 1` per gap). Outputs
  `missing_detection_method: "timestamp_gap_inference"`, `gap_count`,
  `gap_positions` and `gap_total_duration`. Frame-index detection, when
  available, is labelled `"frame_index"`.
- **Gap-aware robust jitter** (`sampling_jitter`): report
  `cv_raw` over all intervals plus a robust CV that excludes gap
  intervals (`method: "robust_cv_excluding_gaps"`, `gaps_excluded`).
- Hierarchical LeRobot v3.0+ `state.*` handling for joint limits and
  schema checks, including the G1_WBT observation naming convention.

### Notes
- 0.9.10 was not released (version number skipped during local builds).

## 0.9.9 - 2026-09-15

### Changed
- Rewrite README to be concise and results-driven (recommendations
  section removed).
- Loader passes `joint_limits` and declared features through to episodes;
  `schema_consistency` cross-validates actual vs declared dimensions.
- Sync API client User-Agent to the package version.

## 0.9.8 - 2026-09-15

### Fixed
- Support hierarchical LeRobot v3.0+ `state.*` observation keys (e.g.
  `state.ee_state`, `state.hand_state`, `state.robot_q_current`) in
  `VelocityMetric`, `CoverageMetric` and `JointLimitMetric`. Previously
  these metrics only looked for a flat `"state"` key and reported N/A
  for datasets using the hierarchical naming convention (such as the
  Unitree G1_WBT series).
- Added `_resolve_state_array()` helper: first tries flat `"state"`
  (backward-compatible), then searches `state.*` keys, preferring
  known comprehensive sub-states (`robot_q_current` > `qpos` >
  `joint_pos`), then highest-dimension 2-D array.
- Improved `_primary_action_array()` fallback: when no preferred key
  (`joint_pos`, `position`, `action`) matches, now selects the
  highest-dimension 2-D float array instead of the first one found,
  providing better coverage for multi-action datasets.

## 0.9.7 - 2026-09-14

### Added
- D-17 video audit tiered execution model
- UI execution tier selector in Streamlit web UI

### Fixed
- Multiple fixes for entry point errors and UI function signature
  issues across v0.9.5-v0.9.7 patch cycle

## 0.9.1 - 2026-09-10

Patch release fixing two latent bugs surfaced by the new golden
regression set, plus the golden set itself.

### Fixed
- `rda.report.json_report`: `_verifiability_for_metric` referenced
  `MetricAvailability.NA`, an enum member that does not exist
  (the enum only defines `AVAILABLE` / `NOT_AVAILABLE` /
  `ERROR`). Any available metric crashed report generation with
  `AttributeError`. N/A is now derived from the not-available
  reason (`single_camera`, `no_video_features`, `dep_*`); other
  unavailable metrics report "Not verifiable".
- `rda.metrics.temporal`: `SamplingJitterMetric` pass path omitted
  `jitter_ratio` from its measurement, so
  `compute_behavior_severity` never saw it and the sampling-jitter
  behavior diagnostic could never fire. `jitter_ratio` (= CV) is
  now emitted on every path.

### Added — tests
- `tests/golden/`: deterministic golden dataset — 7 non-video and 4
  PyAV-synthesized video episodes with pinned verdicts and
  must-trigger metrics, covering all four audit layers (Integrity
  Gate / Trajectory Diagnostics / Dataset Profile / Dataset
  Summary).
- `tests/test_golden_regression.py`: end-to-end regression guards —
  verdict per scenario, must-trigger metrics, diagnostics never
  escalate to hard excludes, N/A behaviour without video,
  verifiability fields, dataset aggregation, and a pinned verdict
  distribution.

## 0.9.0 - 2026-09-10

Four-layer audit architecture: Integrity Gate → Trajectory Diagnostics →
Dataset Profile → Dataset Summary. The audit verdict is now driven
exclusively by hard integrity checks; diagnostic metrics report
measurements and findings without flipping episode verdicts. See
`RDA_v0.9_核心参数规范.md` / `RDA_v0.9_迁移计划.md` for the full design.

### Changed — rules layer (Phase 1)
- `REVIEW_METRICS` renamed to `DIAGNOSTIC_METRICS` (backward-compatible
  alias `REVIEW_METRICS` kept). Diagnostic findings no longer trigger a
  REVIEW verdict — they produce measurements + findings only.
- `upgrade_verdict_by_behavior` is now opt-in (`enabled=False` by
  default).
- `compute_behavior_severity`: added `sampling_jitter` branch
  (jitter_ratio > 0.3 → +30 severity, > 0.1 → +10); fixed the
  `distribution` → `coverage` field mapping (`occupancy_rate`).
- `video_frame_integrity` wired into `CRITICAL_METRICS` (fixes the v0.8
  wiring gap; a failed frame-integrity check now hard-excludes).

### Changed — video stream metrics split (Phase 2)
- `video_stream_sync` split into four independent metrics:
  - `video_stream_presence` (L1, hard check — missing camera stream →
    EXCLUDE),
  - `video_stream_span_consistency` (L2 diagnostic),
  - `video_stream_temporal_offset` (L2 measurement, pairwise frame-level
    offset; "Not verifiable" without frame timestamps),
  - `video_stream_temporal_drift` (L2 measurement, clock drift rate).
  Old `VideoStreamSyncMetric` names are kept as compatibility aliases.

### Changed — naming and layer placement (Phase 3)
- `temporal_sufficiency` renamed to `temporal_structure` (compatibility
  alias kept). It now belongs to the Dataset Profile (L3) layer.
- L3 metrics (`distribution`, `coverage`, `temporal_structure`) no
  longer participate in episode verdicts; their measurements feed the
  Dataset Profile aggregation only.

### Added — Dataset Summary layer (Phase 4)
- `rda/report/dataset_summary.py` (new): cross-episode aggregation of
  idle_ratio (median/P10/P90), sensor sync (median/p90/max of worst p95
  offset) and visual quality (median blur variance + exposure anomaly
  ratio). Attached to `DatasetAuditResult.dataset_summary` and emitted
  in both text and JSON reports.

### Added — verifiability levels + report format (Phase 5)
- Every metric result now carries a `verifiability` field:
  **Verified** (hard-checked), **Measured** (observational
  measurement), **Not verifiable** (missing reference/timestamps),
  **N/A** (modality absent).
- JSON report: `report_schema_version` bumped to 1.1; new top-level
  `dataset_summary` section.
- Text report restructured into Integrity Gate / Trajectory
  Diagnostics / Dataset Profile / Dataset Summary sections with a
  dedicated Video Temporal Verification block.

### Added — recommend audit_signals (Phase 6)
- Recommend contract bumped to v4: clients may send `audit_signals`
  (smoothness_summary / calibration_summary / coverage_summary)
  extracted from the audit pipeline, enabling SMOOTHING_REVIEW /
  CALIBRATION_CHECK / COVERAGE_SUGGESTION server-side rules. Older
  servers ignore the unknown key; older clients are unaffected.
- CLI `rda recommend --audit-report <rda_report.json>`: extracts
  diagnostic signals from a pre-computed audit report and sends them
  with the recommendation request.
- Cache key namespaced `v4-` (includes audit signals).

### Versioning
- Package version bumped 0.8.0 → 0.9.0 (`__init__.py`, `pyproject.toml`,
  User-Agent all aligned; enforced by `test_version_consistency.py`).

## 0.8.0 - 2026-09-06

### Added (REQ-5 — dataset-level relative baselines + acceptance summary)
- `rda/report/acceptance.py` (new): dataset-level relative baselines.
  - `tukey_fence` / `is_tukey_outlier` — episode-runtime outlier detection
    using the Tukey fence `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, ported from the
    [SLE] reference implementation (`score_lerobot_episodes`
    `is_time_outlier(mode="iqr")`). No threshold is invented here.
  - `compute_percentile_baselines` — P10 / P50 / P90 (+ min / max) for the
    Tier-1 portable metric set (`duration_sec`, `spike_count`,
    `effective_motion_ratio`; rho = 0.960 vs the full metric set, see
    `rda/calibration/portable.py`). Baselines are relative to the audited
    population, not an absolute threshold borrowed from another platform.
  - `build_acceptance_summary` — folds verdict distribution, percentile
    baselines, runtime outliers, the calibration layer and the
    not-checked inventory into one block.
- `acceptance_summary` is emitted by **both** report formats: the engine
  (CLI `--format json`) and the product/dataset report. `schema_version`
  `1.0`.
- UI: new page **8 · Acceptance** (`rda/ui_app/pages/8_Acceptance.py`) —
  dataset identity, verdict distribution, quality layer, baseline table,
  outlier table, not-checked inventory, calibration layer, and an export +
  disclaimer section. Registered in `app.py` navigation.
- i18n: 48 new bilingual (zh/en) `acc_*` keys plus `nav_acceptance`.
- `skipped_by_missing_dep` is now a public function in `json_report`
  (the private `_skipped_by_missing_dep` name is kept as an alias so
  existing callers and tests keep working).

### Red line honoured
REQ-5 measures and presents; it does **not** adjudicate. The acceptance
summary contains no accept/reject decision — that call belongs to the
human reviewer. `test_acceptance_summary_does_not_adjudicate` pins this.

### Degenerate-fence guard
When every episode has the same duration, `IQR = 0` and the Tukey fence
collapses to a single point. Reporting "0 outliers" there would read as
"checked and clean" when the truth is "no spread to detect against" — the
same class of trap as REQ-11's "not checked must not read as pass". The
block now sets `degenerate: true` with an explicit note, and the UI shows
a warning. Verified on aloha_insertion (50 episodes, all 9.98 s).

### Fixed
- **Test infrastructure silently faking green (pre-existing).**
  `tests/test_i18n.py` and `tests/test_ui_visual_audit.py` injected a stub
  `streamlit` module into `sys.modules` unconditionally, which broke
  `streamlit.testing.v1` imports for the rest of the session — every
  `AppTest` case was skipped instead of run. The stub now applies only
  when the real streamlit is genuinely unavailable.
- `tests/test_ui_visual_audit.py` never imported `DatasetAuditor`
  (a `NameError` swallowed by a bare `except:`). Combined with the above,
  **the REQ-4 UI page tests had never actually executed**. Now imported;
  both cases run and pass.
- `AppTest.from_file` does not put `rda/ui_app` on `sys.path` (unlike
  `streamlit run`, where `app.py` handles it), so pages importing
  `components.common` raised `ModuleNotFoundError`. Tests now insert the
  directory explicitly. Diagnostics were also added to the previously
  silent `except:` so skipped-for-missing-data is visible rather than
  silent.
- Version strings had drifted apart for two releases: `rda.__version__`
  and the `rda-cli/<ver>` User-Agent both said 0.7.2 while
  `pyproject.toml` said 0.7.3. All three are now 0.8.0, and
  `tests/test_version_consistency.py` pins them together.

### Tests
- `tests/test_req5_acceptance_summary.py` (new, 14 cases): Tukey fence
  arithmetic against hand-computed values, degenerate-fence branch,
  unavailable metrics never counted as 0, outlier ordering/truncation,
  no-adjudication guarantee, i18n key parity, and a Streamlit `AppTest`
  end-to-end render of the new page.
- `tests/test_version_consistency.py` (new, 2 cases).
- Suite: 154 passed, 2 skipped (the 2 skips need a real LIBERO v3.0
  dataset that is not present on this machine).

### Measured on real datasets
- libero_10 (379 episodes): 36 runtime outliers, fence `[16.5 s, 35.7 s]`,
  p50 = 25.8 s, max = 50.4 s — all on the long side.
- aloha_insertion (50 episodes): all durations 9.98 s → degenerate branch,
  correctly flagged rather than reported as "0 outliers".

## 0.7.3 - 2026-09-05

### Added (v0.7.x UI catch-up + REQ-3②)
- UI: Visual Audit panel in Episode Explorer detail — VA-A integrity
  trio status lines with freeze-region table (per-stream span,
  duration, moving ratio) and VA-B quality breakdown (per-camera
  penalty table + per-sample quality curves with clipped-frac on a
  secondary axis). `video_deps_missing` renders a loud "not audited ≠
  pass" warning and hides the quality tables instead of faking them.
- UI: Visual Integrity Overview section in Health Overview —
  dataset-level checked / flagged / not-checked episode counts with a
  per-metric findings list and a page link into the review queue.
- i18n: 45 new bilingual (zh/en) keys covering the visual panels,
  visual recommendation texts (`rec_video_freeze`,
  `rec_video_timestamp_alignment`, `rec_video_stream_sync`,
  `rec_visual_quality`) and the health visual overview.
- REQ-3②: fps-aware usable-run floor — `usable_retention_ratio` now
  uses `max(16, 1s-in-frames)` as the minimum run length when the
  episode's fps is known (from dataset info.json via the loader), and
  exactly 16 otherwise (byte-identical to 0.6.x at ≤16 Hz). Rationale:
  DROID's `min_non_idle_len=16` encodes "≈1 s at 15-30 Hz"; at 50 Hz
  16 frames is 0.32 s, letting sub-second twitches count as usable
  (measured on aloha_insertion @50 Hz: usable retention 24.6% → 0.0%
  on a genuine static episode). New field `usable_run_floor_frames`
  (per episode and dataset median) reports which floor produced the
  ratio; server payload gains the field additively (v0.6.1+ parsers
  ignore unknown keys).
- Tests: `tests/test_ui_visual_audit.py` (stats layer + Streamlit
  AppTest end-to-end runs of both pages, incl. the dep-missing
  never-pass guarantee) and `tests/test_req3b_fps_floor.py` (floor
  semantics + fallbacks). Suite: 136 passed, 4 skipped.

## 0.7.2 - 2026-09-05

### Fixed
- `skipped_by_missing_dep` crashed on real `DatasetAuditResult` shapes
  (`EpisodeAuditResult` exposes `metrics`, not `metric_results`) — the
  0.7.1 wheel shipped before this fix was caught in the post-upload
  smoke test. 0.7.2 is the first wheel containing the corrected
  attribute access. All 0.7.1 features otherwise unchanged.

## 0.7.1 - 2026-09-05

### Added (REQ-10 — visual detection calibration closure)
- `benchmarks/visual_inject.py`: corruption-injection benchmark — five
  visual corruption families (blur gradient, dark, blown-out, freeze
  spans, static camera) re-encoded onto real episodes; produces
  per-scenario detection tables. Calibration run (libero_10 × 5
  episodes × 11 scenarios): all acceptance checks passed, including
  both boundary cases (majority-frozen episode and fully static camera
  — the adaptive epsilon does not desensitize).
- `visual_quality`: highlight-clipping detection (`clipped_frac`,
  fraction of pixels ≥ 250; ≥ 20% → blown-out). Calibration finding:
  mean luminance alone misses saturation on dark scenes (×3 gain clips
  22–33% of pixels while mean stays ≈ 180, under the 230 band).
- Multi-dataset threshold check (libero_10 / jaco_play / xarm_lift):
  penalty p50 = 0.00 everywhere; jaco_play shows expected REVIEW
  signals on genuinely blurred episodes (blur_var p10 = 61.5 vs xarm
  464.6). Thresholds hold across resolutions.
- Decode performance numbers (README): freeze ≈ 0.11 s/episode,
  quality ≈ 0.17 s/episode (worst ≈ 0.17 s) on 10k-frame 224×224
  chunk files.

### Added (REQ-11 — optional-dependency UX)
- Visual metrics grade NA with reason `video_deps_missing` when PyAV is
  absent ("not audited", never "pass").
- JSON report: top-level `skipped_by_missing_dep` counts (e.g.
  `{"av": 12}`); CLI prints a warning with the install hint.
- `[video]` extra (`pip install robot-data-audit[video]`); README
  dependency-tier table.

### Added (UI catch-up)
- `7_Recommend.py`: policy_chunk_size input (CLI had it since 0.6.0)
  and icons for the visual / advice actions.

### Tests
- 9 new tests (REQ-10 detection anchors + REQ-11 semantics);
  122 passed, 2 skipped, zero regressions.

## 0.7.0 - 2026-09-05

### Added

- **REQ-4: visual-stream audit — VA-A integrity trio + VA-B quality measurement.** RDA now audits the visual modality with the same hard-evidence discipline as the kinematics:
  - `video_freeze` (EXCLUDE-grade): detects camera drop-out — consecutive codec-identical video frames while the arm is moving. Decodes 64×64 grayscale spans via PyAV; frozen detection uses an adaptive epsilon (`max(0.10, 0.25 × p10` of frame diffs`)`) calibrated on libero_10 so slow motion is not misclassified as a stall. A single brief freeze is REVIEW; conclusive corruption requires repeated spans, a ≥3 s stall, or ≥30% frozen time.
  - `video_timestamp_alignment` (EXCLUDE beyond 10% drift, REVIEW 2–10%): compares the chunk-video time span against the parquet timeline, catching systematic video/state misalignment that frame-count checks miss.
  - `video_stream_sync` (EXCLUDE): all referenced camera streams must resolve (a missing wrist camera is silent modality loss) and multi-camera spans must agree.
  - `visual_quality` (REVIEW-grade, never a veto — VA-B): blur (Laplacian variance), exposure (dark/blown-out bands), and contrast (P5–P95) on 10 uniformly sampled frames; aggregates as `max(blur, exposure)` per SLE, reports the worst-frame timestamp for direct human review. Thresholds are SLE-derived priors, calibrated on healthy data (penalty 0.00 on libero_10) — calibrate per camera before treating them as gates.
- **Server engine v5 (rules 0.8.0)**: new `VISUAL_REPAIR_FIRST` (HIGH) and `VISUAL_QUALITY_REVIEW` (EXPERIMENTAL) recommendations driven by an extended `audit_signals.visual_summary` (whitelisted, bounded). v1–v3 clients and payloads behave exactly as before; new actions degrade to `NO_RECOMMENDATION` on old clients via `from_dict` (safe).
- Metric provenance extended: all four new metrics ship the four-file record (`docs/provenance/<metric>/`), 18 metrics × 4 files.

### Fixed

- `VisualQualityMetric` and alignment metric now read `episode.timestamps` (the canonical field name) — verified against the schema dataclass.

## 0.6.0 - 2026-09-05

### Added

- **REQ-3: DROID-aligned idle rules.** Per-episode temporal metrics now include `usable_retention_ratio` (fraction of frames inside consecutive non-idle runs of ≥16 frames — openpi's DROID `min_non_idle_len=16` ≈ 1s at typical 15–30Hz) and `max_idle_run_frames` (longest static stretch). New `rda recommend --policy-chunk-size N` flag (e.g. 100 for ACT, 16 for Diffusion Policy) sends `policy_chunk_size` under `contract_version: 3`; the server then evaluates window integrity AT the chunk length instead of the legacy fixed 5/10/20 tiers, and may emit a chunk-window-collapse `DO_NOT_PRUNE` warning plus an EXPERIMENTAL tail-trim suggestion (DROID `filter_last_n_in_ranges=10` reference). Near-static datasets (median idle >95% AND usable retention <5%) now receive a single `DISCARD_STATIC` recommendation, mirroring openpi/DROID dropping near-static demonstrations — human review required, never automated. Offline fallback implements the same DISCARD_STATIC branch conservatively (dataset-level signal only). New enum actions degrade to `NO_RECOMMENDATION` on older clients via `from_dict` (safe).
- **REQ-2: advisory recommendation types.** New `SMOOTHING_REVIEW` (action spikes beyond 5×MAD of the episode's own motion — teleop glitches, sensor noise, or post-hoc smoothing; human review, never auto-deletion), `CALIBRATION_CHECK` (sensor sync p95 > 2× the dataset's own baseline — calibration drift detection without absolute hardware thresholds), and `COVERAGE_SUGGESTION` (episodes below the dataset's P5 state-space occupancy — collection guidance for data producers, not a deletion hint). All three are driven by optional `audit_signals` in the v3 payload (whitelisted keys, bounded size), advisory-only, and rendered with an explicit "human-review signal" marker in the CLI output.
- **REQ-8: metric provenance.** `docs/provenance/<metric>/` ships a four-file record (algorithm.md, source.md, implementation_origin.md, license.md) for all 14 metrics, with a README index — making the IP boundary explicit: public precedents inform ideas only, all implementation is original.
- Server engine v4 (rules 0.7.0): chunk-aligned window evaluation, DISCARD_STATIC, tail-trim, and the three REQ-2 advisory rules. v1/v2 requests behave exactly as before (all new parameters optional, unknown payload fields tolerated).

### Fixed

- Audit JSON report top-level `version: "0.2.0"` was a hardcoded report-schema version that read like a tool version next to `tool_version`. Renamed to `report_schema_version: "1.0"`; no consumer ever read the old key (verified repo-wide).
- `rda audit` on a LeRobot v3.0 dataset without pandas/pyarrow installed suggested `pip install lerobot` — wrong for the direct-parquet path, which needs only pandas/pyarrow. The loader now raises a precise message naming the actual missing packages; the CLI only suggests lerobot when the issue is really about lerobot.
- CLI output now reports the new DROID-aligned metrics (`usable_retention`, `max_idle_run`) in the temporal-sufficiency overview.

## 0.5.9 - 2026-09-05

### Added

- **REQ-1: verdict-gated recommendations.** Audit verdicts now enter the recommendation chain: `rda recommend` re-evaluates the five deterministic CRITICAL metrics (missing/dropout, NaN/Inf, schema, timestamp validity, joint limits) in the same single pass as the temporal metrics (zero extra dataset traversal, near-zero cost), and any episode failure suppresses all pruning advice. The dataset receives a single `REPAIR_FIRST` recommendation instead, citing per-episode evidence (episode index, failed metrics) and a damage class: `INVALID` (NaN/Inf, broken time axis — irrecoverable at audit time) vs `REPAIRABLE` (frame loss, schema mismatch, joint excursion — fixable by trimming/re-export). This holds on ALL result paths: server response, cache, and offline fallback (client-side final gate), so the guarantee is independent of server upgrade order. New module `rda/recommend/preflight.py`; API payload carries `contract_version: 2` + `verdict_summary` (aggregates only, <1KB — privacy posture unchanged); server rules v0.6.0/engine v3 grade the same gate; v1 clients and servers are fully unaffected.
- `RecommendationResult.verdict_summary` field: JSON reports now embed the verdict evidence (`exclude_count`, `dominant_reason_code`, `excluded_episodes[]` with `reason_code`/`failed_metrics`) next to the REPAIR_FIRST recommendation.
- Regression coverage: `tests/test_preflight.py` (17 tests) — per-episode verdict classification (NaN → INVALID, dropout → REPAIRABLE, 0-frame guard), aggregation with INVALID-dominates-REPAIRABLE, payload size budget (<1KB for 200+ episodes), the gate itself (TRIM blocked on excluded datasets, passthrough on healthy, zh/en copy), single-pass integration, and cache-key namespacing.

### Fixed

- Windows cache-write failure: the v2 cache-key namespace originally used a colon (`v2:<hash>`), which is a reserved character on Windows — cache writes silently produced 0-byte files. The namespace is now `v2-<hash>`; old `v2:`-less (v1) cache entries simply age out via TTL.

## 0.5.8 - 2026-09-03

### Fixed

- `video_frame_integrity` false-flagged every episode of a chunked v3.0 dataset: it compared a shared MP4's *total* frame count against a single episode's parquet count (e.g. 101469 vs 214), producing a 100% hard-mismatch on `lerobot/libero_10`. The metric now derives each episode's own frame span inside the chunk via `(to_timestamp - from_timestamp) * fps` (timestamps are preserved as floats and `fps` is injected by the loader), and separately detects genuine file truncation (when the span runs past the end of the MP4).
- Episode verdicts carried no task identity: `EpisodeAuditResult` now surfaces `task_index` and `task_description` (propagated from `EpisodeData.meta`), so per-episode results in reports answer "which task is this episode from?" — completing the report layer of the v3.0 task-identity fix started in 0.5.7.

### Added

- Regression coverage for chunked-video frame-count cross-check and file-truncation detection (`tests/test_video_frame_integrity.py`).
- Regression coverage for episode-level task-identity propagation to the audit result (`tests/test_lerobot_task_identity.py`).

## 0.5.7 - 2026-09-03

### Fixed

- LeRobot v3.0 task identity was silently dropped for multi-task datasets: the per-frame `task_index` column was filtered out by the audit projection and `meta/tasks.parquet` was never read, so reports could not answer "which task is this episode from?" (the same failure mode reported for the community tool Calibra on `lerobot/libero_10`). `DatasetInfo.meta` now surfaces `num_tasks` and the `task_index -> description` mapping, and `EpisodeData.meta` now carries `task_index` and `task_description`. Single-task and task-less datasets remain fully backward compatible.
- `video_frame_integrity` no longer decodes every frame of videos lacking the `nb_frames` metadata key: it now reads the exact count from PyAV's `stream.frames` (FFmpeg's MP4/MOV `stts` box) and memoizes the result per file, so chunked videos shared across many episodes are parsed once. This cut the metric's cost from >1h to milliseconds on `lerobot/libero_10`.

### Added

- Regression coverage for v3.0 task identity: `tests/test_lerobot_task_identity.py` (synthetic fixtures + a real `lerobot/libero_10` end-to-end check).
- Regression coverage for the video frame-count fast paths: `tests/test_video_frame_integrity.py` (asserts the decode fallback is only reached when neither `nb_frames` metadata nor `stream.frames` is available).

## 0.5.5 - 2026-08-31

### Added

- `rda recommend --offline`: conservative local fallback when the recommendation API is unreachable and no cache exists. Previously this scenario exited with an error even though all required measurements were already computed locally.
- Offline mode only emits the "clearly safe" rule set (`DO_NOT_PRUNE`, and `TRIM_INITIAL` for long initial idle prefixes under frame-wise temporal policy); it never invents risky pruning advice. Output is clearly labeled as offline conservative mode with rule version `offline-fallback`.
- Offline recommendation results are cached like online ones, so a later successful API call can reuse them.
- Test suite for offline fallback behavior and for `tools/rda_render.py` (badge color semantics, HTML report rendering).

### Fixed

- `recommend` no longer fails hard when network is unavailable: it now degrades to local fallback with a stderr warning instead of `exit 1`.

## 0.5.4 - 2026-08-21

### Added

- Evidence-aware report fields for `HARD_FAIL`, `RISK_SIGNAL`, and `UNVERIFIABLE`.
- Episode-level evidence summaries in JSON reports.
- Blind-validation helpers for annotator-safe sample manifests and external QC comparison.
- Regression coverage for evidence boundaries and blind QC comparison.

### Changed

- CLI, Markdown, and UI copy now distinguishes confirmed structural failures from statistical risk signals.
- Risk signals such as action discontinuity, low effective motion, and unusual distributions are described as observations requiring human review, not confirmed corruption.
- Unverifiable metrics are not presented as PASS.
- Recommendation UI copy clarifies that recommendation evidence and audit-risk evidence are separate evidence chains.

### Validation boundary

The release does not claim independent precision, recall, threshold optimality, or training/rollout improvement without human labels, customer QC, or held-out outcome data.

## 0.5.3 - 2026-08-20

- Fixed LeRobot v3.0 metadata/data-file episode mapping fallback.
- Re-ran 12 real datasets with corrected LIBERO handling.
