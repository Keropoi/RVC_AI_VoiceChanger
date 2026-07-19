# RVC 自动训练、质量检查与测试系统

日文目标音色转中文对白的数据归档、固定测试集与复核命令见 [数据流程](docs/DATA_WORKFLOW_ZH.md)。多人对白的自动聚类、目标声纹排序和人工指定步骤见 [多说话人分拣](docs/SPEAKER_SORTING_ZH.md)。

这是一个面向 Windows、NVIDIA GPU 和官方 RVC 仓库的独立自动化控制层。它不复制、修改或重新实现 RVC 模型，而是负责训练音频发现与质检、可恢复的阶段编排、RVC 子进程调用、GPU 监控、测试推理，以及离线 HTML 试听报告。

> 仅可使用你拥有合法权利或已获得明确授权的声音素材。生成的模型是 AI 音色转换模型，不是角色、演员或版权方的官方配音。

## 当前边界

- 本项目提供 CLI、YAML 配置、HTML 报告和 Windows BAT 启动脚本，不包含 GUI。
- 官方 RVC 必须作为独立仓库放在 `external/RVC/`，并使用自己的虚拟环境。
- RVC 不同分支的脚本布局和参数可能不同。本项目在运行时检查实际仓库；找不到兼容入口时会明确失败，不会猜测命令或修改 RVC。
- 本项目不会静默下载模型。仅当配置明确允许、来源和 SHA-256 均已配置时才应添加下载流程。
- 没有真实 RVC checkout、预训练资产和授权音频时，只能完成控制层测试与 `--dry-run`，不能宣称真实训练完成。

## 系统要求

- Windows 10 或 Windows 11
- Python 3.9 或更高版本（推荐 3.10/3.11）
- NVIDIA GPU 与可用驱动；示例配置针对 RTX 3090 24 GB
- FFmpeg 和 FFprobe 已安装并加入 `PATH`
- Git（用于识别 RVC commit）
- 官方 RVC checkout 及其所需模型

先在 PowerShell 检查外部工具：

```powershell
python --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

## 1. 创建自动化控制层虚拟环境

从项目根目录执行：

```bat
scripts\setup_venv.bat
```

等价的 PowerShell 命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

后续命令均可使用 `.venv\Scripts\python.exe`，无需激活环境。

## 2. 准备独立的 RVC 环境

将你要使用的官方 RVC 版本放到：

```text
external/RVC/
```

按照该 RVC 版本自己的说明，在 `external/RVC/.venv` 安装依赖。不要把 RVC 的 PyTorch/CUDA 依赖强行安装进本项目的 `.venv`。配置中的两个解释器路径分别是：

```yaml
paths:
  orchestration_python: .venv/Scripts/python.exe
  rvc_python: external/RVC/.venv/Scripts/python.exe
```

把该版本要求的 HuBERT/ContentVec、RMVPE 和 RVC v2 预训练底模放到 RVC 文档指定的位置。本项目的 `doctor` 会报告缺失项，不会从未知来源下载。

## 3. 准备数据

训练音频放入 `data/training_audio/`，可使用子目录。支持：

```text
.wav .flac .mp3 .m4a .aac .ogg
```

测试音频放入 `data/test_audio/`，建议 3～5 条。训练集和测试集会计算 SHA-256；同一音频不能同时出现在两个集合中。

需要逐条指定推理参数时：

```powershell
Copy-Item data\test_audio\test_manifest.example.yaml data\test_audio\test_manifest.yaml
```

再编辑 `test_manifest.yaml`。

原始训练和测试音频不会被删除。质检失败的训练文件默认只会复制到 `data/rejected_audio/<run_id>/`。

## 4. 配置

`config/default.yaml` 包含完整默认值。建议复制示例后编辑：

```powershell
Copy-Item config\example_windows_3090.yaml config\my_voice.yaml
```

重点确认：

- `paths.*`：音频、运行目录、RVC 仓库和两个 Python 解释器
- `model.*`：模型名、RVC 版本、采样率、F0 方法
- `quality.*`：时长、LUFS、削波、DC offset、静音和 SNR 阈值
- `preprocessing.*`：单声道、重采样、轻度裁切与响度目标
- `training.*`：epoch、checkpoint、batch size 和 OOM 降级候选值
- `testing.*`：测试数量、音高、索引比例和可选参数扫描

相对路径以项目根目录解析；Windows 中文、日文、空格路径均通过 `pathlib.Path` 处理。

## 5. 初始化与环境检查

初始化缺失目录和提示文件（不会覆盖已有文件）：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer init
```

执行环境检查：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer doctor --config config\my_voice.yaml
```

或：

```bat
scripts\doctor.bat
```

报告包括 Python、Windows、FFmpeg/FFprobe、CUDA、GPU/显存、驱动、磁盘、RVC commit、解释器、已探测脚本、模型资产和输入目录。JSON 与文本报告会写入运行目录或命令输出指示的位置。

## 6. 只做音频审计

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer audit --config config\my_voice.yaml
```

每个文件会记录可解码状态、时长、采样率、声道、编码/位深、LUFS、RMS、峰值、crest factor、削波比例、DC offset、静音比例、动态范围、噪声底、估算 SNR、非有限值和立体声差异，并分类为 `PASS`、`WARNING` 或 `FAIL`。

主要输出：

```text
runs/<run_id>/input_manifest.json
runs/<run_id>/quality/audio_quality.csv
runs/<run_id>/quality/audio_quality.json
runs/<run_id>/quality/audio_quality.html
data/rejected_audio/<run_id>/rejection_reasons.csv
```

## 7. Dry run

真实训练前强烈建议：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer run --config config\my_voice.yaml --dry-run
```

Dry run 只显示输入、将创建的目录、探测到的 RVC 入口、训练参数和预期输出，不执行长时间训练。

## 8. 完整运行

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer run --config config\my_voice.yaml
```

或：

```bat
scripts\run_all.bat
```

阶段顺序：

```text
INITIALIZED
→ AUDIO_DISCOVERED
→ QUALITY_CHECKED
→ PREPROCESSED
→ FEATURES_EXTRACTED
→ MODEL_TRAINED
→ INDEX_BUILT
→ TEST_INFERENCE_COMPLETED
→ REPORT_GENERATED
```

每个成功阶段会原子更新 `state.json` 并保存输入指纹。配置或输入变化会使相关下游阶段失效，但不会删除旧 run。

## 9. 恢复、重新测试和重新生成报告

恢复指定 run：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer resume --run-id 20260718_153000_character_voice_v001
```

或：

```bat
scripts\resume.bat 20260718_153000_character_voice_v001
```

只重新测试已有模型：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer test --run-id <run_id> --test-dir data\test_audio
```

重新生成离线报告：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer report --run-id <run_id>
```

## 10. Run 目录

```text
runs/<run_id>/
├─ config_resolved.yaml
├─ state.json
├─ input_manifest.json
├─ environment_report.json
├─ quality/
├─ preprocessed/
├─ rvc_workspace/
├─ checkpoints/
├─ artifacts/
├─ test_results/
├─ logs/
└─ report/
```

报告入口为 `runs/<run_id>/report/index.html`。它使用本地相对路径和原生 `<audio controls>`，整个 run 目录复制到另一台电脑后仍可试听。`manual_review_template.csv` 用于保存人工评分。

模型清单会明确标注 AI 音色转换用途，并记录模型、索引、checkpoint、配置和 SHA-256。旧模型不会被同名覆盖。

## 11. 自动 batch size 降级

当 `training.automatic_batch_size=true` 时，流程按 `batch_size_candidates` 选择起点。训练子进程返回失败、stderr/日志包含 CUDA OOM 证据且 checkpoint 可验证时，流程记录失败、降到下一个候选值，并从最近有效 checkpoint 恢复。达到最小值仍失败会停止并给出恢复命令。

监控失败本身不会终止训练。正常情况下每隔 `monitoring.interval_seconds` 写入 CPU、内存、磁盘和 GPU 指标；GPU 优先读取 NVML，失败时尝试 `nvidia-smi`。

## 12. 测试

控制层测试不进行真实长时间 RVC 训练：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

手动端到端验收：

1. 准备至少 5 分钟合法授权的训练音频。
2. 准备 3 条测试音频。
3. 运行 `doctor` 并解决所有关键缺失项。
4. 运行 `audit` 并检查质量报告及拒绝原因。
5. 运行一次 `--dry-run`，确认 RVC 命令和参数。
6. 启动完整 `run`，中途手动按一次 Ctrl+C。
7. 用 `resume` 恢复，并确认已有有效阶段未重复覆盖。
8. 检查 `.pth`、`.index`、模型清单和 checkpoint 清单。
9. 打开 `report/index.html` 对比原音与转换结果。

## 常见问题

### `doctor` 报告 FFmpeg 或 FFprobe 缺失

安装可信来源的 FFmpeg，并把包含 `ffmpeg.exe` 与 `ffprobe.exe` 的 `bin` 目录加入系统 `PATH`，重新打开终端后再次执行 `doctor`。

### 找不到兼容的 RVC 脚本

确认 `paths.rvc_repository` 指向实际 checkout，查看 `doctor` 报告中的已探测文件。RVC 分支差异较大；适配器不会假定某个历史路径永久存在。必要时根据该 checkout 的真实 CLI，在适配器的集中命令构造位置增加明确支持并补充测试。

### CUDA OOM

降低 `training.batch_size_candidates`，关闭 `cache_dataset_in_gpu`，确认没有其他进程占用显存。流程不会把所有失败都误判为 OOM；请同时查看返回码、stderr、训练日志和 checkpoint。

### 为什么仓库中没有 `.venv`、训练音频、RVC 或模型？

这些内容体积大、与本机环境相关，或可能包含敏感/授权素材，因此被 `.gitignore` 排除。提交的是可复现配置、代码、测试和目录说明。
