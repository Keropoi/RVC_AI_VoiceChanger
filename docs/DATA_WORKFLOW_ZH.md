# 日文目标音色 → 中文对白：可执行数据流程

本流程只处理已获授权的音频。所有归档、分拣和复核步骤均采用“复制到下一层”的方式；程序不会自动永久删除原素材。

## 1. 单人干净素材

把原始 WAV/FLAC（压缩格式也支持）放入 `data/raw_archive/`。长录音无需手工逐句切分；`prepare-data` 会先将超过 5 分钟的录音按静音粗切为约 2–5 分钟候选，正式 RVC 预处理仍使用官方约 3.7 秒切片参数。

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer prepare-data `
  --config config\japanese_to_chinese_dialogue.yaml `
  --source-label "素材批次或作品名" `
  --rights-note "授权或录音同意说明" `
  --language ja
```

命令会记录来源说明、授权说明、语言、文件大小和 SHA-256，并生成：

- `data/dataset_manifests/raw_archive_manifest.json`
- `data/dataset_manifests/candidate_quality/`
- `data/dataset_manifests/review_queue.csv`

在 `review_queue.csv` 中填写 `decision`：

- `KEEP`：复制到正式训练输入；
- `REFERENCE`：仅作 1–2 分钟的音色参照，不参加训练；
- `REJECT`：只记录拒绝决定，不删除候选或原始文件。

所有 `WARNING`、所有 `FAIL`，以及不少于 PASS 的 10% 或 50 条（取较大者，但不超过实际数量）都会要求试听。当前实现不会自动把 WARNING 提升到训练集。

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer apply-data-review `
  --config config\japanese_to_chinese_dialogue.yaml `
  --review-queue data\dataset_manifests\review_queue.csv
```

先用最干净的约 10 分钟试训，再扩至 15–20 分钟核心集；新增素材只有在固定测试集上确实改善时才保留。

## 2. 固定中文/日文测试集

训练语言与推理语言不必相同，但跨语言音素和中文声调仍需固定测试。将五条输入放入 `data/test_audio/`：

1. 10–15 秒普通中文对白；
2. 10–15 秒中文音素/四声压力句；
3. 10–20 秒快速或情绪化中文；
4. 30–60 秒中文自然段落；
5. 10–20 秒日文控制样本。

复制模板并按实际文件名调整：

```powershell
Copy-Item data\test_audio\test_manifest.example.yaml data\test_audio\test_manifest.yaml
.\.venv\Scripts\python.exe -m rvc_auto_trainer freeze-tests `
  --config config\japanese_to_chinese_dialogue.yaml
.\.venv\Scripts\python.exe -m rvc_auto_trainer validate-tests `
  --config config\japanese_to_chinese_dialogue.yaml
```

冻结会强制检查五种角色、时长、唯一内容、训练/候选集泄漏，并写入 SHA-256。之后文件内容或规则发生变化，正式流水线会在训练前失败，不会悄悄换测试样本。

该配置固定音高偏移为 0，并比较 `index_rate` 0.35、0.50、0.65、0.80。另准备一条 2–3 分钟中文压力样本作最终人工检查，但不要用它替代上述五条定位样本。

## 3. 开始训练前

```powershell
scripts\doctor.bat
.\.venv\Scripts\python.exe -m rvc_auto_trainer audit `
  --config config\japanese_to_chinese_dialogue.yaml
.\.venv\Scripts\python.exe -m rvc_auto_trainer run `
  --config config\japanese_to_chinese_dialogue.yaml --dry-run
```

`doctor PASS` 只表示依赖、GPU、RVC 入口、资源和项目索引构建器可用。真实 GPU 训练与中文推理验收仍必须使用你的授权音频完成。
