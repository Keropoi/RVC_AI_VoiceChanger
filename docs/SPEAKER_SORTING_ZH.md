# 多说话人录音：自动分拣目标角色

清晰、轮流说话的多人对白适合“说话人日志分离（diarization）”；它回答“谁在什么时间说话”。只有多人同时说话时才涉及音源分离，而重叠对白通常不适合作为 RVC 训练素材，本项目会把它们从可自动提升的片段中排除。

## 推荐流程

1. 原始多人录音放入 `data/mixed_speaker_audio/`，不要剪改原件。
2. 在 `data/voice_references/` 放入目标角色 3–10 条干净参考，合计约 30–60 秒。没有参考也可聚类，但每个录音都需人工指定匿名说话人。
3. 自动完成 VAD/说话人聚类、按说话人切片、重叠标记和声纹相似度排序。
4. 每个匿名簇试听三个样本，只填写 `TARGET`、`OTHER` 或 `REJECT`。
5. `TARGET` 的非重叠片段复制到 `data/speaker_selected_audio/`，再进入普通 `prepare-data` 质检；原录音和候选均保留。

匿名标签（如 `SPEAKER_00`）只在单个录音内有意义，不能假设另一个文件的 `SPEAKER_00` 是同一人。目标参考声纹用于跨文件排序，但相似度只是复核依据，不是身份认证，也不会自动批准素材。

## 安装可选的独立环境

说话人分拣使用独立 `.speaker_venv`，不会改变 RVC 的 `.venv`。`pyannote.audio 4.x` 要求 Python 3.10 或更高；当前安装脚本使用 Python 3.11，未检测到时会先询问是否通过 winget 安装：

```bat
scripts\setup_speaker_venv.bat
```

首次使用需在 [pyannote Community-1 模型页](https://huggingface.co/pyannote/speaker-diarization-community-1)接受条件，然后登录 Hugging Face：

```powershell
.\.speaker_venv\Scripts\hf.exe auth login
```

当前后端为 `pyannote/speaker-diarization-community-1`；官方说明它会自动转为 16 kHz 单声道、支持本地 GPU，并可传入确切/最小/最大说话人数。目标排序使用官方 [WeSpeaker 声纹嵌入模型](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)。

## 执行

已知每段录音恰好两人：

```powershell
.\.speaker_venv\Scripts\python.exe -m rvc_auto_trainer sort-speakers `
  --config config\japanese_to_chinese_dialogue.yaml `
  --num-speakers 2
```

只知道大约 2–5 人：

```powershell
.\.speaker_venv\Scripts\python.exe -m rvc_auto_trainer sort-speakers `
  --config config\japanese_to_chinese_dialogue.yaml `
  --min-speakers 2 --max-speakers 5
```

结果位于：

- `data/speaker_segments/<run_id>/`：按来源和匿名说话人保存的 PCM24 候选；
- `data/speaker_manifests/<run_id>/speaker_segments.csv`：逐片段时间、哈希和重叠标记；
- `data/speaker_manifests/<run_id>/speaker_review.csv`：按说话人簇汇总的试听与决策表。

编辑 `speaker_review.csv` 后执行：

```powershell
.\.venv\Scripts\python.exe -m rvc_auto_trainer apply-speaker-review `
  --config config\japanese_to_chinese_dialogue.yaml `
  --review-queue data\speaker_manifests\<run_id>\speaker_review.csv
```

随后运行 `prepare-data`。分拣不会修复 BGM、混响、电话音质或强降噪伪影；这些仍由质量审计和人工复核剔除。角色刻意改变声线、耳语、哭喊或远距离录音可能降低声纹相似度，应放入人工复核而不是直接拒绝。
