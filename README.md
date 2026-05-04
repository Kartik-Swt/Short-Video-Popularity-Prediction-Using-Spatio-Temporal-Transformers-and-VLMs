# Short Video Popularity Prediction

> Predict whether a short-form video will go viral using fine-tuned video transformers and vision-language models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project tackles the problem of short-video popularity prediction as a **binary classification task** — given a video, predict whether it belongs to the *popular* or *not popular* category based on an engagement metric (e.g. ECR).

Two distinct model families are fine-tuned and compared:

| Model | Architecture | Fine-tuning Strategy |
|---|---|---|
| **TimeSformer** | Divided Space-Time Attention Transformer | QLoRA (4-bit, `r=16`) |
| **SmolVLM2** | Multimodal VLM (vision + language encoder) | QLoRA (4-bit, `r=8`) + classification head |

Both models are memory-efficient through **4-bit NF4 quantization** (`bitsandbytes`) and **LoRA adapters** (`peft`), making them trainable on consumer-grade GPUs.

---

## How It Works

### Popularity Labelling

Raw engagement metrics (e.g. ECR — Engagement-to-Click Rate) are converted into binary labels using a **quantile-based threshold**:

- Videos above the **67th percentile** → `popular` (label `1`)
- Videos below the **33rd percentile** → `not_popular` (label `0`)
- Videos in between are **excluded** to reduce label ambiguity

This creates a cleaner, more discriminative training set compared to a simple median split.

### Frame Sampling

Both models sample **8 evenly-spaced frames** from each video using [decord](https://github.com/dmlc/decord) for efficient GPU-accelerated video loading.

### TimeSformer Pipeline

```
Video → 8 frames → AutoImageProcessor → TimeSformerForVideoClassification
                                             ↓
                                    QLoRA adapters on [qkv, dense, temporal_dense]
                                             ↓
                                    Linear classifier (2 classes)
```

### SmolVLM2 Pipeline

```
Video → 8 PIL frames → AutoProcessor (chat template) → SmolVLM2 language backbone
                                                              ↓
                                                     Last-token pooling
                                                              ↓
                                                    Linear score head (2 classes)
```

The VLM receives a prompt: *"Classify if this video is popular or not."* alongside the video frames, then classifies based on the final hidden state of the last token.

---

## 📂 Project Structure

```
├── models/
│   ├── timesformer_model.py   # TimeSformer + QLoRA setup
│   └── smolvlm.py             # SmolVLMForVideoClassification wrapper
├── data/
│   └── dataset.py             # prepare_dataframe, Dataset classes, collators
├── utils/
│   └── metrics.py             # Accuracy, F1, Precision, Recall
├── train_timesformer.py       # CLI training script for TimeSformer
├── train_smolvlm.py           # CLI training script for SmolVLM2
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

**Prerequisites:** Python 3.8+, a CUDA-compatible GPU (12 GB+ VRAM recommended with QLoRA; 24 GB+ for larger batch sizes).

```bash
git clone https://github.com/Kartik-Swt/Short-Video-Popularity-Prediction-Using-Video-Transformers-and-VLMs.git
cd Short-Video-Popularity-Prediction-Using-Video-Transformers-and-VLMs
pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `peft`, `bitsandbytes`, `decord`, `datasets`, `accelerate`, `scikit-learn`, `pandas`.

---

## 💾 Datasets

Two short-form video datasets are supported:

### YouTube Shorts

A curated collection of YouTube Shorts with engagement metadata.

- **Download:** [Google Drive](https://drive.google.com/file/d/1aDJftxei6qqjHREEGgRQ8rCFfwRemw1Q/view?usp=sharing)
- **Format:** CSV with video file paths and engagement metrics (e.g. `ECR`)

### Snapchat UGC (SnapUGC Engagement)

User-generated content from Snapchat with engagement labels.

- **Source:** [SnapUGC_Engagement on GitHub](https://github.com/dasongli1/SnapUGC_Engagement)
- Follow the instructions in that repository to download raw videos and metadata.

### CSV Format Requirements

Your CSV must contain:
- A **video path column** (default: `video_path`) — absolute or relative path to each video file.
- An **engagement metric column** (default: `ECR`) — any numeric column representing popularity (views, likes, ECR, etc.).

The scripts handle normalization and quantile-based label creation automatically.

---

## 🛠️ Usage

### Train TimeSformer

```bash
python train_timesformer.py \
  --csv_path ./data/youtube_shorts.csv \
  --video_col video_path \
  --metric_col ECR \
  --model_id facebook/timesformer-base-finetuned-k400 \
  --output_dir ./checkpoints/timesformer \
  --epochs 20 \
  --batch_size 8 \
  --lr 5e-5
```

| Argument | Default | Description |
|---|---|---|
| `--csv_path` | *(required)* | Path to the dataset CSV |
| `--video_col` | `video_path` | CSV column containing video file paths |
| `--metric_col` | `ECR` | Numeric engagement column used for labelling |
| `--model_id` | `facebook/timesformer-base-finetuned-k400` | HuggingFace model ID |
| `--output_dir` | `./timesformer_output` | Directory to save checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `8` | Per-device batch size (effective batch = `batch_size × 4` with grad accumulation) |
| `--lr` | `5e-5` | Learning rate |

### Train SmolVLM2

```bash
python train_smolvlm.py \
  --csv_path ./data/snapchat_ugc.csv \
  --video_col video_path \
  --model_id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --output_dir ./checkpoints/smolvlm \
  --epochs 50 \
  --batch_size 4
```

| Argument | Default | Description |
|---|---|---|
| `--csv_path` | *(required)* | Path to the dataset CSV |
| `--video_col` | `video_path` | CSV column containing video file paths |
| `--model_id` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | HuggingFace model ID (`500M` or `2.2B` variants) |
| `--output_dir` | `./smolvlm_output` | Directory to save checkpoints |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `8` | Per-device batch size (effective batch = `batch_size × 16` with grad accumulation) |

> **Tip:** The SmolVLM2-2.2B variant requires more VRAM but may achieve better accuracy. Start with `500M` for faster iteration.

---

## 🏋️ Training Details

Both training scripts share the same setup:

- **Data split:** 83% train / 12% validation / 5% test (stratified)
- **Optimizer:** `paged_adamw_8bit` (memory-efficient)
- **Best model selection:** highest weighted F1 on the validation set
- **Early stopping:** patience of 3 epochs
- **Experiment tracking:** [Weights & Biases](https://wandb.ai/) (`report_to="wandb"`)

| | TimeSformer | SmolVLM2 |
|---|---|---|
| Precision | `fp16` | `bf16` |
| Gradient accumulation | 4 steps | 16 steps |
| Warmup | 10% of steps | — |
| LoRA rank | 16 | 8 |
| LoRA alpha | 32 | 16 |

---

## 📊 Evaluation

Evaluation runs automatically after training on the held-out test set. Metrics reported:

- **Accuracy**
- **F1-Score** (weighted)
- **Precision** (weighted)
- **Recall** (weighted)

Checkpoints are saved every epoch, and the best checkpoint (by F1) is reloaded at the end of training before final evaluation.

---

## 🔧 Extending the Project

- **Custom engagement metric:** pass any numeric CSV column via `--metric_col` (e.g. `--metric_col views`).
- **Different quantile thresholds:** modify `threshold` in `data/dataset.py:prepare_dataframe` (default `0.33` gives a 33/33 popular/not-popular split).
- **Larger VLM:** swap `--model_id` to `HuggingFaceTB/SmolVLM2-2.2B-Video-Instruct` for the bigger model.
- **More frames:** adjust `num_frames` in `TimesformerDataset` or `get_collate_fn_smolvlm` (currently `8`).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
