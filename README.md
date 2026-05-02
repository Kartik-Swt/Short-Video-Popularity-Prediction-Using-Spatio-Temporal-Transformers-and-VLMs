# Short-Video Popularity Prediction Using Video Transformers and VLMs

Predict short-form video popularity with a video transformer (TimeSformer) and a vision-language model (SmolVLM2) using parameter-efficient fine-tuning (QLoRA). The training pipeline normalizes a popularity metric, labels the top and bottom quantiles as popular/not-popular, and trains lightweight classifiers on video clips.

## Highlights
- **TimeSformer + QLoRA**: Efficient 4-bit fine-tuning of a video transformer.
- **SmolVLM2 classifier**: Adapts a vision-language model for discriminative video classification.
- **Quantile-based labeling**: Normalized metric with top/bottom quantiles mapped to labels.
- **Built-in evaluation**: Accuracy, weighted F1, precision, and recall.

## 📂 Project Structure

```text
├── models/               # Model architectures and QLoRA configurations
│   ├── __init__.py
│   ├── timesformer.py    # TimeSformer implementation with QLoRA
│   └── smolvlm.py        # SmolVLM2 classification wrapper
├── data/                 # Dataset loading and processing
│   ├── __init__.py
│   └── dataset.py        # Custom Dataset classes and collators
├── utils/                # Utility scripts
│   ├── __init__.py
│   └── metrics.py        # Evaluation metrics (Accuracy, F1, etc.)
├── train_timesformer.py  # CLI training script for TimeSformer
├── train_smolvlm.py      # CLI training script for SmolVLM2
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 12GB+ for QLoRA, 24GB+ for full fine-tuning)

### Setup

```bash
git clone https://github.com/Kartik-Swt/Short-Video-Popularity-Prediction-Using-Video-Transformers-and-VLMs.git
cd Short-Video-Popularity-Prediction-Using-Video-Transformers-and-VLMs
pip install -r requirements.txt
```

> **Weights & Biases**: Training logs are sent to W&B by default. Set `WANDB_DISABLED=true` to disable logging or run `wandb login` before training.

## 💾 Datasets

This project expects a CSV with video paths and a popularity metric column. Two example sources:

### 1. YouTube Shorts Dataset
- **Download:** [Google Drive Link](https://drive.google.com/file/d/1aDJftxei6qqjHREEGgRQ8rCFfwRemw1Q/view?usp=sharing)
- **Format:** CSV containing video paths and engagement metrics.

### 2. Snapchat UGC Dataset
- **Source:** [SnapUGC_Engagement GitHub](https://github.com/dasongli1/SnapUGC_Engagement.git)

### CSV Requirements
- A video path column (default: `video_path`).
- A popularity metric column (default: `ECR`).
- Videos must be accessible at the paths referenced in the CSV.

### Labeling Logic
The loader normalizes the metric column and assigns labels based on quantiles:
- **popular**: top 33% (default)
- **not_popular**: bottom 33%
- The middle quantile is filtered out to create a cleaner binary split.

## 🛠️ Usage

### Training TimeSformer

```bash
python train_timesformer.py \
  --csv_path ./path/to/data.csv \
  --video_col video_path \
  --metric_col ECR \
  --epochs 20 \
  --batch_size 8 \
  --output_dir ./checkpoints/timesformer
```

**Key Arguments**
- `--model_id`: Base model ID (default: `facebook/timesformer-base-finetuned-k400`).
- `--metric_col`: CSV column used to compute popularity labels.
- `--lr`: Learning rate (default: `5e-5`).

Output model is saved to `output_dir/final_model`.

### Training SmolVLM2

```bash
python train_smolvlm.py \
  --csv_path ./path/to/data.csv \
  --video_col video_path \
  --model_id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --epochs 100 \
  --batch_size 4 \
  --output_dir ./checkpoints/smolvlm
```

**Key Arguments**
- `--model_id`: SmolVLM2 model ID (default: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`).
- `--epochs`: Defaults to 100 since VLM fine-tuning typically converges more slowly than TimeSformer.
- `--batch_size`: Tune based on available VRAM (e.g., 4 for 12GB-class GPUs).

Output model is saved to `output_dir/final`.

## 📊 Evaluation

Evaluation runs automatically after training. Metrics include:
- **Accuracy**
- **F1-Score (Weighted)**
- **Precision & Recall (Weighted)**

The best checkpoint is selected based on **F1-Score**.

## 🔍 Notes
- Each video is sampled to 8 frames for both pipelines.
- Train/validation/test splits are stratified with a fixed random seed for reproducibility.
