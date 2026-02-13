This repository contains the implementation for predicting short-form video popularity using **TimeSformer** and **SmolVLM2** with Parameter-Efficient Fine-Tuning (PEFT/QLoRA).



## Features
* **TimeSformer + QLoRA**: Efficient video transformer fine-tuning.
* **SmolVLM2 Classifier**: Adapting a Vision-Language Model for discriminative video classification.
* **Memory Optimized**: Support for 4-bit quantization and gradient checkpointing.

## 📂 Project Structure

```text
├── models/               # Model architectures and QLoRA configurations
│   ├── __init__.py
│   ├── timesformer.py    # TimeSformer implementation with QLoRA
│   └── smolvlm.py        # SmolVLM2 classification wrapper
├── data/                 # Dataset loading and processing
│   ├── __init__.py
│   └── dataset.py        # Custom Dataset classes and Collators
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

* Python 3.8+
* CUDA-compatible GPU (Recommended: 24GB VRAM for full fine-tuning, 12GB+ for QLoRA)

### Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/Kartik-Swt/Short-Video-Popularity-Prediction-Using-Spatio-Temporal-Transformers-and-VLMs.git](https://github.com/Kartik-Swt/Short-Video-Popularity-Prediction-Using-Spatio-Temporal-Transformers-and-VLMs.git)
cd video-popularity-prediction

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## 💾 Datasets

This project utilizes two primary datasets for short-form video analysis. Please download them from the sources below and ensure they are accessible to the training scripts.

### 1. YouTube Shorts Dataset

A curated collection of YouTube Shorts with associated metadata and popularity metrics.

* **Download:** [Google Drive Link](https://drive.google.com/file/d/1aDJftxei6qqjHREEGgRQ8rCFfwRemw1Q/view?usp=sharing)
* **Format:** CSV containing video paths and engagement metrics.

### 2. Snapchat UGC Dataset

Derived from the SnapUGC Engagement dataset.

* **Source:** [SnapUGC_Engagement GitHub](https://github.com/dasongli1/SnapUGC_Engagement.git)
* **Note:** Please follow the instructions in the linked repository to download the raw video files and metadata.

**Preprocessing Note:**
Ensure your CSV files have a column for the video file path (e.g., `video_path`) and a metric column (e.g., `ECR` or `popularity_score`) before running the training scripts. The scripts handle normalization and quantile-based labeling automatically.

## 🛠️ Usage

### Training TimeSformer

To train the TimeSformer model using QLoRA 4-bit quantization:

```bash
python train_timesformer.py \
  --csv_path ./path/to/data.csv \
  --video_col video_path \
  --metric_col ECR \
  --epochs 20 \
  --batch_size 8 \
  --output_dir ./checkpoints/timesformer

```

**Key Arguments:**

* `--model_id`: Base HuggingFace model (default: `facebook/timesformer-base-finetuned-k400`).
* `--metric_col`: The column name in CSV used to determine popularity (e.g., `ECR`, `views`).

### Training SmolVLM2

To train the Vision-Language Model (SmolVLM2) for discriminative classification:

```bash
python train_smolvlm.py \
  --csv_path ./path/to/data.csv \
  --video_col video_path \
  --model_id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --epochs 50 \
  --batch_size 4 \
  --output_dir ./checkpoints/smolvlm

```

**Key Arguments:**

* `--model_id`: Choose between `500M` (faster) or `2.2B` variants.
* `--epochs`: VLM fine-tuning typically requires more epochs for convergence.

## 📊 Results & Metrics

The models are evaluated using the following metrics on a held-out test set:

* **Accuracy**
* **F1-Score (Weighted)**
* **Precision & Recall (Weighted)**

Evaluation is performed automatically after training completes. Best checkpoints are saved based on the **F1-Score**.
