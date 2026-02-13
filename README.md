This repository contains the implementation for predicting short-form video popularity using **TimeSformer** and **SmolVLM2** with Parameter-Efficient Fine-Tuning (PEFT/QLoRA).



## Features
* **TimeSformer + QLoRA**: Efficient video transformer fine-tuning.
* **SmolVLM2 Classifier**: Adapting a Vision-Language Model for discriminative video classification.
* **Memory Optimized**: Support for 4-bit quantization and gradient checkpointing.

## 📂 Project Structure

```text
.
├── models/               # Model architectures and QLoRA configurations
│   ├── timesformer.py    # TimeSformer implementation
│   └── smolvlm.py        # SmolVLM2 classification wrapper
├── data/                 # Dataset loading and processing
├── utils/                # Evaluation metrics
├── train_timesformer.py  # Training script for TimeSformer
├── train_smolvlm.py      # Training script for SmolVLM2
└── requirements.txt      # Dependencies

🚀 Installation
Clone the repository:

Bash
git clone [https://github.com/yourusername/video-popularity-prediction.git](https://github.com/yourusername/video-popularity-prediction.git)
cd video-popularity-prediction
Install dependencies:

Bash
pip install -r requirements.txt
Pw Datasets
This project utilizes two primary datasets for short-form video analysis. Please download them from the sources below:

1. YouTube Shorts Dataset
A curated collection of YouTube Shorts with associated metadata and popularity metrics.

Download: Google Drive Link

Format: CSV containing video paths and engagement metrics.

2. Snapchat UGC Dataset
Derived from the SnapUGC Engagement dataset.

Source: SnapUGC_Engagement GitHub

Note: Please follow the instructions in the linked repository to download the raw video files.

Preprocessing:
Ensure your CSV files have a column for the video file path (e.g., video_path) and a metric column (e.g., ECR or popularity_score) before running the training scripts.

🛠️ Usage
Training TimeSformer
To train the TimeSformer model using QLoRA 4-bit quantization:

Bash
python train_timesformer.py \
  --csv_path ./path/to/data.csv \
  --video_col video_path \
  --metric_col ECR \
  --epochs 20 \
  --batch_size 8 \
  --output_dir ./checkpoints/timesformer
Training SmolVLM2
To train the Vision-Language Model (SmolVLM2) for discriminative classification:

Bash
python train_smolvlm.py \
  --csv_path ./path/to/data.csv \
  --model_id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --epochs 50 \
  --batch_size 4 \
  --output_dir ./checkpoints/smolvlm
📊 Results & Metrics
The models are evaluated using the following metrics:

Accuracy

F1-Score (Weighted)

Precision & Recall

Evaluation is performed automatically on the held-out test set after training completes.

📜 Citation
If you use this code or dataset in your research, please cite our paper:

Code snippet
@article{yourname2026video,
  title={Multimodal Popularity Prediction in Short-form Videos},
  author={Kartikeya, Name2, Name3},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026}
}
⚖️ License
This project is licensed under the MIT License.
