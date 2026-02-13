import argparse
import torch
from transformers import AutoProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from data.dataset import prepare_dataframe, get_collate_fn_smolvlm
from models.smolvlm import SmolVLMForVideoClassification
from utils.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Train SmolVLM2 for Video Classification")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--video_col", type=str, default="video_path")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--output_dir", type=str, default="./smolvlm_output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()

    # 1. Data
    df = prepare_dataframe(args.csv_path)
    label_map = {0: 'not_popular', 1: 'popular'}
    df['label'] = df['popularity'].map(label_map)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.video_col].tolist(), df['label'].tolist(),
        test_size=0.05, stratify=df['label'], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.12, stratify=y_train, random_state=42
    )
    
    data = DatasetDict({
        'train': Dataset.from_dict({'label': y_train, 'video': X_train}),
        'validation': Dataset.from_dict({'label': y_val, 'video': X_val}),
        'test': Dataset.from_dict({'label': y_test, 'video': X_test})
    })

    # 2. Model & Processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SmolVLMForVideoClassification.from_pretrained(args.model_id)
    model.print_trainable_parameters()
    
    collate_fn = get_collate_fn_smolvlm(processor, {"not_popular": 0, "popular": 1})
    
    # 3. Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to='wandb'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    print(trainer.evaluate(data["test"]))

if __name__ == "__main__":
    main()
