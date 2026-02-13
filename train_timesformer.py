import argparse
import torch
from transformers import AutoImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split

from data.dataset import prepare_dataframe, TimesformerDataset, collate_fn_timesformer
from models.timesformer import get_timesformer_model
from utils.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Train TimeSformer for Video Classification")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--video_col", type=str, default="video_path", help="Column name for video paths")
    parser.add_argument("--metric_col", type=str, default="ECR", help="Metric column for popularity")
    parser.add_argument("--model_id", type=str, default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--output_dir", type=str, default="./timesformer_output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    # 1. Prepare Data
    df = prepare_dataframe(args.csv_path, metric_col=args.metric_col)
    ID2LABEL = {0: "not_popular", 1: "popular"}
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.video_col].tolist(), df['popularity'].tolist(),
        test_size=0.05, stratify=df['popularity'], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.12, stratify=y_train, random_state=42
    )
    
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    train_ds = TimesformerDataset(X_train, y_train, processor)
    val_ds = TimesformerDataset(X_val, y_val, processor)
    test_ds = TimesformerDataset(X_test, y_test, processor)
    
    # 2. Model
    model = get_timesformer_model(args.model_id, LABEL2ID, ID2LABEL)
    
    # 3. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="timesformer-run",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="wandb"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=collate_fn_timesformer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting Training...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final_model")
    
    print("Evaluating...")
    print(trainer.evaluate(test_ds))

if __name__ == "__main__":
    main()
