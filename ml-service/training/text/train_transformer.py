import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.text.transformer_model import (
    TransformerTextClassifier, TemperatureScaler, DEFAULT_MODEL_NAME, DEFAULT_TOKENIZER_NAME
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "text" / "transformer_v1"
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation"

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_transformer(sample_size=1200, epochs=2, batch_size=16, lr=3e-5):
    print("=" * 60)
    print("TRAINING TRANSFORMER TEXT MODEL (DistilBERT Fine-Tuning)")
    print("=" * 60)

    train_df = pd.read_csv(DATA_DIR / "train.csv", engine="python").fillna("")
    val_df = pd.read_csv(DATA_DIR / "val.csv", engine="python").fillna("")
    test_df = pd.read_csv(DATA_DIR / "test.csv", engine="python").fillna("")

    # Stratified sub-sample for responsive CPU execution
    if sample_size and sample_size < len(train_df):
        def sample_by_label(df_in, count):
            parts = []
            for lbl in ["fake", "real", "suspicious"]:
                sub = df_in[df_in["label"] == lbl]
                parts.append(sub.sample(min(len(sub), count // 3), random_state=42))
            return pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

        train_df = sample_by_label(train_df, sample_size)
        val_df = sample_by_label(val_df, 300)
        test_df = sample_by_label(test_df, 450)

    print(f"Training on {len(train_df)} samples | Val: {len(val_df)} | Test: {len(test_df)}")

    label2id = {"fake": 0, "real": 1, "suspicious": 2}
    id2label = {0: "fake", 1: "real", 2: "suspicious"}

    train_texts = (train_df["title"] + " " + train_df["text"]).tolist()
    train_labels = [label2id[l] for l in train_df["label"].tolist()]

    val_texts = (val_df["title"] + " " + val_df["text"]).tolist()
    val_labels = [label2id[l] for l in val_df["label"].tolist()]

    test_texts = (test_df["title"] + " " + test_df["text"]).tolist()
    test_labels = [label2id[l] for l in test_df["label"].tolist()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading pretrained tokenizer ({DEFAULT_TOKENIZER_NAME}) & model ({DEFAULT_MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    model = BertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Unfreeze Transformer backbone for rich end-to-end fine-tuning
    print("Fine-tuning BERT-tiny end-to-end for authentic semantic contextual representations...")
    for param in model.bert.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    train_ds = NewsDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_ds = NewsDataset(val_texts, val_labels, tokenizer, max_len=128)
    test_ds = NewsDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    print("Starting training epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        val_logits_all = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                val_logits_all.append(outputs.logits.cpu())
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        print(f"  --> Epoch {epoch+1} Validation Accuracy: {val_acc * 100:.2f}%")

    # Learn optimal temperature T on validation logits via NLL loss
    print("\nCalibrating confidence via Temperature Scaling...")
    val_logits_tensor = torch.cat(val_logits_all, dim=0)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    temp_scaler = TemperatureScaler()
    calib_optimizer = torch.optim.LBFGS([temp_scaler.temperature], lr=0.01, max_iter=50)

    def eval_temp():
        calib_optimizer.zero_grad()
        loss = criterion(temp_scaler(val_logits_tensor), val_labels_tensor)
        loss.backward()
        return loss

    calib_optimizer.step(eval_temp)
    learned_temperature = float(temp_scaler.temperature.item())
    print(f"Learned Optimal Temperature T: {learned_temperature:.3f}")

    # Evaluate on Unseen Held-out Test Set
    print("\nEvaluating fine-tuned Transformer on Held-out Test Set...")
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            calibrated_logits = outputs.logits / learned_temperature
            probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            test_preds.extend(preds)
            test_targets.extend(batch["labels"].numpy())
            test_probs.extend(probs)

    test_acc = float(accuracy_score(test_targets, test_preds))
    target_names = ["fake", "real", "suspicious"]
    report = classification_report(test_targets, test_preds, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds).tolist()

    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Macro F1: {report['macro avg']['f1-score']:.4f}")
    for c_idx, c_name in enumerate(target_names):
        m = report[c_name]
        print(f"    {c_name.upper():12s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1-score']:.4f}")

    # Save Model & Artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(OUTPUT_DIR / "temperature.json", "w") as f:
        json.dump({
            "temperature": round(learned_temperature, 4),
            "labels": target_names,
            "test_accuracy": round(test_acc, 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4)
        }, f, indent=2)

    # Load baseline metrics for direct empirical comparison
    baseline_metrics_path = Path(__file__).resolve().parent.parent.parent / "models" / "text" / "baseline_metrics.json"
    baseline_acc = None
    baseline_f1 = None
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r") as f:
            b_data = json.load(f)
            baseline_acc = b_data.get("test", {}).get("accuracy")
            baseline_f1 = b_data.get("test", {}).get("macro_f1")

    comparison_report = {
        "baseline_model": {
            "name": "TF-IDF + Logistic Regression",
            "accuracy": baseline_acc,
            "macro_f1": baseline_f1
        },
        "transformer_model": {
            "name": f"Fine-Tuned {DEFAULT_MODEL_NAME}",
            "accuracy": round(test_acc, 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "temperature_calibration": round(learned_temperature, 4),
            "confusion_matrix": cm,
            "classification_report": report
        },
        "empirical_improvement": {
            "accuracy_delta": round(test_acc - (baseline_acc or 0.0), 4),
            "f1_delta": round(report["macro avg"]["f1-score"] - (baseline_f1 or 0.0), 4)
        }
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "text_model_comparison.json", "w") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\nSaved fine-tuned Transformer model to: {OUTPUT_DIR}")
    print(f"Saved comparative evaluation to: {EVAL_DIR / 'text_model_comparison.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    train_transformer(sample_size=args.sample, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
