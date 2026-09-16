import os
import csv
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.text.preprocessor import clean_news_text

ISOT_FAKE_PATH = Path(r"C:\Users\priya\Downloads\News-_dataset\Fake.csv")
ISOT_TRUE_PATH = Path(r"C:\Users\priya\Downloads\News-_dataset\True.csv")
LIAR2_TRAIN_PATH = Path(r"C:\Users\priya\Downloads\train.csv")
LIAR2_VALID_PATH = Path(r"C:\Users\priya\Downloads\valid.csv")
LIAR2_TEST_PATH = Path(r"C:\Users\priya\Downloads\test.csv")
PROCESSED_DIR = Path("ml-service/dataset/processed")

def load_isot_dataset(max_samples_per_class: int = 6000) -> List[Dict[str, Any]]:
    """
    Adapter for ISOT Fake News Dataset.
    Harmonizes Fake.csv and True.csv into unified schema.
    """
    records = []
    if not ISOT_FAKE_PATH.exists() or not ISOT_TRUE_PATH.exists():
        print(f"Warning: ISOT dataset paths not found at {ISOT_FAKE_PATH}")
        return records

    print(f"Loading ISOT dataset from {ISOT_FAKE_PATH.parent}...")

    # Load Fake
    fake_df = pd.read_csv(ISOT_FAKE_PATH, nrows=max_samples_per_class).fillna("")
    for _, row in fake_df.iterrows():
        title = clean_news_text(str(row.get("title", "")))
        text = clean_news_text(str(row.get("text", "")))
        if not title and not text:
            continue
        records.append({
            "id": f"isot_fake_{len(records)}",
            "title": title,
            "text": text,
            "label": "fake",
            "source": str(row.get("subject", "general")),
            "author": "Unknown",
            "date": str(row.get("date", "")),
            "url": "",
            "dataset_origin": "ISOT",
            "domain": "politics/news",
            "raw_label": "fake"
        })

    # Load True
    true_df = pd.read_csv(ISOT_TRUE_PATH, nrows=max_samples_per_class).fillna("")
    for _, row in true_df.iterrows():
        title = clean_news_text(str(row.get("title", "")))
        text = clean_news_text(str(row.get("text", "")))
        if not title and not text:
            continue
        records.append({
            "id": f"isot_true_{len(records)}",
            "title": title,
            "text": text,
            "label": "real",
            "source": "Reuters",
            "author": "Reuters Staff",
            "date": str(row.get("date", "")),
            "url": "",
            "dataset_origin": "ISOT",
            "domain": "politics/news",
            "raw_label": "true"
        })

    print(f"Loaded {len(records)} harmonized records from ISOT.")
    return records

def load_liar2_dataset(max_samples: int = 15000) -> List[Dict[str, Any]]:
    """
    Adapter for LIAR2 / LIAR-PLUS dataset.
    Loads from train.csv, valid.csv, and test.csv if present.
    Harmonizes 6-level labels into unified {real, fake, suspicious} representation:
      - 5 (True) -> 'real'
      - 4 (Mostly-True) -> 'real'
      - 3 (Half-True) -> 'suspicious'
      - 2 (Barely-True) -> 'suspicious'
      - 1 (False) -> 'fake'
      - 0 (Pants-on-fire) -> 'fake'
    """
    records = []
    source_files = [p for p in [LIAR2_TRAIN_PATH, LIAR2_VALID_PATH, LIAR2_TEST_PATH] if p.exists()]
    if not source_files:
        print(f"Warning: No LIAR2 dataset files found in {LIAR2_TRAIN_PATH.parent}")
        return records

    print(f"Loading LIAR2 dataset from {[f.name for f in source_files]}...")
    
    label_map = {
        "0": "fake",       # Pants on fire
        "1": "fake",       # False
        "2": "suspicious", # Barely true
        "3": "suspicious", # Half true
        "4": "real",       # Mostly true
        "5": "real",       # True
        0: "fake",
        1: "fake",
        2: "suspicious",
        3: "suspicious",
        4: "real",
        5: "real"
    }

    for sf in source_files:
        liar_df = pd.read_csv(sf).fillna("")
        for _, row in liar_df.iterrows():
            if len(records) >= max_samples:
                break
            raw_lbl = str(row.get("label", "")).strip()
            mapped_label = label_map.get(raw_lbl)
            if not mapped_label:
                continue

            statement = clean_news_text(str(row.get("statement", "")))
            context = str(row.get("context", "")).strip()
            justification = clean_news_text(str(row.get("justification", "")))
            full_text = f"{statement}. {justification}".strip() if justification else statement

            records.append({
                "id": f"liar2_{row.get('id', len(records))}",
                "title": statement,
                "text": full_text,
                "label": mapped_label,
                "source": str(row.get("speaker_description", row.get("speaker", "Unknown"))),
                "author": str(row.get("speaker", "Unknown")),
                "date": str(row.get("date", "")),
                "url": "",
                "dataset_origin": "LIAR2",
                "domain": str(row.get("subject", "politics")),
                "raw_label": raw_lbl
            })
        if len(records) >= max_samples:
            break

    print(f"Loaded {len(records)} harmonized records from LIAR2.")
    return records

def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prevent data leakage across splits by deduplicating exact and near-exact text hashes.
    """
    seen_hashes = set()
    deduped = []
    duplicates_count = 0

    for rec in records:
        content_for_hash = f"{rec['title'].lower()}|{rec['text'][:200].lower()}"
        sha = hashlib.sha256(content_for_hash.encode("utf-8")).hexdigest()
        if sha in seen_hashes:
            duplicates_count += 1
            continue
        seen_hashes.add(sha)
        deduped.append(rec)

    print(f"Deduplication: removed {duplicates_count} duplicates. Retained {len(deduped)} unique records.")
    return deduped

def harmonize_and_split(
    isot_sample: int = 6000,
    liar2_sample: int = 8000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Execute end-to-end dataset harmonization across ISOT and LIAR2.
    Produces leak-free stratified train, val, test splits.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    isot_records = load_isot_dataset(max_samples_per_class=isot_sample)
    liar_records = load_liar2_dataset(max_samples=liar2_sample)
    all_records = isot_records + liar_records

    deduped_records = deduplicate_records(all_records)
    df = pd.DataFrame(deduped_records)

    # Class distribution
    label_counts = df["label"].value_counts().to_dict()
    origin_counts = df["dataset_origin"].value_counts().to_dict()
    print("Class distribution:", label_counts)
    print("Origin distribution:", origin_counts)

    # Stratified Train / Validation / Test split
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_state
    )

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    report = {
        "total_records": len(df),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "class_distribution": label_counts,
        "origin_distribution": origin_counts,
        "train_distribution": train_df["label"].value_counts().to_dict(),
        "val_distribution": val_df["label"].value_counts().to_dict(),
        "test_distribution": test_df["label"].value_counts().to_dict(),
        "mapping_strategy": {
            "ISOT": "Fake.csv -> fake, True.csv -> real",
            "LIAR2": "0,1 -> fake; 2,3 -> suspicious; 4,5 -> real"
        }
    }

    with open(PROCESSED_DIR / "harmonization_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved harmonized splits to {PROCESSED_DIR}:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    return report

if __name__ == "__main__":
    harmonize_and_split()
