import argparse
import os
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------------
# Dataset filename handling
# -------------------------
# The BanFakeNews Kaggle dataset commonly includes files like:
#   - Authentic-48.csv
#   - Fake-1K.csv
#   - LabeledAuthentic-7K.csv
#   - LabeledFake-1K.csv
# Some older project drafts expect "Authentic-48K.csv" (note the K). We support both.

AUTH_CANDIDATES = [
    "Authentic-48K.csv",
    "Authentic-48.csv",
    "LabeledAuthentic-7K.csv",  # fallback only
]

FAKE_CANDIDATES = [
    "Fake-1K.csv",
    "LabeledFake-1K.csv",  # fallback only
]


def _pick_existing_file(data_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of the expected files were found in {data_dir}. Tried: {candidates}"
    )


def _detect_text_column(df: pd.DataFrame) -> str:
    """Detect the text column in a robust way."""
    preferred = ["content", "text", "news", "article", "headline"]
    cols_lower = {c.lower(): c for c in df.columns}

    for key in preferred:
        if key in cols_lower:
            return cols_lower[key]

    # Fallback: choose the first object/string-like column with sufficient non-null values
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if object_cols:
        return object_cols[0]

    raise ValueError(
        f"Could not detect a text column. Columns available: {list(df.columns)}"
    )


def load_data(data_path: str) -> pd.DataFrame:
    data_dir = Path(data_path)
    logging.info(f"Loading data from {data_dir}...")

    auth_path = _pick_existing_file(data_dir, AUTH_CANDIDATES)
    fake_path = _pick_existing_file(data_dir, FAKE_CANDIDATES)

    logging.info(f"Using authentic file: {auth_path.name}")
    logging.info(f"Using fake file: {fake_path.name}")

    auth = pd.read_csv(auth_path)
    fake = pd.read_csv(fake_path)

    auth_col = _detect_text_column(auth)
    fake_col = _detect_text_column(fake)

    # Normalize schema to a single column: content
    auth = auth.rename(columns={auth_col: "content"})
    fake = fake.rename(columns={fake_col: "content"})

    auth["label"] = 0
    fake["label"] = 1

    df = pd.concat([auth[["content", "label"]], fake[["content", "label"]]], ignore_index=True)

    # Drop duplicates and do robust basic cleaning
    df["content"] = df["content"].fillna("").astype(str)
    df["content"] = df["content"].str.replace("\u200c", "", regex=False)  # remove ZWNJ if present
    df["content"] = df["content"].str.replace("\u200d", "", regex=False)  # remove ZWJ if present

    df = df[df["content"].str.strip() != ""]
    df.drop_duplicates(subset=["content"], inplace=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logging.info(f"Data loaded. Shape: {df.shape}")
    logging.info(f"Label counts: {df['label'].value_counts().to_dict()}")

    return df


def _train_model(model_type: str, X_train_vec, y_train):
    """
    Train model with GPU (CUDA) whenever supported by the algorithm and environment.
    CPU is used only as a fallback.
    """

    if model_type == "logistic":
        # Logistic Regression in sklearn is CPU-only
        logging.info("Logistic Regression does not support CUDA; using CPU.")
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train_vec, y_train)
        return model

    if model_type == "svm":
        # LinearSVC is CPU-only; calibration also CPU-only
        logging.info("LinearSVM does not support CUDA; using CPU.")
        base_svm = LinearSVC(
            C=1.0,
            dual="auto",
            random_state=42,
            class_weight="balanced",
        )
        model = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
        model.fit(X_train_vec, y_train)
        return model

    if model_type == "lightgbm":
        # LightGBM supports CUDA if built with GPU support
        logging.info("Attempting to train LightGBM with CUDA (GPU)...")
        try:
            model = lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                force_col_wise=True,
                device="gpu",
                gpu_use_dp=False,
            )
            model.fit(X_train_vec, y_train)
            logging.info("LightGBM successfully trained using CUDA (GPU).")
            return model
        except Exception as e:
            logging.warning(f"CUDA unavailable or failed ({e}). Falling back to CPU.")
            model = lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                force_col_wise=True,
                device="cpu",
            )
            model.fit(X_train_vec, y_train)
            logging.info("LightGBM trained on CPU.")
            return model

    raise ValueError(f"Unknown model type: {model_type}")


def train(data_path: str, models_dir: str, model_type: str = "logistic") -> None:
    df = load_data(data_path)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logging.info("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    logging.info(f"Training model: {model_type}...")
    model = _train_model(model_type, X_train_vec, y_train)

    logging.info("Evaluating...")
    y_pred = model.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred)
    logging.info(f"F1 Score: {f1:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)

    # Print a readable report for local runs
    print(classification_report(y_test, y_pred))

    # Save artifacts
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    with open(models_path / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open(models_path / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metrics for reproducibility / reviewers
    metrics = {
        "model_type": model_type,
        "f1": float(f1),
        "support": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "classification_report": report,
    }

    with open(models_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logging.info(f"Artifacts saved to {models_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bangla Fake News Detector (BanFakeNews)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw",
        help="Path to directory containing the dataset CSV files",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained artifacts",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "svm", "lightgbm"],
        help="Model algorithm",
    )

    args = parser.parse_args()
    train(args.data_path, args.models_dir, args.model_type)
