import argparse
import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    logging.info(f"Loading data from {data_path}...")
    try:
        auth_path = os.path.join(data_path, 'Authentic-48K.csv')
        fake_path = os.path.join(data_path, 'Fake-1K.csv')
        
        auth = pd.read_csv(auth_path)
        fake = pd.read_csv(fake_path)
        
        auth['label'] = 0
        fake['label'] = 1
        
        df = pd.concat([auth, fake], ignore_index=True)
        # Drop duplicates based on content
        df.drop_duplicates(subset=['content'], inplace=True)
        
        # Robust basic cleaning
        df['content'] = df['content'].fillna('').astype(str)
        df = df[df['content'].str.strip() != '']
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train(data_path, models_dir, model_type='logistic'):
    df = load_data(data_path)
    
    X = df['content']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    logging.info(f"Training model: {model_type}...")
    if model_type == 'logistic':
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
    elif model_type == 'svm':
        # Calibrated for predict_proba
        base_svm = LinearSVC(C=1.0, dual='auto', random_state=42)
        model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
        model.fit(X_train_vec, y_train)
    elif model_type == 'lightgbm':
        # Attempt GPU if possible, else CPU
        try:
             model = lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True, device='gpu')
             model.fit(X_train_vec, y_train)
             logging.info("Trained with GPU.")
        except Exception as e:
             logging.warning(f"GPU failed: {e}. Retrying on CPU.")
             model = lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True, device='cpu')
             model.fit(X_train_vec, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    logging.info("Evaluating...")
    y_pred = model.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred)
    logging.info(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
        
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    logging.info(f"Artifacts saved to {models_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bangla Fake News Detector')
    parser.add_argument('--data_path', type=str, default='data/raw', help='Path to raw CSV files')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save artifacts')
    parser.add_argument('--model_type', type=str, default='logistic', choices=['logistic', 'svm', 'lightgbm'], help='Model algorithm')
    
    args = parser.parse_args()
    
    train(args.data_path, args.models_dir, args.model_type)
