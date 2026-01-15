import pickle
import os
import argparse
import numpy as np

def load_artifacts(models_dir):
    try:
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        print(f"Artifacts not found in {models_dir}. Please train first.")
        return None, None

def get_risk_label(prob):
    # Fake label is 1. Probability of being Fake.
    if prob < 0.33:
        return "Safe"
    elif prob <= 0.66:
        return "Suspicious"
    else:
        return "Fake"

def predict(text, vectorizer, model):
    processed = vectorizer.transform([str(text)])
    prob_fake = model.predict_proba(processed)[0][1] # Probability of class 1 (Fake)
    
    label = get_risk_label(prob_fake)
    return prob_fake, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--models_dir', type=str, default='models')
    
    args = parser.parse_args()
    
    tfidf, clf = load_artifacts(args.models_dir)
    
    if tfidf and clf:
        prob, label = predict(args.text, tfidf, clf)
        print(f"Text: {args.text[:50]}...")
        print(f"Fake Probability: {prob:.4f}")
        print(f"Risk Label: {label}")
