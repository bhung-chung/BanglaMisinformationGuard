import argparse
import os
import pickle
import numpy as np


def load_artifacts(models_dir: str):
    """Load TF-IDF vectorizer and trained model."""
    try:
        with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        print(f"Artifacts not found in '{models_dir}'. Please train first (python src/train.py ...).")
        return None, None


def get_risk_label(prob_fake: float, suspicious_threshold: float = 0.33, fake_threshold: float = 0.66) -> str:
    """Convert a fake-probability score into a human-friendly risk label.

    Defaults match the original rubric:
      - prob < 0.33  -> Safe
      - 0.33..0.66   -> Suspicious
      - > 0.66       -> Fake

    For highly imbalanced datasets, you may want lower thresholds.
    """
    if prob_fake < suspicious_threshold:
        return "Safe"
    if prob_fake <= fake_threshold:
        return "Suspicious"
    return "Fake"


def _probability_of_fake(model, X_vec) -> float:
    """Return P(fake=1) robustly for different sklearn/lightgbm estimators."""

    # Preferred: predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)
        # proba shape: (n, 2) for binary
        return float(proba[0][1])

    # Some models expose decision_function but not probabilities
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_vec)
        # score can be scalar or array
        score = float(np.ravel(score)[0])
        # convert margin to probability via sigmoid
        prob = 1.0 / (1.0 + np.exp(-score))
        return float(prob)

    raise TypeError(
        "Model does not support predict_proba or decision_function. "
        "Use LogisticRegression, CalibratedClassifierCV(LinearSVC), or LightGBM."
    )


def predict(
    text: str,
    vectorizer,
    model,
    suspicious_threshold: float = 0.33,
    fake_threshold: float = 0.66,
):
    X_vec = vectorizer.transform([str(text)])
    prob_fake = _probability_of_fake(model, X_vec)
    label = get_risk_label(prob_fake, suspicious_threshold=suspicious_threshold, fake_threshold=fake_threshold)
    return prob_fake, label


def main():
    parser = argparse.ArgumentParser(description="Predict whether a Bangla news text is fake")
    parser.add_argument("text", type=str, help="Text to classify (wrap in quotes)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing .pkl artifacts")

    # Optional decision thresholds
    parser.add_argument(
        "--suspicious_threshold",
        type=float,
        default=0.33,
        help="If P(fake) >= this value, label becomes at least 'Suspicious' (default: 0.33)",
    )
    parser.add_argument(
        "--fake_threshold",
        type=float,
        default=0.66,
        help="If P(fake) > this value, label becomes 'Fake' (default: 0.66)",
    )

    args = parser.parse_args()

    tfidf, model = load_artifacts(args.models_dir)
    if tfidf is None or model is None:
        raise SystemExit(1)

        # Guard rails
    if not (0.0 <= args.suspicious_threshold <= 1.0 and 0.0 <= args.fake_threshold <= 1.0):
        raise SystemExit("Thresholds must be within [0, 1].")
    if args.suspicious_threshold > args.fake_threshold:
        raise SystemExit("suspicious_threshold must be <= fake_threshold")

    prob, label = predict(
        args.text,
        tfidf,
        model,
        suspicious_threshold=args.suspicious_threshold,
        fake_threshold=args.fake_threshold,
    )

    preview = args.text.replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."

    print(f"Text: {preview}")
    print(f"Fake Probability: {prob:.4f}")
    print(f"Risk Label: {label}")


if __name__ == "__main__":
    main()
