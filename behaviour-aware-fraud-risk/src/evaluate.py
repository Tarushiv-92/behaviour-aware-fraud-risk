"""PR curve, confusion matrix plots, and metric dump for the demo."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.config import (
    DATA_FEATURES,
    FEATURE_COLS,
    MODELS_DIR,
    RF_PATH,
    TARGET,
    THRESHOLD_PATH,
    TIME_COL,
)
from src.train import compute_metrics, time_split


def load_threshold(default: float = 0.5) -> float:
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return default


def main() -> None:
    df = pd.read_csv(DATA_FEATURES, parse_dates=[TIME_COL])
    _, test = time_split(df)
    X, y = test[FEATURE_COLS], test[TARGET].astype(int)
    rf = joblib.load(RF_PATH)
    prob = rf.predict_proba(X)[:, 1]
    threshold = load_threshold()
    report = compute_metrics(y, prob, threshold=threshold)
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, prob, ax=ax, name="Random Forest")
    ax.set_title("Precision–Recall (use this, not accuracy)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, (prob >= threshold).astype(int), ax=ax)
    ax.set_title(f"Confusion matrix @ {threshold}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {MODELS_DIR / 'pr_curve.png'}")
    print(f"Wrote {MODELS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
