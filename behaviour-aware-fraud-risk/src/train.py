"""Train LR baseline + Random Forest. Time-based split. Imbalance-aware metrics."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_FEATURES,
    FEATURE_COLS,
    FEATURE_LIST_PATH,
    LR_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RF_PATH,
    TARGET,
    THRESHOLD_PATH,
    TIME_COL,
)


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COL)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
        "n_positive": int(y_true.sum()),
        "n_rows": int(len(y_true)),
    }


def choose_threshold(y_true, y_prob) -> float:
    """Pick a probability cutoff on TRAIN only (max F1). Never tune on test."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.20, 0.80, 25):
        f1 = compute_metrics(y_true, y_prob, threshold=float(t))["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 4)


def train_rf_smote(X_train, y_train):
    """Optional: oversample the minority class on TRAIN only."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def build_models() -> tuple[Pipeline, RandomForestClassifier]:
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return lr, rf


def main() -> None:
    if not DATA_FEATURES.exists():
        raise SystemExit(
            f"Missing {DATA_FEATURES}. Run: python -m src.make_dummy_features"
        )

    df = pd.read_csv(DATA_FEATURES, parse_dates=[TIME_COL])
    missing = [c for c in FEATURE_COLS + [TARGET, TIME_COL] if c not in df.columns]
    if missing:
        raise SystemExit(f"features.csv missing columns: {missing}")

    train, test = time_split(df)
    X_train, y_train = train[FEATURE_COLS], train[TARGET].astype(int)
    X_test, y_test = test[FEATURE_COLS], test[TARGET].astype(int)

    print(
        f"Label rate overall={df[TARGET].mean():.2%} "
        f"(train={y_train.mean():.2%}, test={y_test.mean():.2%})"
    )
    if not (0.005 <= float(df[TARGET].mean()) <= 0.08):
        print("WARNING: fraud rate is outside the ~1–3% target band.")

    lr, rf = build_models()
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    rf_train_prob = rf.predict_proba(X_train)[:, 1]
    lr_test_prob = lr.predict_proba(X_test)[:, 1]
    rf_test_prob = rf.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_train, rf_train_prob)

    smote_report = None
    smote_pr_auc = None
    try:
        rf_smote = train_rf_smote(X_train, y_train)
        smote_prob = rf_smote.predict_proba(X_test)[:, 1]
        smote_report = compute_metrics(y_test, smote_prob, threshold=threshold)
        smote_pr_auc = smote_report["pr_auc"]
        joblib.dump(rf_smote, MODELS_DIR / "random_forest_smote.pkl")
    except Exception as exc:
        smote_report = {"error": str(exc)}

    rf_metrics = compute_metrics(y_test, rf_test_prob, threshold=threshold)
    production = "random_forest_balanced"
    if isinstance(smote_pr_auc, float) and smote_pr_auc > rf_metrics["pr_auc"] + 0.01:
        production = "random_forest_smote"

    report = {
        "note": "Do not headline accuracy. Fraud is rare; use precision, recall, F1, PR-AUC.",
        "data_source": str(DATA_FEATURES),
        "person2_status": "blocked — training on dummy Person-2 contract features",
        "split": "time-based 80/20 (earlier txns train, later txns test)",
        "fraud_rate_overall": float(df[TARGET].mean()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": FEATURE_COLS,
        "threshold": threshold,
        "threshold_rule": "max F1 on train probabilities",
        "production_model": production,
        "logistic_regression": compute_metrics(y_test, lr_test_prob, threshold=threshold),
        "random_forest": rf_metrics,
        "random_forest_smote": smote_report,
        "rf_feature_importance": dict(
            zip(FEATURE_COLS, [float(x) for x in rf.feature_importances_])
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, LR_PATH)
    joblib.dump(rf, RF_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(FEATURE_COLS, indent=2))
    THRESHOLD_PATH.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "risk_bands": {"LOW": "<0.40", "MEDIUM": "0.40-0.70", "HIGH": ">=0.70"},
                "production_model": production,
            },
            indent=2,
        )
    )
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved {LR_PATH.name}, {RF_PATH.name}, {THRESHOLD_PATH.name}")


if __name__ == "__main__":
    main()
