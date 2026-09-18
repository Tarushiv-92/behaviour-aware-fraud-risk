"""Local reason codes for a scored transaction (SHAP when available)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, REASON_MAP, risk_band


def _shap_vector(shap_values: Any, n_features: int) -> np.ndarray:
    """Normalize shap output across shap/sklearn versions."""
    if isinstance(shap_values, list):
        arr = np.array(shap_values[1])
    else:
        arr = np.array(shap_values)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # (n_features, n_classes) or (1, n_features)
        if arr.shape[-1] == 2 and arr.shape[0] == n_features:
            arr = arr[:, 1]
        else:
            arr = arr[0]
    return np.asarray(arr, dtype=float).reshape(-1)[:n_features]


def reason_codes_from_contributions(
    contrib: pd.Series, top_k: int = 3
) -> tuple[list[str], dict[str, float]]:
    toward_fraud = contrib.sort_values(ascending=False)
    top = toward_fraud.head(top_k)
    reasons = [REASON_MAP.get(str(name), str(name)) for name, val in top.items() if val > 0]
    return reasons, {str(k): round(float(v), 4) for k, v in top.items()}


def score_one(feature_row: pd.DataFrame, model, explainer=None, top_k: int = 3) -> dict:
    """feature_row must be 1 row with FEATURE_COLS in training order."""
    row = feature_row[FEATURE_COLS]
    prob = float(model.predict_proba(row)[0, 1])

    if explainer is not None:
        raw = explainer.shap_values(row, check_additivity=False)
        values = _shap_vector(raw, len(FEATURE_COLS))
        contrib = pd.Series(values, index=FEATURE_COLS)
    else:
        # Fallback: RF impurity importance * feature value (not SHAP, still demo-able)
        importances = getattr(model, "feature_importances_", np.ones(len(FEATURE_COLS)))
        contrib = pd.Series(importances * row.iloc[0].to_numpy(dtype=float), index=FEATURE_COLS)

    reasons, shap_top = reason_codes_from_contributions(contrib, top_k=top_k)
    return {
        "risk_score": round(prob, 4),
        "risk_band": risk_band(prob),
        "reason_codes": reasons,
        "contributions": shap_top,
    }


def try_tree_explainer(model):
    try:
        import shap

        return shap.TreeExplainer(model)
    except Exception:
        return None


def global_importance(explainer, X: pd.DataFrame) -> dict[str, float]:
    raw = explainer.shap_values(X, check_additivity=False)
    values = np.array(raw)
    if values.ndim == 3:
        values = values[1] if values.shape[0] == 2 else values[:, :, 1]
    mean_abs = np.mean(np.abs(np.squeeze(values)), axis=0)
    return {
        FEATURE_COLS[i]: round(float(mean_abs[i]), 6)
        for i in range(min(len(FEATURE_COLS), len(mean_abs)))
    }


def main() -> None:
    import json

    import joblib
    import pandas as pd

    from src.config import (
        DATA_FEATURES,
        FEATURE_COLS,
        RF_PATH,
        SHAP_EXAMPLES_PATH,
        TARGET,
        TIME_COL,
    )
    from src.train import time_split

    df = pd.read_csv(DATA_FEATURES, parse_dates=[TIME_COL])
    _, test = time_split(df)
    model = joblib.load(RF_PATH)
    explainer = try_tree_explainer(model)
    X = test[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    test = test.copy()
    test["risk_score"] = probs

    examples = []
    ranked = test.sort_values("risk_score", ascending=False)
    for _, row in ranked.head(5).iterrows():
        feat = pd.DataFrame([row[FEATURE_COLS]])
        payload = score_one(feat, model, explainer)
        payload["transaction_id"] = row.get("transaction_id")
        payload["customer_id"] = row.get("customer_id")
        payload["is_fraud"] = int(row[TARGET])
        examples.append(payload)

    global_shap = None
    if explainer is not None:
        sample = X.sample(n=min(400, len(X)), random_state=42)
        global_shap = dict(
            sorted(global_importance(explainer, sample).items(), key=lambda kv: -kv[1])
        )

    SHAP_EXAMPLES_PATH.write_text(
        json.dumps(
            {
                "local_examples": examples,
                "global_mean_abs_shap": global_shap,
                "explainer": "TreeExplainer" if explainer is not None else "fallback",
            },
            indent=2,
            default=str,
        )
    )
    print(json.dumps(json.loads(SHAP_EXAMPLES_PATH.read_text()), indent=2))


if __name__ == "__main__":
    main()
