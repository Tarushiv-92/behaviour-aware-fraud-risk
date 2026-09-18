"""Shared paths, feature contract, and risk-band rules.

Person 3, 4, and 5 must keep FEATURE_COLS in the same order.
When Person 2 ships real features.csv, update this list to match.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
DATA_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
LR_PATH = MODELS_DIR / "logistic_regression.pkl"
RF_PATH = MODELS_DIR / "random_forest.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
SHAP_EXAMPLES_PATH = MODELS_DIR / "shap_examples.json"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"

TIME_COL = "timestamp"
TARGET = "is_fraud"
ID_COLS = ["transaction_id", "customer_id"]

# Customer-relative deviation features (Person 2 contract)
FEATURE_COLS = [
    "amount_ratio",
    "amount_zscore",
    "is_unusual_hour",
    "is_new_merchant",
    "is_new_beneficiary",
    "transaction_velocity",
    "days_since_last_transaction",
    "hour",
    "amount",
]

REASON_MAP = {
    "amount_ratio": "Amount much higher than this customer's normal spend",
    "amount_zscore": "Amount is a statistical outlier vs the customer's history",
    "is_unusual_hour": "Unusual hour compared with the customer's typical times",
    "is_new_merchant": "Merchant not seen in this customer's history",
    "is_new_beneficiary": "Beneficiary not seen in this customer's history",
    "transaction_velocity": "Unusually high number of transactions in a short window",
    "days_since_last_transaction": "Abnormal gap versus the customer's typical activity",
    "hour": "Transaction hour is atypical for this customer",
    "amount": "Raw amount contributed to the risk score",
}

LOW_MAX = 0.40
MEDIUM_MAX = 0.70


def risk_band(score: float) -> str:
    if score >= MEDIUM_MAX:
        return "HIGH"
    if score >= LOW_MAX:
        return "MEDIUM"
    return "LOW"
