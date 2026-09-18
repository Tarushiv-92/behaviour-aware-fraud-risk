import pandas as pd

from src.config import FEATURE_COLS, TARGET, TIME_COL, risk_band
from src.make_dummy_features import generate
from src.train import compute_metrics, time_split


def test_risk_band():
    assert risk_band(0.2) == "LOW"
    assert risk_band(0.5) == "MEDIUM"
    assert risk_band(0.94) == "HIGH"


def test_dummy_has_contract_and_fraud():
    df = generate(n_customers=30, fraud_rate=0.03)
    for col in FEATURE_COLS + [TARGET, TIME_COL, "customer_id"]:
        assert col in df.columns
    assert df[TARGET].mean() > 0
    assert df[TARGET].mean() < 0.15


def test_time_split_order():
    df = pd.DataFrame(
        {
            TIME_COL: pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            TARGET: [0, 0, 1],
        }
    )
    train, test = time_split(df, train_frac=0.66)
    assert train[TIME_COL].max() <= test[TIME_COL].min()


def test_metrics_keys():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    for key in ("precision", "recall", "f1", "pr_auc", "confusion_matrix"):
        assert key in m
