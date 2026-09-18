"""Synthetic Person-2 output so Person 3 can train before real features land.

Generates personas, genuine history, then injects behavioural fraud.
Features are computed from *prior* transactions of the same customer
(no future leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_FEATURES, DATA_RAW, FEATURE_COLS, TARGET, TIME_COL


rng = np.random.default_rng(42)

PERSONAS = {
    "low_evening_shopper": {
        "mu_amount": 700,
        "sigma_amount": 180,
        "hour_mu": 20,
        "hour_sigma": 1.5,
        "txns_per_week": 4,
        "merchants": ["BigBazaar", "Swiggy", "Amazon", "LocalKirana"],
        "beneficiaries": ["self_upi", "friend_a"],
    },
    "high_business": {
        "mu_amount": 12000,
        "sigma_amount": 2500,
        "hour_mu": 11,
        "hour_sigma": 2.0,
        "txns_per_week": 10,
        "merchants": ["OfficeSupplies", "AWS", "Indigo", "TajHotels"],
        "beneficiaries": ["vendor_x", "payroll", "self_neft"],
    },
    "mid_weekend": {
        "mu_amount": 2500,
        "sigma_amount": 600,
        "hour_mu": 15,
        "hour_sigma": 3.0,
        "txns_per_week": 6,
        "merchants": ["Zomato", "Myntra", "BookMyShow", "IRCTC"],
        "beneficiaries": ["family_upi", "self_upi"],
    },
}


def _clip_hour(h: float) -> int:
    return int(np.clip(round(h), 0, 23))


def _sample_genuine(persona: dict, n: int, start: pd.Timestamp) -> list[dict]:
    rows = []
    last = start
    known_m, known_b = set(), set()
    amounts = []
    for i in range(n):
        gap = max(0.05, rng.exponential(7 / persona["txns_per_week"]))
        ts = last + pd.Timedelta(days=float(gap))
        hour = _clip_hour(rng.normal(persona["hour_mu"], persona["hour_sigma"]))
        ts = ts.normalize() + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
        amount = float(np.clip(rng.normal(persona["mu_amount"], persona["sigma_amount"]), 50, None))
        merchant = str(rng.choice(persona["merchants"]))
        beneficiary = str(rng.choice(persona["beneficiaries"]))
        # Genuine but messy: festival spend, new shop, slightly odd hour
        if rng.random() < 0.06:
            amount *= float(rng.uniform(2.2, 4.0))
        if rng.random() < 0.05:
            merchant = f"NewShop_{int(rng.integers(1, 40))}"
        if rng.random() < 0.04:
            hour = int(rng.integers(0, 24))
            ts = ts.normalize() + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
        is_new_m = int(merchant not in known_m)
        is_new_b = int(beneficiary not in known_b)
        known_m.add(merchant)
        known_b.add(beneficiary)
        mu = float(np.mean(amounts)) if amounts else persona["mu_amount"]
        sd = float(np.std(amounts, ddof=0)) if len(amounts) > 2 else persona["sigma_amount"]
        sd = max(sd, 1.0)
        velocity = int(sum(1 for r in rows[-8:] if (ts - r["timestamp"]) <= pd.Timedelta(hours=1)))
        days_since = (ts - last).total_seconds() / 86400 if i else 0.0
        unusual = int(abs(hour - persona["hour_mu"]) > 3 * persona["hour_sigma"])
        rows.append(
            {
                TIME_COL: ts,
                "amount": round(amount, 2),
                "merchant": merchant,
                "beneficiary": beneficiary,
                "hour": hour,
                "amount_ratio": amount / mu,
                "amount_zscore": (amount - mu) / sd,
                "is_unusual_hour": unusual,
                "is_new_merchant": is_new_m,
                "is_new_beneficiary": is_new_b,
                "transaction_velocity": velocity,
                "days_since_last_transaction": round(days_since, 3),
                TARGET: 0,
            }
        )
        amounts.append(amount)
        last = ts
    return rows


def _inject_fraud(customer_id: str, persona: dict, history: list[dict]) -> dict:
    """Fraud styles are mixed so the label is not a perfect rule."""
    cut = int(rng.integers(8, max(9, len(history) - 1)))
    prior = history[:cut]
    last = prior[-1]
    mu = float(np.mean([r["amount"] for r in prior]))
    sd = max(float(np.std([r["amount"] for r in prior], ddof=0)), 1.0)
    style = str(rng.choice(["obvious", "amount", "night", "velocity", "new_payee"]))

    amount = float(np.clip(rng.normal(persona["mu_amount"], persona["sigma_amount"]), 50, None))
    hour = _clip_hour(rng.normal(persona["hour_mu"], persona["hour_sigma"]))
    merchant = str(rng.choice(persona["merchants"]))
    beneficiary = str(rng.choice(persona["beneficiaries"]))
    velocity = int(rng.integers(0, 3))
    known_m = {r["merchant"] for r in prior}
    known_b = {r["beneficiary"] for r in prior}

    if style == "obvious":
        amount = float(mu * rng.uniform(5.0, 9.0))
        hour = int(rng.choice([1, 2, 3, 4]))
        merchant = f"UnknownStore_{int(rng.integers(1, 80))}"
        beneficiary = "new_wallet_" + str(rng.integers(1000, 9999))
        velocity = int(rng.integers(4, 8))
    elif style == "amount":
        amount = float(mu * rng.uniform(3.5, 6.5))
    elif style == "night":
        hour = int(rng.choice([1, 2, 3, 4]))
        amount = float(mu * rng.uniform(1.8, 3.2))
        merchant = f"NightPOS_{int(rng.integers(1, 40))}"
    elif style == "velocity":
        velocity = int(rng.integers(5, 9))
        amount = float(mu * rng.uniform(1.2, 2.4))
        beneficiary = "quick_payee_" + str(rng.integers(100, 999))
    else:
        beneficiary = "new_wallet_" + str(rng.integers(1000, 9999))
        merchant = f"P2P_{int(rng.integers(1, 50))}"
        amount = float(mu * rng.uniform(2.0, 4.5))

    ts = last["timestamp"] + pd.Timedelta(
        hours=float(rng.uniform(0.5, 36)), minutes=int(rng.integers(0, 60))
    )
    ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)), second=0)
    if ts <= last["timestamp"]:
        ts += pd.Timedelta(days=1)

    unusual = int(abs(hour - persona["hour_mu"]) > 3 * persona["hour_sigma"])
    return {
        TIME_COL: ts,
        "amount": round(amount, 2),
        "merchant": merchant,
        "beneficiary": beneficiary,
        "hour": hour,
        "amount_ratio": amount / mu,
        "amount_zscore": (amount - mu) / sd,
        "is_unusual_hour": unusual,
        "is_new_merchant": int(merchant not in known_m),
        "is_new_beneficiary": int(beneficiary not in known_b),
        "transaction_velocity": velocity,
        "days_since_last_transaction": round((ts - last["timestamp"]).total_seconds() / 86400, 3),
        TARGET: 1,
        "customer_id": customer_id,
        "persona": last["persona"],
    }


def generate(n_customers: int = 180, fraud_rate: float = 0.02) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01")
    personas = list(PERSONAS.keys())
    all_rows: list[dict] = []
    txn_id = 1
    for i in range(n_customers):
        name = personas[i % len(personas)]
        persona = PERSONAS[name]
        cid = f"C{i:04d}"
        n = int(rng.integers(18, 55))
        hist = _sample_genuine(
            persona, n, start + pd.Timedelta(days=int(rng.integers(0, 20)))
        )
        for row in hist:
            row["customer_id"] = cid
            row["persona"] = name
            row["transaction_id"] = f"T{txn_id:07d}"
            txn_id += 1
        all_rows.extend(hist)

    df = pd.DataFrame(all_rows)
    n_fraud = max(1, int(len(df) * fraud_rate / (1 - fraud_rate)))
    victims = rng.choice(df["customer_id"].unique(), size=n_fraud, replace=True)
    fraud_rows = []
    for cid in victims:
        hist = df[df["customer_id"] == cid].sort_values(TIME_COL).to_dict("records")
        fraud = _inject_fraud(cid, PERSONAS[hist[-1]["persona"]], hist)
        fraud["transaction_id"] = f"T{txn_id:07d}"
        txn_id += 1
        fraud_rows.append(fraud)
    df = pd.concat([df, pd.DataFrame(fraud_rows)], ignore_index=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def main() -> None:
    df = generate()
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    DATA_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    raw_cols = [
        "transaction_id",
        "customer_id",
        "persona",
        TIME_COL,
        "amount",
        "merchant",
        "beneficiary",
        TARGET,
    ]
    df[raw_cols].to_csv(DATA_RAW, index=False)
    keep = ["transaction_id", "customer_id", "persona", TIME_COL] + FEATURE_COLS + [TARGET]
    df[keep].to_csv(DATA_FEATURES, index=False)
    print(f"Wrote {DATA_RAW} ({len(df)} rows)")
    print(f"Wrote {DATA_FEATURES}")
    print(f"Fraud rate: {df[TARGET].mean():.3%}")


if __name__ == "__main__":
    main()
