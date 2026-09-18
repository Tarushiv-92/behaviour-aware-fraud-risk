# Behaviour-Aware Transaction Fraud Risk

AIML GLA Bootcamp 2026 — Problem 11.

Customer-relative fraud risk: learn each customer's normal behaviour, score how far a new transaction deviates, then explain the score to an analyst.

```text
Raw txn → customer baseline → deviation features → ML risk score → reason codes → workbench
```

## Why this is not "large amount = fraud"

₹8,000 is normal for a business traveller and suspicious for someone who usually spends ₹500–₹1,000 at 7 PM. The model sees **deviation features**, not a global rupee cutoff.

## Repo layout

| Path | Owner |
|---|---|
| `src/data_generation.py` | Person 1 |
| `src/feature_engineering.py`, notebooks 01–02 | Person 2 |
| `src/train.py`, `evaluate.py`, `explain.py`, notebook 03 | **Person 3 (you)** |
| `api/main.py` | Person 4 |
| `dashboard/app.py` | Person 5 |

Git: do not work on `main`. Person 3 branch: `feature/model`.

## Person 3 — run this first

Dummy `features.csv` is included so you can train before Person 2 finishes. Regenerate anytime:

```bash
cd behaviour-aware-fraud-risk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.make_dummy_features
python -m src.train
python -m src.evaluate
pytest -q
```

Artifacts:

- `models/logistic_regression.pkl` — baseline
- `models/random_forest.pkl` — main model
- `models/feature_list.json` — column order for Person 4
- `models/metrics.json` — precision / recall / F1 / PR-AUC
- `models/pr_curve.png`, `models/confusion_matrix.png`

Open `notebooks/03_model_evaluation.ipynb` for the demo narrative.

### Metrics you defend

Fraud is ~2% of rows. **Do not brag about accuracy.** Use Precision, Recall, F1, PR-AUC, confusion matrix.

Split is **time-based** (earlier transactions train, later test) to reduce leakage.

### Contract with Person 4

`POST /score` must rebuild the same `FEATURE_COLS` as `src/config.py`, then `predict_proba` + `src.explain.score_one`.

Risk bands: LOW `< 0.40`, MEDIUM `0.40–0.70`, HIGH `≥ 0.70`.

## Judge lines

- Approach: customer-relative baselines → deviation features → risk score.
- Metrics: imbalance makes accuracy misleading.
- Explainability: transaction-level reason codes (SHAP when installed).

## Optional stretch (after core works)

Review-capacity threshold, cost-sensitive analysis, SHAP waterfall on the dashboard.
