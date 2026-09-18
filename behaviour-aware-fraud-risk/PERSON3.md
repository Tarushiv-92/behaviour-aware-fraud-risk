# Person 3 — ML Engineer

You convert Person 2's behavioural features into an explainable fraud **risk score**.

## Do

1. Stay on `feature/model`.
2. Train Logistic Regression (baseline) and Random Forest (main).
3. Use `class_weight="balanced"`. Optional SMOTE on **train only**.
4. Report Precision, Recall, F1, PR-AUC — not accuracy.
5. Hand Person 4: `.pkl` + `feature_list.json` + `explain.score_one`.

## Do not

- Random row shuffle if timestamps exist.
- Fit SMOTE on the test set.
- Headline 99% accuracy.
- Skip the LR baseline.

## Commands

```bash
python -m src.make_dummy_features
python -m src.train
python -m src.evaluate
```
