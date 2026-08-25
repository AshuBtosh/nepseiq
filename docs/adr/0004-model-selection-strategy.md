# ADR-0004: Model Selection Strategy

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Planning

## Context
Model choice must be defensible, not fashionable. The panel grades technical understanding — a complex model that cannot be explained scores worse than a simple one that can.

## Decision
Train and compare exactly three models:
1. **Logistic Regression** — interpretable baseline, gives coefficient direction
2. **Random Forest** — non-linear, robust, provides feature importance
3. **XGBoost** — gradient boosting, typically strongest on tabular data

Selection criterion: mean ROC-AUC across walk-forward folds, with accuracy, precision, recall, and F1 reported alongside. Ties broken in favour of the simpler model.

**Explicitly excluded:** LSTM, GRU, Transformers, and any deep learning approach.

## Alternatives Considered
| Option | Why rejected |
|---|---|
| LSTM / deep learning | High time cost, needs far more data, near-impossible to interpret for feature-importance discussion, marginal gain on tabular financial data |
| SVM | Poor scaling, weak interpretability, no feature importance |
| Naive Bayes | Feature-independence assumption badly violated by correlated technical indicators |
| Single model only | Loses the comparative analysis the brief asks for ("algorithm selection") |

## Consequences
**Positive:** Three models give a real comparison narrative. All three produce interpretable feature importance or coefficients. All are fully explainable in a viva.

**Negative:** Cannot claim a "deep learning project." Framed as a deliberate, justified choice, not a limitation.

## Defense Note
"Why no LSTM?" — three reasons: (1) dataset size is small for a sequence model; (2) gradient-boosted trees consistently match or beat deep models on tabular data of this scale; (3) interpretability was a project requirement, since explaining *which* indicators carry signal is part of the contribution. Listed under future work.
