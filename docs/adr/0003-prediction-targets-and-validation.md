# ADR-0003: Prediction Targets and Validation Strategy

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Planning

## Context
Financial time-series data violates the i.i.d. assumption underlying standard cross-validation. A random shuffled train/test split trains on future data and tests on past data, producing inflated scores that collapse in reality. This is the single most common flaw in student stock-prediction projects and the most likely thing an evaluation panel will probe.

## Decision
1. **Targets:** binary directional classification at two horizons — `target_1d` and `target_5d`. Nothing else.
2. **Validation:** expanding-window **walk-forward validation**. Every train fold strictly precedes its test fold in time.
3. **Scaling** happens inside a Scikit-learn `Pipeline` so the scaler is fit only on each fold's training data.
4. **Hyperparameter search** uses `TimeSeriesSplit`, never `KFold`.
5. Every model is reported against the **majority-class baseline**.
6. A written **leakage audit** is produced on Day 3 (`docs/defense/leakage_audit.md`).

## Alternatives Considered
| Option | Why rejected |
|---|---|
| `train_test_split(shuffle=True)` | Look-ahead bias — invalid for time series |
| Standard k-fold cross-validation | Same problem |
| Single fixed train/test cut | Valid but gives one estimate with no variance information |
| Price regression targets | Noisier, harder to evaluate meaningfully, weaker story |

## Consequences
**Positive:** Results are trustworthy. Directly answers the hardest likely viva question. Multiple folds give a variance estimate, not a single lucky number.

**Negative:** Lower headline accuracy than a leaky split would show. More implementation work. Accepted deliberately.

## Defense Note
If accuracy is questioned as low: a shuffled split on this data would report a much higher number and be **wrong**. Walk-forward validation is what an honest result looks like. State the majority-class baseline and show the model's margin over it.
