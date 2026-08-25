# Viva Q&A Prep

> Started at planning. Fill in real numbers on Day 8 once results exist.
> Rule: never invent a number here. Blanks stay blank until a real run fills them.

## Methodology

**Q: Why walk-forward validation instead of a normal train/test split?**
Stock data is time-ordered. A shuffled split trains on future data and tests on past data, which leaks information and inflates the score. Walk-forward keeps every training fold strictly before its test fold — it simulates how the model would actually have been used. See ADR-0003.

**Q: How did you prevent data leakage?**
Three ways: (1) every feature at time *t* uses only data up to *t*, audited and written up in `leakage_audit.md`; (2) scaling happens inside a Pipeline so the scaler fits only on training folds; (3) targets are computed as forward-looking labels and never used as inputs.

**Q: How do you know your features are causing the prediction and not noise?**
Feature importance from tree models plus logistic regression coefficient signs. Cross-checked against the majority-class baseline — if a model can't beat that baseline, its "importance" scores mean nothing.

## Results

**Q: Your accuracy is only ~__%. Is that useful?**
The majority-class baseline is __%. A shuffled split would report a much higher number and be wrong. In financial prediction, a small consistent edge over baseline, surviving transaction costs, is the meaningful result — which is why the backtest matters more than the accuracy figure.

**Q: What does ROC-AUC actually mean here?**
The probability that a randomly chosen "up" day is ranked higher by the model than a randomly chosen "down" day. 0.5 is random. It matters more than accuracy because it evaluates the ranking across all thresholds, not one arbitrary cutoff.

**Q: Why did model X win?**
<fill Day 5 — cite mean ROC-AUC across folds, and note the tie-break rule favours the simpler model>

**Q: Did it beat buy-and-hold?**
<fill Day 5 — with and without transaction costs>

## Design Choices

**Q: Why no LSTM / deep learning?**
Dataset size is small for a sequence model; gradient-boosted trees match or beat deep models on tabular data at this scale; and interpretability was a requirement, since identifying *which* indicators carry signal is part of the contribution. Listed under future work. See ADR-0004.

**Q: Why two horizons?**
1-day captures short-term noise-dominated movement, 5-day captures a slightly stronger trend signal. Comparing them shows how predictability changes with horizon — a finding in itself.

**Q: Why separate the FastAPI service from Laravel?**
Model inference and business logic have different dependency trees and scaling profiles. Behind an HTTP boundary the model is a versioned, swappable dependency — retrainable without redeploying the web app. See ADR-0005.

## Limitations & Future Work

**Q: What are the risks of someone actually trading on this?**
Real ones. The model has no macro awareness, no news awareness, and no regime-change detection. Backtested performance is not a forward guarantee. It is a decision-support signal, not advice — stated explicitly in the model card and the UI.

**Q: What would you do with three more months?**
News sentiment features from Nepali financial media; regime detection so the model knows when it is out of distribution; sector-specific models; walk-forward retraining on a schedule; proper deployment with monitoring for feature drift.

**Q: Biggest challenge?**
<fill Day 8 — be specific and honest; data acquisition is the likely answer>

**Q: What would you do differently?**
<fill Day 8>
