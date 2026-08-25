# PLAN.md — 8 Working Days

**Working mode:** 9 hours/day, 8 days. Calendar allowance is 15 days — days 9–15 are **declared buffer**, not extra scope.

**Rule:** A day is COMPLETE only when every DoD item is checked. If a day ends PARTIAL, the unfinished items roll into the buffer — they do **not** get absorbed into the next day.

---

## Day 1 — Data Acquisition & Repo Scaffold

**Objective:** Get a frozen, documented NEPSE dataset on disk and the repo skeleton in place.

**Tasks**
- Initialize git repo, create the folder structure from CLAUDE.md §6
- Create `ml/requirements.txt`, set up virtualenv
- Choose a fixed basket of **8–12 liquid NEPSE stocks** across sectors (banking, hydro, insurance, microfinance) — record the list and the reason for each pick
- Acquire historical OHLCV data (target: 3+ years/stock). Sources in priority order: Sasto Share export → Mero Share → NEPSE site scrape → manual CSV
- Write `ml/src/data_loader.py` — loads raw CSVs into a tidy long-format DataFrame
- Save to `ml/data/raw/` and **freeze it** — never regenerate after today
- Write `ml/data/raw/README.md` documenting source, date pulled, columns, row counts, known gaps

**Definition of Done**
- [ ] Git repo initialized, structure matches CLAUDE.md §6
- [ ] Raw data on disk for every stock in the basket
- [ ] `data_loader.py` loads it into a single tidy DataFrame without errors
- [ ] Row counts, date ranges, and missing-value counts printed and recorded
- [ ] `data/raw/README.md` written
- [ ] Day 1 committed to git

**Hard cutoff:** If data acquisition is unsolved by hour 7, invoke the fallback (smaller basket, manual export) and write an ADR documenting the pivot. Do not spend Day 2 on this.

---

## Day 2 — Exploratory Data Analysis

**Objective:** Understand the data well enough to defend every claim about it.

**Tasks**
- Notebook `01_eda.ipynb`
- Data quality: missing values, trading-holiday gaps, duplicates, outliers, stock splits/bonus adjustments
- Price/volume trends per stock; sector comparison
- Return distribution — is it fat-tailed? skewed?
- Volatility over time; volatility clustering
- Correlation heatmap across stocks
- **Class balance check** for `target_1d` and `target_5d` — establishes the majority-class baseline
- Every chart gets a written interpretation cell beneath it

**Definition of Done**
- [ ] ≥6 distinct visualizations, each with written interpretation
- [ ] Missing-data strategy decided and documented (→ ADR if non-obvious)
- [ ] Class balance for both targets computed and recorded
- [ ] Majority-class baseline accuracy recorded — this is the number every model must beat
- [ ] 5 defense-worthy findings written into `docs/defense/eda_findings.md`
- [ ] Committed

---

## Day 3 — Feature Engineering

**Objective:** A leak-free, reusable feature pipeline.

**Tasks**
- Notebook `02_features.ipynb` → productionized into `ml/src/features.py`
- Technical indicators: RSI(14), MACD(12,26,9), SMA(5,10,20,50), EMA(12,26), Bollinger Bands(20,2), ATR
- Volume features: volume MA, volume ratio, OBV
- Price features: daily return, log return, high-low range, close-open gap
- Lag features (t-1..t-5) and rolling stats (mean, std over 5/10/20)
- Build `target_1d` and `target_5d`
- **Leakage audit:** confirm every feature at time `t` uses only data ≤ `t`. Write the audit down.
- Drop warm-up rows created by rolling windows

**Definition of Done**
- [ ] `features.py` is importable and produces a feature matrix from the raw loader output
- [ ] ≥25 engineered features
- [ ] Both targets built correctly (verified by manual spot-check on ≥3 rows)
- [ ] Written leakage audit in `docs/defense/leakage_audit.md`
- [ ] Processed dataset saved to `ml/data/processed/`
- [ ] Committed

---

## Day 4 — Modeling & Walk-Forward Validation

**Objective:** Three models trained and honestly compared.

**Tasks**
- `ml/src/train.py`
- Implement **walk-forward validation** (expanding window). Explicitly document why `train_test_split(shuffle=True)` is invalid here — this is a likely viva question.
- Train: Logistic Regression (baseline) → Random Forest → XGBoost
- Scaling in a `Pipeline` so the scaler is fit inside each fold, not on the full dataset
- Modest hyperparameter tuning (`GridSearchCV` with `TimeSeriesSplit`) — timebox to 2 hours
- Metrics per fold and averaged: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrices, ROC curves, feature importance plots

**Definition of Done**
- [ ] Walk-forward validation implemented and explained in code comments
- [ ] All 3 models trained, metrics recorded in a comparison table
- [ ] Every model compared against the majority-class baseline from Day 2
- [ ] Confusion matrix + ROC curve + feature importance exported to `ml/reports/`
- [ ] If any model exceeds ~70% accuracy: leakage investigation performed and documented
- [ ] Committed

---

## Day 5 — Backtest, Evaluation & Model Freeze

**Objective:** Prove the model means something economically, then lock it.

**Tasks**
- `ml/src/backtest.py` — simple long-only strategy: go long when P(up) > threshold, else hold cash
- Benchmark against buy-and-hold on the same basket
- Include realistic NEPSE transaction costs (broker commission + SEBON + DP charges)
- Equity curve chart, total return, max drawdown, win rate, number of trades
- Threshold sensitivity analysis (0.5 / 0.55 / 0.6)
- Select final model, retrain on full history, **freeze**: pickle model + scaler + feature list + metadata JSON to `ml/models/`
- Write `docs/defense/model_card.md` — what it does, what it doesn't, limitations

**Definition of Done**
- [ ] Backtest runs, produces equity curve vs. buy-and-hold
- [ ] Transaction costs included and their impact stated
- [ ] Final model selected with written justification
- [ ] Frozen artifacts in `ml/models/`: `model.pkl`, `scaler.pkl`, `features.json`, `metadata.json`
- [ ] `model_card.md` written
- [ ] **ML work is now closed.** No retraining after this point.
- [ ] Committed

---

## Day 6 — FastAPI Service & Laravel Backend

**Objective:** The frozen model reachable over HTTP, persisted by Laravel.

**Tasks**
- `service/`: FastAPI app loading frozen artifacts at startup
  - `GET /health`
  - `POST /predict` → `{symbol, horizon}` → `{direction, probability, model_version, as_of_date}`
  - Input validation with Pydantic
- `backend/`: Laravel 12
  - Migrations: `stocks`, `predictions`, users (default auth)
  - `PredictionService` — HTTP client calling FastAPI, handles timeout/failure gracefully
  - Endpoints: `GET /api/stocks`, `GET /api/stocks/{symbol}/history`, `POST /api/predict`
  - Seed `stocks` table from the Day 1 basket
  - Cache predictions so repeated requests don't re-hit the service

**Definition of Done**
- [ ] FastAPI serves a real prediction from the frozen model (verified via curl)
- [ ] Laravel migrations run clean
- [ ] Laravel `/api/predict` returns a prediction end-to-end
- [ ] Prediction persisted to DB
- [ ] Service-down path returns a clean error, not a 500 stack trace
- [ ] Committed

---

## Day 7 — React Frontend & Integration

**Objective:** A demoable dashboard.

**Tasks**
- `frontend/`: React app
  - Stock selector (from the basket)
  - Price history chart (Recharts) with an indicator overlay (e.g. SMA)
  - Prediction panel: direction, confidence %, horizon toggle (1d / 5d)
  - Model performance page: metrics table + confusion matrix + equity curve (static images from `ml/reports/` are fine)
  - Loading and error states
- Full integration test: React → Laravel → FastAPI → model → back
- **Capture screenshots of every screen** and record a 2–3 minute demo video

**Definition of Done**
- [ ] Dashboard runs and renders real data from the backend
- [ ] Prediction displays correctly for both horizons
- [ ] Performance page renders
- [ ] Error states don't crash the UI
- [ ] Screenshots saved to `docs/defense/screenshots/`
- [ ] Demo video recorded — this is the live-demo fallback
- [ ] Committed and tagged `v1.0`

---

## Day 8 — Report, Slides & Rehearsal

**Objective:** Be ready to walk in and defend.

**Tasks**
- `docs/defense/report.md` (→ export to PDF/DOCX) structured exactly to the required outline:
  1. **Core Problem** — objective, problem statement, tools, data collection & preprocessing
  2. **Analysis & Modeling** — EDA, visualization, algorithm selection, model development
  3. **Outcomes & Insights** — metrics, key findings, challenges, future improvements
- Slide deck (12–15 slides) mirroring the same three sections
- `docs/defense/viva_qa.md` — anticipated questions with prepared answers:
  - Why walk-forward instead of a random split?
  - Why XGBoost over Random Forest (or vice versa)?
  - Why is your accuracy only ~55%? Is that useful?
  - How did you prevent data leakage?
  - What does your ROC-AUC actually mean?
  - What would you do with 3 more months?
  - Why these features? Which mattered most?
  - What are the ethical/practical risks of deploying this?
- **Rehearse the full talk twice, out loud, timed.**

**Definition of Done**
- [ ] Report complete, covering all three required sections
- [ ] Slide deck complete
- [ ] ≥10 viva questions with written answers
- [ ] Full talk rehearsed twice, timed to fit the slot
- [ ] Backup materials ready (PDF on USB, screenshots, demo video)
- [ ] Final commit

---

## Days 9–15 — Buffer

Reserved for overflow only. Priority order if time remains:

1. Finish anything PARTIAL from Days 1–8
2. More rehearsal
3. Tighten the report
4. Polish the UI

**Do not use buffer days to add scope.** Parking Lot items stay parked and get presented as "future work."
