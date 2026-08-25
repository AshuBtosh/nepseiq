# CLAUDE.md — Project Constitution

> **This file is authoritative.** If anything in a chat session contradicts this file, this file wins.
> Claude must read this file, `PLAN.md`, and all files in `docs/adr/` before doing any work in a new session.

---

## 1. Project Identity

**Name:** NepseIQ — NEPSE Market Intelligence Platform
**Type:** Capstone project for Data Science & Machine Learning Diploma
**Institution:** Skill Shikshya, Sankhamul, Kathmandu
**Author:** Ashutosh
**Deliverable:** Working ML system + full-stack app + written report + defense presentation

---

## 2. The Actual Goal

Pass the Project Defense & Certification panel. The panel grades three things:

1. **Project** — does it work, is it non-trivial
2. **Technical understanding** — can you defend *why* you made each choice
3. **Presentation skills** — clarity of the story

**Implication that must drive every decision:** A simpler model you can *fully explain* beats a complex model you cannot. Depth of reasoning > breadth of features.

### Required presentation outline (from the institution's email — non-negotiable)

| Section | Must cover |
|---|---|
| **Core Problem** | Objective, problem statement, tools used (Python, Pandas, Scikit-learn, SQL), data collection & preprocessing |
| **Analysis & Modeling** | EDA, visualization, algorithm selection, model development |
| **Outcomes & Insights** | Evaluation metrics, key findings, implementation challenges, future improvements |

---

## 3. Problem Statement (locked)

> Retail investors on the Nepal Stock Exchange (NEPSE) make decisions with limited analytical tooling. NepseIQ applies supervised machine learning to engineered technical indicators from NEPSE historical price data to predict short-horizon directional price movement, and exposes those predictions through a web application with honest, backtested performance reporting.

**Two prediction targets, no more:**
- `target_1d` — will the closing price be higher 1 trading day from now? (binary)
- `target_5d` — will the closing price be higher 5 trading days from now? (binary)

---

## 4. Scope

### IN SCOPE (build this)
- NEPSE historical OHLCV data collection for a fixed basket of liquid stocks
- Technical indicator feature engineering (RSI, MACD, SMA/EMA, Bollinger Bands, volume features, lag/rolling features)
- Exploratory data analysis with visualizations
- Baseline model: Logistic Regression
- Advanced models: Random Forest, XGBoost
- **Walk-forward (time-series) validation** — never a random train/test split
- Backtest vs. buy-and-hold benchmark
- FastAPI microservice serving the frozen model
- Laravel backend (auth, stock records, prediction persistence, API orchestration)
- React frontend (symbol search, price chart, prediction + confidence, backtest view)
- Written report + slide deck + rehearsed defense

### OUT OF SCOPE (do not build — say "future work" in the defense)
- LSTM / deep learning models
- News sentiment analysis
- Portfolio optimization, watchlists, alerts, notifications
- Real-time / live market data streaming
- Multi-user roles, admin panels, billing
- Mobile app
- Docker/Kubernetes/CI-CD pipelines
- Any broker integration or actual order placement

**Anything not in the IN SCOPE list goes to the Parking Lot in `PROGRESS.md`. It does not get built.**

---

## 5. Tech Stack (locked — see ADR-0002)

| Layer | Choice |
|---|---|
| ML | Python 3.11, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| Notebooks | Jupyter |
| Model serving | FastAPI + Uvicorn |
| Backend | Laravel 12 |
| Database | MySQL |
| Frontend | React + Recharts |
| Version control | Git |

---

## 6. Repository Structure

```
nepseiq/
├── CLAUDE.md               # this file — the constitution
├── PLAN.md                 # day-by-day plan + Definition of Done
├── PROGRESS.md             # running log, updated at the END of every session
├── SESSION_PROMPTS.md      # copy-paste start & end prompts
├── docs/
│   ├── adr/                # architecture decision records
│   └── defense/            # report, slides, viva Q&A prep
├── ml/
│   ├── data/
│   │   ├── raw/            # FROZEN. Never edited in place. Never regenerated after Day 1.
│   │   └── processed/      # feature-engineered outputs
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_features.ipynb
│   │   └── 03_modeling.ipynb
│   ├── src/                # reusable .py modules (features, training, backtest)
│   ├── models/             # pickled model + scaler + feature list
│   ├── reports/            # exported figures & metrics tables
│   └── requirements.txt
├── service/                # FastAPI model server
├── backend/                # Laravel
└── frontend/               # React
```

---

## 7. Working Rules for Claude (every session, no exceptions)

1. **Read first.** Read `CLAUDE.md`, `PLAN.md`, `PROGRESS.md`, and `docs/adr/*` before producing anything.
2. **One day at a time.** Work only on the current day's Definition of Done. Do not start the next day's work "since we have time."
3. **Refuse scope creep — including from Ashutosh.** If asked for something in the OUT OF SCOPE list, say no, name the rule, and add it to the Parking Lot.
4. **ADRs are binding.** Never silently contradict an ADR. If a change is genuinely needed, stop and propose a *new* ADR that supersedes the old one. The old ADR is marked `Superseded`, never deleted.
5. **Restate before building.** Before writing any code, restate the day's DoD in one line so drift is visible immediately.
6. **No invented numbers.** Never fabricate accuracy scores, dataset sizes, row counts, or results. If a number isn't from a real run, say so explicitly.
7. **No invented data sources.** If a NEPSE endpoint or scraping method is uncertain, say it is uncertain. Do not present a guessed URL as working.
8. **Freeze means freeze.** Once the model is frozen (end of Day 5), no retraining, no feature changes. App work consumes the frozen artifact only.
9. **End every session with the End Prompt block.** No session ends without an updated PROGRESS entry and the next session's start prompt.

---

## 8. Anti-Drift Guarantees

The three failure modes this system exists to prevent:

| Failure | Guard |
|---|---|
| **Scope creep** — "let's also add sentiment analysis" | Section 4 OUT OF SCOPE list + Parking Lot |
| **Silent architecture drift** — a new chat re-decides something already settled | `docs/adr/` + Rule 4 |
| **Lost state** — new chat doesn't know what's done | `PROGRESS.md` + self-chaining session prompts |

---

## 9. Honesty Contract (important for the defense)

Short-horizon stock direction prediction is genuinely hard. Realistic directional accuracy on NEPSE daily data is roughly **52–58%**. That is a legitimate, defensible result.

- **Do not chase a high number.** If accuracy comes out near 90%, assume data leakage and hunt for it — most likely a look-ahead feature or a target computed before the split.
- **The panel will probe this.** A candidate who reports 54% and explains why that is meaningful scores higher than one reporting 95% who cannot explain it.
- Report the **majority-class baseline** alongside every model. If the model doesn't beat it, say so.

---

## 10. Whole-Project Definition of Done

- [ ] Frozen raw dataset committed, documented, reproducible
- [ ] EDA notebook with at least 6 meaningful visualizations and written interpretation
- [ ] Feature engineering pipeline as importable `.py` module (not notebook-only)
- [ ] Three models trained and compared under walk-forward validation
- [ ] Full metric suite: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importance
- [ ] Backtest vs. buy-and-hold with equity curve chart
- [ ] FastAPI `/predict` endpoint working locally
- [ ] Laravel backend calling the service and persisting predictions
- [ ] React dashboard rendering chart + prediction + confidence
- [ ] Written report covering all three required outline sections
- [ ] Slide deck matching the required outline
- [ ] Viva Q&A prep sheet completed
- [ ] Demo video + screenshots captured as live-demo fallback
- [ ] Full rehearsal done at least twice

---

## 11. Known Risks

| Risk | Mitigation |
|---|---|
| NEPSE data hard to scrape / incomplete | Day 1 has a hard cutoff. If unsolved by end of Day 1, switch to the fallback: manual Mero Share / Sasto Share CSV export for a smaller stock basket. Document the pivot as an ADR. |
| Model accuracy near random | This is an acceptable outcome. Report it honestly, analyze *why* (efficient market, noise, limited features) — that analysis is itself defense material. |
| App integration eats report time | Report and slides have their own dedicated day. If the app is incomplete on Day 8, ship it partial and present it as-is. **The report is never sacrificed for the app.** |
| Live demo fails on defense day | Screenshots + recorded demo video prepared in advance. |
