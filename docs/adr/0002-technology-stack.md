# ADR-0002: Technology Stack

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Planning

## Context
The institution's brief explicitly names Python, Pandas, Scikit-learn, and SQL. The author works professionally in Laravel and needs an app layer that is fast to build under time pressure.

## Decision
- **ML:** Python 3.11, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter
- **Model serving:** FastAPI + Uvicorn
- **Backend:** Laravel 12
- **Database:** MySQL
- **Frontend:** React + Recharts
- **VCS:** Git

## Alternatives Considered
| Option | Why rejected |
|---|---|
| Serve the model directly from Laravel (via shell exec) | Fragile, slow, poor separation of concerns, bad to defend |
| Streamlit only (no Laravel/React) | Simpler, but discards the author's differentiating strength |
| Django instead of Laravel | Author has no production Django experience; learning cost unjustifiable |
| Node/Express instead of Laravel | Same reason |
| PostgreSQL | No advantage here; MySQL is already in the author's toolchain |

## Consequences
**Positive:** Uses every tool the brief names. FastAPI/Laravel separation is clean and easy to explain. Author's existing Laravel skill removes most backend risk.

**Negative:** Three services to run during the demo. Mitigated by a recorded demo video as fallback (PLAN.md Day 7).

## Defense Note
The Python/Laravel split is a real architectural argument: ML runtime and business logic have different scaling profiles and different dependency trees. Keeping the model behind its own HTTP boundary means the model can be retrained and redeployed without touching the web application.
