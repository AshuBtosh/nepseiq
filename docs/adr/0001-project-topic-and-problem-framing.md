# ADR-0001: Project Topic and Problem Framing

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Planning

## Context
The diploma requires a capstone demonstrating the full DSML lifecycle: data collection, preprocessing, EDA, modeling, evaluation. The author has an existing NEPSE investment portfolio and prior Laravel/React engineering experience, giving both domain familiarity and the ability to ship an application layer that differentiates the project from notebook-only submissions.

An earlier concept — a plain binary "NEPSE stock movement predictor" — was judged too thin for the available time.

## Decision
Build **NepseIQ**: a NEPSE market intelligence platform predicting short-horizon directional price movement (1-day and 5-day) from engineered technical indicators, served through a FastAPI → Laravel → React stack.

Two prediction horizons only. Directional (binary classification), not price regression.

## Alternatives Considered
| Option | Why rejected |
|---|---|
| Single-horizon binary classifier | Too thin for the timeline available |
| Price regression (predict exact close) | Harder, noisier, less defensible; directional accuracy is the honest framing |
| Non-finance domain (health/HR) | Loses the author's domain knowledge advantage |
| Recommendation system | Weaker fit with available NEPSE data |

## Consequences
**Positive:** Domain knowledge accelerates EDA interpretation. Full-stack layer is a genuine differentiator and doubles as a portfolio piece. Two horizons allow comparative analysis without doubling the workload.

**Negative:** Financial prediction has a low accuracy ceiling. The project must be framed around methodology quality rather than headline accuracy — see CLAUDE.md §9.

## Defense Note
If asked "why stocks, isn't that unpredictable?" — that is the correct question, and the answer is the point: the project measures *how much* signal exists in technical indicators on NEPSE, honestly benchmarked against a majority-class baseline and buy-and-hold. Rigorous methodology on a hard problem is the contribution.
