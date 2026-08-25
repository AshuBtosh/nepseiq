# ADR-0005: Application Architecture

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Planning

## Context
The ML model must be reachable by a web frontend. The model runs in Python; the author's web expertise is PHP/Laravel.

## Decision
Three-tier architecture:

```
React (UI)  →  Laravel 12 (API, auth, persistence)  →  FastAPI (model inference)
                        ↓
                     MySQL
```

- FastAPI is **stateless** — loads frozen artifacts at startup, holds no business logic
- Laravel owns auth, the stock basket, prediction persistence, and caching
- React never calls FastAPI directly
- Laravel handles FastAPI being unreachable with a clean error response

## Alternatives Considered
| Option | Why rejected |
|---|---|
| React → FastAPI directly | No auth layer, no persistence, no caching |
| Laravel invoking Python via `shell_exec` | Fragile, slow, no error handling, indefensible architecturally |
| Monolithic Streamlit app | Discards the differentiating full-stack work |
| ONNX export + PHP inference | Unnecessary complexity for the timeline |

## Consequences
**Positive:** Clean separation. Model retrainable without touching the web app. Each tier is independently testable. Realistic production shape.

**Negative:** Three processes to run during the demo; network hop adds latency. Mitigated by prediction caching and a recorded demo video.

## Defense Note
The key architectural point is the model boundary: inference is stateless and versioned behind HTTP, so the model becomes a swappable dependency rather than a coupled part of the application.
