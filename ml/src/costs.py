"""
NEPSE transaction cost model.

Sources (retrieved 2026-08-27):
  - Broker commission tiers: SEBON revised schedule effective Jestha 1, 2081 BS
    (10% reduction from the prior 0.40-0.27% structure).
    https://www.investopaper.com/news/stock-broker-commission-nepal/
  - SEBON regulatory fee 0.015% on both buy and sell legs.
    https://kharchapatra.com/blog/nepse-share-trading-costs-nepal
    Corroborated by published broker fee schedules (e.g. citizensbroker.com/equity).
  - DP charge Rs 25 per scrip per settlement, levied by CDSC via the broker.
    https://kharchapatra.com/blog/nepse-share-trading-costs-nepal

KNOWN SOURCE DISAGREEMENT (do not silently resolve):
  1. Several broker sites still publish the pre-2081 tiers (0.40% / 0.30% / 0.27%).
     This module uses the revised SEBON schedule. If the panel asks, the answer is
     that the revised schedule is the current regulated maximum and the older
     published tables are stale.
  2. Sources disagree on whether the DP charge applies to both legs or the sell
     only. This module applies it to BOTH legs by default (APPLY_DP_BOTH_LEGS).
     That is a deliberate, conservative MODELING ASSUMPTION, not a confirmed rule
     -- it makes the strategy look worse, never better. Flip the flag to see the
     sensitivity; report both if asked.

DELIBERATELY EXCLUDED: capital gains tax. PLAN.md Day 5 scopes costs to broker
commission + SEBON fee + DP charge. Current CGT rates are also reported
inconsistently across sources, and CGT depends on realized per-lot holding period,
which a fold-level directional backtest does not model faithfully. Stated as a
limitation in model_card.md rather than approximated.
"""

from __future__ import annotations

# Broker commission tiers: (upper_bound_inclusive_NPR, rate)
# Final tier uses None as an open upper bound.
BROKER_TIERS: list[tuple[float | None, float]] = [
    (50_000.0, 0.0036),      # up to Rs 50,000          -> 0.36%
    (500_000.0, 0.0033),     # Rs 50,000 - 5 lakh       -> 0.33%
    (2_000_000.0, 0.0031),   # Rs 5 lakh - 20 lakh      -> 0.31%
    (10_000_000.0, 0.0027),  # Rs 20 lakh - 1 crore     -> 0.27%
    (None, 0.0024),          # above Rs 1 crore         -> 0.24%
]

SEBON_FEE_RATE = 0.00015     # 0.015%, both legs
DP_CHARGE_NPR = 25.0         # flat, per scrip, per settlement

APPLY_DP_BOTH_LEGS = True    # modeling assumption -- see module docstring


def broker_commission_rate(turnover: float) -> float:
    """Return the SEBON-regulated maximum commission rate for a given turnover."""
    if turnover < 0:
        raise ValueError(f"turnover must be non-negative, got {turnover}")
    for upper, rate in BROKER_TIERS:
        if upper is None or turnover <= upper:
            return rate
    raise AssertionError("unreachable: final tier has open upper bound")


def leg_cost(turnover: float, leg: str) -> dict[str, float]:
    """
    Cost of ONE leg (a buy or a sell) of a single-scrip trade.

    Returns the component breakdown so the backtest can report where the money
    went, rather than only a total. Do not collapse this to a scalar upstream --
    the DoD requires stating the impact of transaction costs, which means showing
    the components.
    """
    if leg not in ("buy", "sell"):
        raise ValueError(f"leg must be 'buy' or 'sell', got {leg!r}")

    commission = turnover * broker_commission_rate(turnover)
    sebon = turnover * SEBON_FEE_RATE
    dp = DP_CHARGE_NPR if (leg == "sell" or APPLY_DP_BOTH_LEGS) else 0.0

    return {
        "commission": commission,
        "sebon_fee": sebon,
        "dp_charge": dp,
        "total": commission + sebon + dp,
    }


def round_trip_cost(buy_turnover: float, sell_turnover: float) -> dict[str, float]:
    """Total cost of a complete buy -> sell round trip on one scrip."""
    buy = leg_cost(buy_turnover, "buy")
    sell = leg_cost(sell_turnover, "sell")
    return {
        "buy_total": buy["total"],
        "sell_total": sell["total"],
        "commission": buy["commission"] + sell["commission"],
        "sebon_fee": buy["sebon_fee"] + sell["sebon_fee"],
        "dp_charge": buy["dp_charge"] + sell["dp_charge"],
        "total": buy["total"] + sell["total"],
    }


def round_trip_drag_pct(turnover: float) -> float:
    """
    Round-trip cost as a percentage of position size, assuming buy and sell
    turnover are equal. This is the break-even move a trade must clear before it
    makes anything -- the single most useful number for the defense, because it
    shows why a 1-day horizon strategy is expensive regardless of accuracy.
    """
    if turnover <= 0:
        raise ValueError(f"turnover must be positive, got {turnover}")
    return round_trip_cost(turnover, turnover)["total"] / turnover * 100.0


if __name__ == "__main__":
    print("Round-trip cost drag by position size (buy+sell, one scrip)")
    print(f"{'Position (NPR)':>16} {'Commission':>11} {'SEBON':>8} "
          f"{'DP':>7} {'Total':>10} {'Drag %':>8}")
    for size in (5_000, 25_000, 50_000, 100_000, 500_000,
                 1_000_000, 5_000_000, 20_000_000):
        c = round_trip_cost(size, size)
        print(f"{size:>16,.0f} {c['commission']:>11,.2f} {c['sebon_fee']:>8,.2f} "
              f"{c['dp_charge']:>7,.2f} {c['total']:>10,.2f} "
              f"{round_trip_drag_pct(size):>7.3f}%")
    print(f"\nAPPLY_DP_BOTH_LEGS = {APPLY_DP_BOTH_LEGS} "
          f"(modeling assumption -- see docstring)")