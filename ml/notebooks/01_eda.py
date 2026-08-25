# %% [markdown]
# # 01 - EDA
# Day 2. Run cell by cell in VSCode/Jupyter (each `# %%` is a cell), or top to
# bottom as a script. Figures save to ml/reports/. After running, paste the
# printed console output back, and attach the saved PNGs so interpretations
# can be written against what they actually show.

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src") if (Path.cwd() / "src").exists() is False else "src")
# If the above path juggling misbehaves in your environment, just hardcode:
# sys.path.append(r"C:\Users\DELL\Downloads\nepseiq-control-system\nepseiq\ml\src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_raw

sns.set_theme(style="whitegrid")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

df = load_raw()
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"Loaded: {df.shape}")

# TODO: confirm this mapping against ml/data/raw/README.md before trusting the
# sector-comparison chart below.
SECTOR_MAP = {
    "NABIL": "Banking", "EBL": "Banking", "GBIME": "Banking",
    "CHCL": "Hydropower", "AHPC": "Hydropower", "RIDI": "Hydropower",
    "NLIC": "Insurance", "SICL": "Insurance", "NLG": "Insurance",
    "CBBL": "Microfinance", "SKBBL": "Microfinance",
}
df["sector"] = df["symbol"].map(SECTOR_MAP)
if df["sector"].isna().any():
    print("!! Unmapped symbols found - fix SECTOR_MAP:", df.loc[df["sector"].isna(), "symbol"].unique())

# %% [markdown]
# ## 1. Coverage / session density — CBBL and RIDI called out
# ADR-0006 flagged CBBL (low session density despite long listing) and RIDI
# (only ~4yr history) as unconfirmed-liquidity picks. Make that visible directly.

# %%
coverage = df.groupby("symbol").agg(
    rows=("date", "count"),
    first_date=("date", "min"),
    last_date=("date", "max"),
).reset_index()
coverage["years_listed"] = (coverage["last_date"] - coverage["first_date"]).dt.days / 365.25
coverage["rows_per_year"] = coverage["rows"] / coverage["years_listed"]
coverage = coverage.sort_values("rows_per_year")
print(coverage.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#d62728" if s in ("CBBL", "RIDI") else "#4c72b0" for s in coverage["symbol"]]
ax.barh(coverage["symbol"], coverage["rows_per_year"], color=colors)
ax.set_xlabel("Trading rows per year listed")
ax.set_title("Session density by symbol (red = flagged in ADR-0006: CBBL, RIDI)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "01_coverage_density.png", dpi=150)
plt.show()
print("Saved: reports/01_coverage_density.png")

# %% [markdown]
# ## 2. Price trends per stock (normalized to 100 at first observation)

# %%
fig, ax = plt.subplots(figsize=(12, 7))
for symbol, g in df.groupby("symbol"):
    normed = g["close"] / g["close"].iloc[0] * 100
    ax.plot(g["date"], normed, label=symbol, linewidth=1)
ax.set_yscale("log")
ax.set_ylabel("Close price (normalized to 100 at listing, log scale)")
ax.set_title("Normalized price trends, all symbols")
ax.legend(loc="upper left", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "02_price_trends.png", dpi=150)
plt.show()
print("Saved: reports/02_price_trends.png")

# %% [markdown]
# ## 3. Volume trends per stock (rolling 60-day mean traded_quantity)

# %%
fig, ax = plt.subplots(figsize=(12, 7))
for symbol, g in df.groupby("symbol"):
    roll = g.set_index("date")["traded_quantity"].rolling(60, min_periods=10).mean()
    ax.plot(roll.index, roll.values, label=symbol, linewidth=1)
ax.set_yscale("log")
ax.set_ylabel("60-day rolling mean traded quantity (log scale)")
ax.set_title("Volume trends, all symbols")
ax.legend(loc="upper left", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "03_volume_trends.png", dpi=150)
plt.show()
print("Saved: reports/03_volume_trends.png")

# %% [markdown]
# ## 4. Sector comparison (mean daily return, mean volatility by sector)

# %%
df["log_return"] = df.groupby("symbol")["close"].transform(lambda s: np.log(s / s.shift(1)))

sector_stats = df.groupby("sector")["log_return"].agg(["mean", "std", "count"]).reset_index()
sector_stats["mean_annualized_pct"] = sector_stats["mean"] * 252 * 100
sector_stats["vol_annualized_pct"] = sector_stats["std"] * np.sqrt(252) * 100
print(sector_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=sector_stats, x="sector", y="mean_annualized_pct", ax=axes[0])
axes[0].set_title("Mean annualized return by sector (%)")
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(data=sector_stats, x="sector", y="vol_annualized_pct", ax=axes[1])
axes[1].set_title("Annualized volatility by sector (%)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "04_sector_comparison.png", dpi=150)
plt.show()
print("Saved: reports/04_sector_comparison.png")

# %% [markdown]
# ## 5. Return distribution — fat tails / skew check

# %%
returns = df["log_return"].dropna()
skew = returns.skew()
kurt = returns.kurtosis()  # excess kurtosis (0 = normal)
print(f"Pooled daily log returns: n={len(returns)}, mean={returns.mean():.5f}, "
      f"std={returns.std():.5f}, skew={skew:.3f}, excess_kurtosis={kurt:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(returns, bins=200, stat="density", ax=ax, color="#4c72b0")
x = np.linspace(returns.min(), returns.max(), 300)
from scipy import stats as sp_stats
ax.plot(x, sp_stats.norm.pdf(x, returns.mean(), returns.std()), color="red",
        label="Normal fit", linewidth=1.5)
ax.set_xlim(returns.quantile(0.001), returns.quantile(0.999))
ax.set_title(f"Pooled daily log return distribution (skew={skew:.2f}, excess kurtosis={kurt:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "05_return_distribution.png", dpi=150)
plt.show()
print("Saved: reports/05_return_distribution.png")

# %% [markdown]
# ## 6. Volatility over time + clustering (rolling 20-day std, example + pooled)

# %%
df["roll_vol_20d"] = df.groupby("symbol")["log_return"].transform(
    lambda s: s.rolling(20, min_periods=10).std()
)

fig, ax = plt.subplots(figsize=(12, 6))
for symbol, g in df.groupby("symbol"):
    ax.plot(g["date"], g["roll_vol_20d"], label=symbol, linewidth=0.8, alpha=0.7)
ax.set_ylabel("20-day rolling std of daily log return")
ax.set_title("Volatility over time, all symbols (visual clustering check)")
ax.legend(loc="upper left", ncol=2, fontsize=7)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "06_volatility_over_time.png", dpi=150)
plt.show()
print("Saved: reports/06_volatility_over_time.png")

# %% [markdown]
# ## 7. Correlation heatmap across stocks (daily returns)

# %%
returns_wide = df.pivot(index="date", columns="symbol", values="log_return")
corr = returns_wide.corr()
print(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Daily return correlation across symbols")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "07_correlation_heatmap.png", dpi=150)
plt.show()
print("Saved: reports/07_correlation_heatmap.png")

# %% [markdown]
# ## 8. Class balance for target_1d and target_5d + majority-class baseline
# NOTE: this is a preliminary class-balance check for Day 2 only. Formal
# target construction happens in features.py on Day 3 per PLAN.md.

# %%
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

def _target(s, horizon):
    future = s.shift(-horizon)
    result = (future > s).astype("Int64")
    result[future.isna()] = pd.NA
    return result

df["target_1d"] = df.groupby("symbol")["close"].transform(lambda s: _target(s, 1))
df["target_5d"] = df.groupby("symbol")["close"].transform(lambda s: _target(s, 5))

for target in ["target_1d", "target_5d"]:
    valid = df[target].dropna()
    counts = valid.value_counts(normalize=True).sort_index()
    majority_baseline = counts.max()
    print(f"\n{target}: n={len(valid)}")
    print(f"  down (0): {counts.get(0, 0):.4f}")
    print(f"  up   (1): {counts.get(1, 0):.4f}")
    print(f"  majority-class baseline accuracy: {majority_baseline:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, target in zip(axes, ["target_1d", "target_5d"]):
    valid = df[target].dropna()
    counts = valid.value_counts(normalize=True).sort_index()
    ax.bar(["Down", "Up"], [counts.get(0, 0), counts.get(1, 0)], color=["#d62728", "#2ca02c"])
    ax.set_title(target)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.suptitle("Class balance, both targets (pooled across all symbols)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "08_class_balance.png", dpi=150)
plt.show()
print("Saved: reports/08_class_balance.png")

# %% [markdown]
# ## Per-symbol class balance (CBBL / RIDI check — do the flagged symbols
# behave differently from the rest?)

# %%
per_symbol_balance = df.groupby("symbol")[["target_1d", "target_5d"]].mean(numeric_only=True)
print(per_symbol_balance.round(4).to_string())

# %% [markdown]
# ## 9. Flat-price day check + extreme-return outlier check
# Two things worth quantifying before writing findings: (a) whether "no
# price movement" days are inflating the "down" class since target uses
# strict >, and (b) whether extreme daily returns are real volatility or
# unadjusted stock splits/bonus share issues (common in NEPSE, per
# PLAN.md Day 2's explicit "stock splits/bonus adjustments" task).

# %%
flat_days = (df["close"] == df.groupby("symbol")["close"].shift(-1)).sum()
total_with_next = df.groupby("symbol")["close"].shift(-1).notna().sum()
print(f"Rows where next-day close == today's close (flat, counts as 'down'): "
      f"{flat_days} / {total_with_next} ({flat_days/total_with_next:.2%})")

flat_by_symbol = df.groupby("symbol")["close"].apply(
    lambda s: (s == s.shift(-1)).sum() / len(s)
).sort_values(ascending=False)
print("\nFlat-day rate by symbol:")
print(flat_by_symbol.round(4).to_string())

# %%
extreme = df.reindex(df["log_return"].abs().sort_values(ascending=False).index).head(25)
print("\nTop 25 largest absolute daily log returns:")
print(extreme[["symbol", "date", "open", "high", "low", "close", "per_change", "log_return"]].to_string(index=False))
