import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"


def analyze_nasdaq_overnight_gaps(ticker: str = "QQQ", period: str = "5y"):
    import yfinance as yf

    print(f"Downloading {period} daily series for {ticker}...")
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df["Prev_Close"] = df["Close"].shift(1)
    df["Gap_Pct"] = ((df["Open"] - df["Prev_Close"]) / df["Prev_Close"]) * 100.0
    df["Gap_Abs_Pct"] = df["Gap_Pct"].abs()
    df = df.dropna()

    total_days = len(df)
    median_gap = df["Gap_Abs_Pct"].median()
    mean_gap = df["Gap_Abs_Pct"].mean()
    max_gap = df["Gap_Pct"].max()
    min_gap = df["Gap_Pct"].min()

    small_noise = len(df[df["Gap_Abs_Pct"] < 0.20]) / total_days * 100.0
    tradable_normal = len(df[(df["Gap_Abs_Pct"] >= 0.20) & (df["Gap_Abs_Pct"] <= 0.80)]) / total_days * 100.0
    extreme_gap = len(df[df["Gap_Abs_Pct"] > 0.80]) / total_days * 100.0

    bull_gaps = len(df[df["Gap_Pct"] > 0]) / total_days * 100.0
    bear_gaps = len(df[df["Gap_Pct"] < 0]) / total_days * 100.0

    print("\n" + "=" * 55)
    print(f" STATISTICAL REPORT: NASDAQ ({ticker}) OVERNIGHT GAPS ({period})")
    print("=" * 55)
    print(f"Total Sessions Analyzed : {total_days} days")
    print(f"Mean Absolute Gap       : {mean_gap:.2f}%")
    print(f"Median Absolute Gap     : {median_gap:.2f}%")
    print(f"Largest Bullish Gap     : +{max_gap:.2f}%")
    print(f"Largest Bearish Gap     : {min_gap:.2f}%")
    print("-" * 55)
    print(f"Noise Regime (< 0.20%)          : {small_noise:.1f}%")
    print(f"Tradable Regime (0.20% - 0.80%) : {tradable_normal:.1f}%")
    print(f"Extreme Regime (> 0.80%)        : {extreme_gap:.1f}%")
    print("-" * 55)
    print(f"Bullish Overnight Bias          : {bull_gaps:.1f}%")
    print(f"Bearish Overnight Bias          : {bear_gaps:.1f}%")
    print("=" * 55)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.histplot(df["Gap_Pct"], bins=90, kde=True, ax=axes[0], color="#1e3a8a")
    axes[0].set_title("Directional Overnight Gap Distribution (%)")
    axes[0].set_xlabel("Overnight Gap (%)")
    axes[0].set_ylabel("Session Count")
    axes[0].axvline(0, color="crimson", linestyle="--")

    sns.ecdfplot(data=df, x="Gap_Abs_Pct", ax=axes[1], color="#0f766e")
    axes[1].set_title("Empirical Cumulative Distribution Function (ECDF) of |Gap|")
    axes[1].set_xlabel("Absolute Overnight Gap (%)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].axvline(0.20, color="gray", linestyle="--", label="0.20% Threshold")
    axes[1].axvline(0.80, color="darkorange", linestyle="--", label="0.80% Threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nasdaq_overnight_gap_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    analyze_nasdaq_overnight_gaps()
