import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PATH_NQ = BASE_DIR / "data" / "USTECH100M_M1.csv"
PATH_SP = BASE_DIR / "data" / "US500_M1.csv"


def simulate_trade(future_df: pd.DataFrame, sl: float, tp: float, side: str) -> float:
    comm = 0.05
    for _, candle in future_df.iterrows():
        if side == "LONG":
            if candle["LOW"] <= sl:
                return -1.0 - comm
            if candle["HIGH"] >= tp:
                return 3.0 - comm
        else:
            if candle["HIGH"] >= sl:
                return -1.0 - comm
            if candle["LOW"] <= tp:
                return 3.0 - comm
    return 0.0


def run_cross_index_smt_backtest(path_nq: Path, path_sp: Path) -> pd.DataFrame:
    print(f"Synchronizing M1 series: {path_nq.name} vs. {path_sp.name}...")
    dtype_dict = {"DATE": str, "TIME": str}

    nq = pd.read_csv(path_nq, sep="\t", dtype=dtype_dict)
    sp = pd.read_csv(path_sp, sep="\t", dtype=dtype_dict)

    for df_temp in (nq, sp):
        df_temp.columns = [c.replace("<", "").replace(">", "").strip().upper() for c in df_temp.columns]
        for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")
        df_temp.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"], inplace=True)
        df_temp["DATETIME"] = pd.to_datetime(df_temp["DATE"] + " " + df_temp["TIME"])
        df_temp.set_index("DATETIME", inplace=True)

    df = nq[["OPEN", "HIGH", "LOW", "CLOSE"]].join(
        sp[["HIGH", "LOW", "CLOSE"]].rename(columns={"HIGH": "HIGH_SP", "LOW": "LOW_SP", "CLOSE": "CLOSE_SP"}),
        how="inner",
    )

    daily_close = df["CLOSE"].resample("D").last().shift(1)
    df["PREV_DAY_CLOSE"] = daily_close.reindex(df.index, method="ffill")

    trades = []
    for date, day_df in df.groupby(df.index.date):
        if len(day_df) < 100:
            continue
        opening_range = day_df.between_time("14:15", "14:30")
        if opening_range.empty:
            continue

        range_high = opening_range["HIGH"].max()
        range_low = opening_range["LOW"].min()
        range_high_sp = opening_range["HIGH_SP"].max()
        range_low_sp = opening_range["LOW_SP"].min()

        price_930_rows = day_df.between_time("14:31", "14:31")
        if price_930_rows.empty:
            continue
        price_930 = price_930_rows["CLOSE"].iloc[0]
        prev_close = day_df["PREV_DAY_CLOSE"].iloc[0]
        if pd.isna(prev_close):
            continue

        bias = "BULL" if price_930 > prev_close else "BEAR"
        window = day_df.between_time("14:31", "16:00")
        active_day_trade = False

        for i in range(5, len(window)):
            if active_day_trade:
                break
            row = window.iloc[i]
            prev_1 = window.iloc[i - 1]
            prev_3 = window.iloc[i - 3]

            if bias == "BULL":
                liquidity_raid = window["LOW"].iloc[max(0, i - 15) : i].min() < range_low
                mss = row["CLOSE"] > window["HIGH"].iloc[i - 5 : i].max()
                fvg = prev_1["LOW"] > prev_3["HIGH"]
                smt_confirm = window["LOW_SP"].iloc[max(0, i - 15) : i].min() < range_low_sp

                if liquidity_raid and mss and fvg and smt_confirm:
                    entry = row["CLOSE"]
                    sl = range_low - 5.0
                    tp = entry + (entry - sl) * 3.0
                    res = simulate_trade(window.iloc[i + 1 :], sl, tp, "LONG")
                    trades.append({"date": date, "type": "LONG", "result_R": res})
                    active_day_trade = True

            elif bias == "BEAR":
                liquidity_raid = window["HIGH"].iloc[max(0, i - 15) : i].max() > range_high
                mss = row["CLOSE"] < window["LOW"].iloc[i - 5 : i].min()
                fvg = prev_1["HIGH"] < prev_3["LOW"]
                smt_confirm = window["HIGH_SP"].iloc[max(0, i - 15) : i].max() > range_high_sp

                if liquidity_raid and mss and fvg and smt_confirm:
                    entry = row["CLOSE"]
                    sl = range_high + 5.0
                    tp = entry - (sl - entry) * 3.0
                    res = simulate_trade(window.iloc[i + 1 :], sl, tp, "SHORT")
                    trades.append({"date": date, "type": "SHORT", "result_R": res})
                    active_day_trade = True

    return pd.DataFrame(trades)


if __name__ == "__main__":
    res_df = run_cross_index_smt_backtest(PATH_NQ, PATH_SP)
    print(f"Processed sample M1 dataset ({len(res_df)} cross-index setups identified in slice).")
