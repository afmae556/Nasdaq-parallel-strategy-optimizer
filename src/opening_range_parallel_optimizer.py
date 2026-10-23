import pandas as pd
import numpy as np
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "USTECH100M_M1.csv"
RESULTS_PATH = BASE_DIR / "results" / "optimizacion_us100_resultados.csv"

EMA_RANGE = range(20, 155, 5)
ATR_SL_RANGE = np.arange(1.0, 3.0, 0.2)
ATR_TP_RANGE = np.arange(1.5, 4.5, 0.5)
OR_DURATION_MINS = [15, 30]
SPREAD_COMMISSION = 2.0


def load_and_prep_data(filepath: Path):
    print(f"Loading and preprocessing M1 index data from '{filepath.name}'...")
    try:
        df = pd.read_csv(filepath, sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, sep=",")
    except Exception:
        df = pd.read_csv(filepath, sep=",")

    df.columns = [c.replace("<", "").replace(">", "").strip() for c in df.columns]
    df["DATETIME"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df.set_index("DATETIME", inplace=True)
    df.sort_index(inplace=True)

    cols = ["OPEN", "HIGH", "LOW", "CLOSE"]
    df[cols] = df[cols].astype(float)

    df_5m = df.resample("5min").agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last"}).dropna()
    df_1h = df.resample("1h").agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last"}).dropna()

    prev_close = df_1h["CLOSE"].shift(1)
    tr1 = df_1h["HIGH"] - df_1h["LOW"]
    tr2 = (df_1h["HIGH"] - prev_close).abs()
    tr3 = (df_1h["LOW"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_1h["ATR_14"] = tr.rolling(14).mean()

    return df, df_5m, df_1h


def get_ny_open_utc(timestamp):
    ny_tz = ZoneInfo("America/New_York")
    dt_ny = timestamp.replace(tzinfo=datetime.timezone.utc).astimezone(ny_tz)
    if dt_ny.dst() != datetime.timedelta(0):
        return 13, 30
    return 14, 30


def run_strategy(params, df_5m, df_1h):
    ema_p, sl_mult, tp_mult, or_dur = params
    trend_series = df_1h["CLOSE"].ewm(span=ema_p, adjust=False).mean()
    aligned_ema = trend_series.shift(1).reindex(df_5m.index, method="ffill")
    aligned_atr = df_1h["ATR_14"].shift(1).reindex(df_5m.index, method="ffill")

    equity = 100000.0
    trades_count = 0
    wins = 0
    max_dd = 0.0
    peak_equity = 100000.0

    days = df_5m.groupby(df_5m.index.date)
    for _, day_data in days:
        if len(day_data) < 10:
            continue
        first_idx = day_data.index[0]
        open_h, open_m = get_ny_open_utc(first_idx)
        start_time = first_idx.replace(hour=open_h, minute=open_m, second=0)
        or_end_time = start_time + datetime.timedelta(minutes=int(or_dur))
        cutoff_time = or_end_time + datetime.timedelta(hours=2, minutes=30)

        or_slice = day_data.loc[start_time:or_end_time]
        if or_slice.empty:
            continue
        or_high = or_slice["HIGH"].max()
        or_low = or_slice["LOW"].min()

        trading_slice = day_data.loc[or_end_time:cutoff_time]
        daily_trades = 0
        for ts, row in trading_slice.iterrows():
            if daily_trades >= 2:
                break
            ema_val = aligned_ema.loc[ts]
            atr_val = aligned_atr.loc[ts]
            if pd.isna(ema_val) or pd.isna(atr_val) or atr_val <= 0:
                continue

            if row["CLOSE"] > or_high and row["CLOSE"] > ema_val:
                sl_dist = sl_mult * atr_val
                tp_dist = tp_mult * atr_val
                future = day_data.loc[ts:]
                for _, f_row in future.iloc[1:].iterrows():
                    if f_row["LOW"] <= row["CLOSE"] - sl_dist:
                        equity -= (sl_dist + SPREAD_COMMISSION)
                        trades_count += 1
                        daily_trades += 1
                        break
                    if f_row["HIGH"] >= row["CLOSE"] + tp_dist:
                        equity += (tp_dist - SPREAD_COMMISSION)
                        trades_count += 1
                        wins += 1
                        daily_trades += 1
                        break
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * 100.0
                max_dd = max(max_dd, dd)

    net_profit = equity - 100000.0
    win_rate = (wins / trades_count * 100.0) if trades_count > 0 else 0.0
    rec_factor = (net_profit / (max_dd * 1000.0)) if max_dd > 0 else 0.0
    return {
        "EMA": ema_p,
        "SL_X": round(sl_mult, 2),
        "TP_X": round(tp_mult, 2),
        "OR_Min": or_dur,
        "Net Profit": round(net_profit, 2),
        "Max DD %": round(max_dd, 2),
        "# Trades": trades_count,
        "Win Rate": round(win_rate, 2),
        "Recovery Factor": round(rec_factor, 2),
    }


if __name__ == "__main__":
    if RESULTS_PATH.exists():
        df_res = pd.read_csv(RESULTS_PATH)
        top10 = df_res.sort_values("Recovery Factor", ascending=False).head(10)
        print("=== TOP 10 PARAMETER CONFIGURATIONS BY RECOVERY FACTOR ===")
        print(top10.to_string(index=False))

