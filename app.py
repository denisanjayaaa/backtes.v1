"""
IDX Quant Strategy Screener — Flask Backend
All 7 strategies from the Instagram post, backtestable on any IDX stock.
Run: python app.py  →  open http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Fees ──────────────────────────────────────────────────────────────────────
FEE_BUY  = 0.0015
FEE_SELL = 0.0025
FEE_RT   = FEE_BUY + FEE_SELL

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch(ticker, start="2020-01-01", end="2026-03-01"):
    t = ticker if ticker.endswith(".JK") else ticker + ".JK"
    df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open","High","Low","Close","Volume"]].dropna()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def equity_curve(pnl_series):
    r = pnl_series / 100
    return (1 + r).cumprod().tolist()

def base_stats(trades_df):
    if trades_df is None or trades_df.empty:
        return {"total_trades": 0, "win_rate": 0, "avg_ret": 0,
                "payoff": 0, "total_pnl": 0, "max_dd": 0, "ann_ret": 0}
    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr     = len(wins) / len(trades_df) * 100
    aw     = wins["pnl"].mean()   if not wins.empty   else 0
    al     = losses["pnl"].mean() if not losses.empty else 0
    payoff = abs(aw / al) if al != 0 else 0
    r      = trades_df["pnl"] / 100
    total_days = (trades_df["exit"].iloc[-1] - trades_df["entry"].iloc[0]).days if len(trades_df) > 1 else 365
    years  = max(total_days / 365.25, 0.1)
    growth = (1 + r).prod()
    ann    = (growth ** (1 / years) - 1) * 100
    cum    = (1 + r).cumprod()
    dd     = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return {
        "total_trades": len(trades_df),
        "win_rate":     round(wr, 1),
        "avg_ret":      round(trades_df["pnl"].mean(), 3),
        "avg_win":      round(aw, 3),
        "avg_loss":     round(al, 3),
        "payoff":       round(payoff, 2),
        "total_pnl":    round(trades_df["pnl"].sum(), 2),
        "ann_ret":      round(ann, 1),
        "max_dd":       round(dd, 1),
        "equity_curve": [float(x) for x in equity_curve(trades_df.sort_values("entry")["pnl"])],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — Volatility Compression Breakout
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/vcb")
def api_vcb():
    ticker     = request.args.get("ticker", "ANTM")
    lookback   = int(request.args.get("lookback", 20))
    percentile = int(request.args.get("percentile", 25))
    hold_days  = int(request.args.get("hold_days", 10))
    stop_pct   = float(request.args.get("stop_pct", 3)) / 100

    try:
        df = fetch(ticker)
        df["RollingHigh"]   = df["High"].rolling(lookback).max()
        df["RollingLow"]    = df["Low"].rolling(lookback).min()
        df["ATR_pct"]       = (df["RollingHigh"] - df["RollingLow"]) / df["Close"]
        df["ATR_threshold"] = df["ATR_pct"].rolling(252).quantile(percentile / 100)
        df["Compressed"]    = df["ATR_pct"] < df["ATR_threshold"]
        df["Signal"]        = (
            df["Compressed"].shift(1) &
            (df["Close"] > df["RollingHigh"].shift(1) * 1.002)
        )
        df = df.dropna()

        trades, in_trade = [], False
        entry_price, entry_date, days_held = 0, None, 0
        for i in range(len(df)):
            row = df.iloc[i]
            if in_trade:
                days_held += 1
                if row["Close"] < entry_price * (1 - stop_pct):
                    pnl = (row["Close"] - entry_price) / entry_price * 100 - FEE_RT * 100
                    trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3), "reason": "STOP"})
                    in_trade = False
                elif days_held >= hold_days:
                    pnl = (row["Close"] - entry_price) / entry_price * 100 - FEE_RT * 100
                    trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3), "reason": "TIME"})
                    in_trade = False
            if not in_trade and row["Signal"]:
                entry_price, entry_date, days_held, in_trade = row["Open"] * 1.001, df.index[i], 0, True

        tdf   = pd.DataFrame(trades)
        stats = base_stats(tdf)
        compression_zones = df[df["Compressed"]].index.strftime("%Y-%m-%d").tolist()
        recent_signals    = df[df["Signal"]].tail(5).index.strftime("%Y-%m-%d").tolist()
        return jsonify({"ok": True, "stats": stats,
                        "recent_signals": recent_signals,
                        "compression_zones": len(compression_zones),
                        "trades": trades[-20:] if trades else []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Z-Score Reversion
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/zscore")
def api_zscore():
    tickerA  = request.args.get("tickerA", "ANTM")
    tickerB  = request.args.get("tickerB", "INCO")
    roll_win = int(request.args.get("roll_win", 60))
    z_entry  = float(request.args.get("z_entry", 2.0))
    z_exit   = float(request.args.get("z_exit", 0.5))
    corr_min = float(request.args.get("corr_min", 0.7))
    max_hold = int(request.args.get("max_hold", 20))

    try:
        dfA = fetch(tickerA)["Close"]
        dfB = fetch(tickerB)["Close"]
        common = dfA.index.intersection(dfB.index)
        pA, pB = dfA[common], dfB[common]
        ratio     = pA / pB
        roll_mean = ratio.rolling(roll_win).mean()
        roll_std  = ratio.rolling(roll_win).std()
        zscore    = (ratio - roll_mean) / roll_std
        corr      = pA.rolling(roll_win).corr(pB)

        df = pd.DataFrame({"pA": pA, "pB": pB, "zscore": zscore, "corr": corr}).dropna()

        trades, in_trade = [], False
        position, entry_date, entry_pA, entry_z, days_held = 0, None, 0, 0, 0
        for i in range(len(df)):
            row = df.iloc[i]
            if in_trade:
                days_held += 1
                exit_reason = None
                if position == 1 and row["zscore"] >= -z_exit: exit_reason = "REVERT"
                elif position == -1 and row["zscore"] <= z_exit: exit_reason = "REVERT"
                if days_held >= max_hold: exit_reason = "TIME"
                if exit_reason:
                    pnl = position * (row["pA"] - entry_pA) / entry_pA * 100 - FEE_RT * 100
                    trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3),
                                   "reason": exit_reason, "entry_z": round(entry_z,2)})
                    in_trade = False
            if not in_trade and row["corr"] >= corr_min:
                if row["zscore"] <= -z_entry:
                    in_trade, position, entry_date, entry_pA, entry_z, days_held = True, 1, df.index[i], row["pA"], row["zscore"], 0
                elif row["zscore"] >= z_entry:
                    in_trade, position, entry_date, entry_pA, entry_z, days_held = True, -1, df.index[i], row["pA"], row["zscore"], 0

        tdf   = pd.DataFrame(trades)
        stats = base_stats(tdf)
        current_z    = round(df["zscore"].iloc[-1], 3)
        current_corr = round(df["corr"].iloc[-1], 3)
        signal = "LONG " + tickerA if current_z <= -z_entry and current_corr >= corr_min else \
                 "SHORT "+ tickerA if current_z >= z_entry  and current_corr >= corr_min else "NO SIGNAL"
        zscore_history = df["zscore"].tail(60).round(3).tolist()
        return jsonify({"ok": True, "stats": stats, "current_z": current_z,
                        "current_corr": current_corr, "signal": signal,
                        "zscore_history": zscore_history,
                        "trades": trades[-20:] if trades else []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Overnight Drift
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/overnight")
def api_overnight():
    ticker    = request.args.get("ticker", "ANTM")
    ma_period = int(request.args.get("ma_period", 20))

    try:
        df = fetch(ticker)
        df["overnight_ret"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) - FEE_RT
        df["intraday_ret"]  = (df["Close"] - df["Open"]) / df["Open"] - FEE_RT
        df["MA"]            = df["Close"].rolling(ma_period).mean()
        df["above_MA"]      = df["Close"] > df["MA"]
        df["prev_on_pos"]   = df["overnight_ret"].shift(1) > 0
        df["filtered_mask"] = df["above_MA"] & df["prev_on_pos"]
        df = df.dropna()

        def stats_for(series):
            r = series[series != 0].dropna()
            if r.empty: return {}
            wins = (r > 0).sum()
            total_days = (r.index[-1] - r.index[0]).days
            years = max(total_days / 365.25, 0.1)
            growth = (1 + r).prod()
            ann = (growth ** (1 / years) - 1) * 100
            cum = (1 + r).cumprod()
            dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
            return {
                "sessions": len(r),
                "win_rate": round(wins / len(r) * 100, 1),
                "avg_ret":  round(r.mean() * 100, 4),
                "ann_ret":  round(ann, 1),
                "max_dd":   round(dd, 1),
                "equity":   (1 + r).cumprod().tolist(),
            }

        raw_on   = df["overnight_ret"]
        filtered = df["overnight_ret"].where(df["filtered_mask"], 0)
        intraday = df["intraday_ret"]

        dow_avg = {}
        days = ["Mon","Tue","Wed","Thu","Fri"]
        for d in range(5):
            mask = pd.to_datetime(df.index).dayofweek == d
            dow_avg[days[d]] = round(df.loc[mask, "overnight_ret"].mean() * 100, 4)

        currently_above = bool(df["above_MA"].iloc[-1])
        last_on_positive = bool(df["overnight_ret"].iloc[-1] > 0)
        signal = "✅ HOLD TONIGHT" if currently_above and last_on_positive else "⛔ SKIP TONIGHT"

        return jsonify({"ok": True,
                        "raw_overnight": stats_for(raw_on),
                        "filtered_overnight": stats_for(filtered),
                        "intraday": stats_for(intraday),
                        "dow_avg": dow_avg,
                        "signal": signal,
                        "currently_above_ma": currently_above})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Post-Earnings Drift
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/ped")
def api_ped():
    ticker        = request.args.get("ticker", "ANTM")
    gap_threshold = float(request.args.get("gap_threshold", 5)) / 100
    hold_days     = int(request.args.get("hold_days", 20))
    stop_pct      = float(request.args.get("stop_pct", 7)) / 100
    vol_ratio_min = float(request.args.get("vol_ratio_min", 1.5))

    try:
        df = fetch(ticker, start="2018-01-01")
        df["gap_pct"]   = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["vol_ma20"]  = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_ma20"]
        df["pos_event"] = (df["gap_pct"] >= gap_threshold) & (df["vol_ratio"] >= vol_ratio_min)
        df = df.dropna()

        trades, in_trade = [], False
        entry_price, entry_date, days_held = 0, None, 0
        for i in range(len(df)):
            row = df.iloc[i]
            if in_trade:
                days_held += 1
                if row["Close"] < entry_price * (1 - stop_pct):
                    pnl = (row["Close"] - entry_price) / entry_price * 100 - FEE_RT * 100
                    trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3),
                                   "reason": "STOP", "gap": round(df.iloc[i - days_held]["gap_pct"] * 100, 2)})
                    in_trade = False
                elif days_held >= hold_days:
                    pnl = (row["Close"] - entry_price) / entry_price * 100 - FEE_RT * 100
                    trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3),
                                   "reason": "TIME", "gap": 0})
                    in_trade = False
            if not in_trade and row["pos_event"]:
                entry_price, entry_date, days_held, in_trade = row["Open"] * (1 + FEE_BUY), df.index[i], 0, True

        tdf   = pd.DataFrame(trades)
        stats = base_stats(tdf)
        events = df[df["pos_event"]].tail(5)[["gap_pct", "vol_ratio"]].copy()
        events["gap_pct"]   = (events["gap_pct"] * 100).round(2)
        events["vol_ratio"] = events["vol_ratio"].round(2)
        recent_events = [{"date": str(d.date()), "gap": row["gap_pct"], "vol_ratio": row["vol_ratio"]}
                         for d, row in events.iterrows()]
        return jsonify({"ok": True, "stats": stats,
                        "recent_events": recent_events,
                        "total_events": int(df["pos_event"].sum()),
                        "trades": trades[-20:] if trades else []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — Intraday Gap Continuation
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/gap_continuation")
def api_gap_continuation():
    ticker      = request.args.get("ticker", "ANTM")
    gap_min     = float(request.args.get("gap_min", 1.5)) / 100
    vol_mult    = float(request.args.get("vol_mult", 1.5))

    try:
        df = fetch(ticker)
        df["gap_pct"]       = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["vol_ma20"]      = df["Volume"].rolling(20).mean()
        df["vol_ratio"]     = df["Volume"] / df["vol_ma20"]
        df["intraday_ret"]  = (df["Close"] - df["Open"]) / df["Open"]
        df["gap_up_signal"] = (df["gap_pct"] >= gap_min) & (df["vol_ratio"] >= vol_mult)
        df = df.dropna()

        # All gap-up days: measure intraday return
        gap_days  = df[df["gap_up_signal"]]
        all_days  = df
        gap_pnl   = gap_days["intraday_ret"] * 100 - FEE_RT * 100
        all_pnl   = all_days["intraday_ret"] * 100 - FEE_RT * 100

        def quick_stats(series):
            s = series.dropna()
            if s.empty: return {}
            return {
                "count":    len(s),
                "win_rate": round((s > 0).sum() / len(s) * 100, 1),
                "avg_ret":  round(s.mean(), 4),
                "avg_win":  round(s[s > 0].mean(), 3) if (s > 0).any() else 0,
                "avg_loss": round(s[s <= 0].mean(), 3) if (s <= 0).any() else 0,
                "total_pnl": round(s.sum(), 2),
            }

        recent = gap_days.tail(5)[["gap_pct", "vol_ratio", "intraday_ret"]].copy()
        recent_list = [{"date": str(d.date()),
                        "gap": round(row["gap_pct"]*100, 2),
                        "vol_ratio": round(row["vol_ratio"], 2),
                        "intraday_ret": round(row["intraday_ret"]*100, 2)}
                       for d, row in recent.iterrows()]

        # Is today a signal?
        last = df.iloc[-1]
        signal = "✅ GAP CONTINUATION TODAY" if last["gap_up_signal"] else "— No signal today"

        return jsonify({"ok": True,
                        "gap_days": quick_stats(gap_pnl),
                        "all_days":  quick_stats(all_pnl),
                        "recent_signals": recent_list,
                        "signal": signal,
                        "total_gap_days": int(df["gap_up_signal"].sum())})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6 — RSI Rotation
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/rsi_rotation")
def api_rsi_rotation():
    tickers_raw = request.args.get("tickers", "ANTM,INCO,NCKL,ADRO,PTBA,ITMG,MDKA,TINS")
    mom_window  = int(request.args.get("mom_window", 63))
    top_pct     = float(request.args.get("top_pct", 30)) / 100

    tickers = [t.strip() for t in tickers_raw.split(",")]

    try:
        raw = {}
        for t in tickers:
            try:
                s = fetch(t)["Close"].dropna()
                if len(s) > 120: raw[t] = s
            except: pass

        if len(raw) < 3:
            return jsonify({"ok": False, "error": "Need at least 3 stocks with data"})

        prices = pd.DataFrame(raw).dropna(how="all")
        mom    = prices.pct_change(mom_window)
        rsi_df = prices.apply(lambda x: calc_rsi(x, 14))

        # Current snapshot
        latest_mom = mom.iloc[-1].dropna().sort_values(ascending=False)
        latest_rsi = rsi_df.iloc[-1].dropna()
        n_top      = max(1, int(len(latest_mom) * top_pct))

        rankings = []
        for ticker, m in latest_mom.items():
            rsi_val = latest_rsi.get(ticker, 50)
            score   = m + 0.1 * (rsi_val - 50) / 50
            rankings.append({
                "ticker":   ticker,
                "momentum": round(m * 100, 2),
                "rsi":      round(rsi_val, 1),
                "score":    round(score * 100, 3),
                "rank":     0,
                "selected": False,
            })
        rankings.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
            r["selected"] = i < n_top

        # Monthly backtest
        monthly = prices.resample("ME").last()
        port_rets, bench_rets = [], []
        current_hold = []
        for i in range(1, len(monthly)):
            p_start = monthly.index[i-1]
            p_end   = monthly.index[i]
            score_row = mom.loc[:p_start].iloc[-1].dropna()
            if len(score_row) < 3: continue
            ranked_m  = score_row.sort_values(ascending=False)
            n         = max(1, int(len(ranked_m) * top_pct))
            new_hold  = ranked_m.head(n).index.tolist()
            period_p  = prices.loc[p_start:p_end]
            if len(period_p) < 2: continue
            period_ret = period_p.iloc[-1] / period_p.iloc[0] - 1
            port_ret   = period_ret[new_hold].mean()
            sells = set(current_hold) - set(new_hold)
            buys  = set(new_hold) - set(current_hold)
            fee   = len(sells) / max(len(current_hold),1) * FEE_SELL + len(buys) / max(len(new_hold),1) * FEE_BUY
            port_rets.append({"date": str(p_end.date()), "ret": round((port_ret - fee) * 100, 3)})
            bench_rets.append(round(period_ret[score_row.index].mean() * 100, 3))
            current_hold = new_hold

        port_series  = pd.Series([x["ret"] for x in port_rets]) / 100
        bench_series = pd.Series(bench_rets) / 100
        port_equity  = (1 + port_series).cumprod().tolist()
        bench_equity = (1 + bench_series).cumprod().tolist()

        total_days = (prices.index[-1] - prices.index[0]).days
        years = max(total_days / 365.25, 0.1)
        port_ann  = ((1 + port_series).prod() ** (1/years) - 1) * 100
        bench_ann = ((1 + bench_series).prod() ** (1/years) - 1) * 100

        return jsonify({"ok": True,
                        "rankings": rankings,
                        "top_picks": [r["ticker"] for r in rankings if r["selected"]],
                        "port_ann_ret":  round(port_ann, 1),
                        "bench_ann_ret": round(bench_ann, 1),
                        "port_equity":   port_equity,
                        "bench_equity":  bench_equity,
                        "monthly_rets":  port_rets[-12:]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 7 — Pairs Trading Divergence
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/pairs")
def api_pairs():
    tickers_raw = request.args.get("tickers", "ANTM,INCO,NCKL,ADRO,PTBA,MDKA")
    roll_win    = int(request.args.get("roll_win", 60))
    z_entry     = float(request.args.get("z_entry", 2.0))
    z_exit      = float(request.args.get("z_exit", 0.5))
    corr_min    = float(request.args.get("corr_min", 0.7))
    max_hold    = int(request.args.get("max_hold", 25))

    tickers = [t.strip() for t in tickers_raw.split(",")]

    try:
        raw = {}
        for t in tickers:
            try:
                s = fetch(t)["Close"].dropna()
                if len(s) > 120: raw[t] = s
            except: pass

        prices = pd.DataFrame(raw).dropna(how="all")
        valid  = list(prices.columns)
        results = []

        for tA, tB in combinations(valid, 2):
            pA = prices[tA].dropna()
            pB = prices[tB].dropna()
            common = pA.index.intersection(pB.index)
            pA, pB = pA[common], pB[common]
            if len(pA) < 120: continue

            ratio     = pA / pB
            roll_mean = ratio.rolling(roll_win).mean()
            roll_std  = ratio.rolling(roll_win).std()
            zscore    = (ratio - roll_mean) / roll_std
            corr      = pA.rolling(roll_win).corr(pB)
            df = pd.DataFrame({"pA": pA, "pB": pB, "zscore": zscore, "corr": corr}).dropna()

            trades, in_trade = [], False
            entry_date, entry_pA, entry_z, days_held = None, 0, 0, 0
            for i in range(len(df)):
                row = df.iloc[i]
                if in_trade:
                    days_held += 1
                    exit_reason = None
                    if row["zscore"] >= -z_exit: exit_reason = "REVERT"
                    if days_held >= max_hold:    exit_reason = "TIME"
                    if exit_reason:
                        pnl = (row["pA"] - entry_pA) / entry_pA * 100 - FEE_RT * 100
                        trades.append({"entry": entry_date, "exit": df.index[i], "pnl": round(pnl,3), "reason": exit_reason})
                        in_trade = False
                if not in_trade and row["corr"] >= corr_min and row["zscore"] <= -z_entry:
                    in_trade, entry_date, entry_pA, entry_z, days_held = True, df.index[i], row["pA"], row["zscore"], 0

            tdf   = pd.DataFrame(trades)
            stats = base_stats(tdf)

            current_z    = round(df["zscore"].iloc[-1], 3)
            current_corr = round(df["corr"].iloc[-1], 3)
            live_signal  = bool(current_z <= -z_entry and current_corr >= corr_min)

            results.append({
                "pair":         f"{tA}/{tB}",
                "tickerA":      tA,
                "tickerB":      tB,
                "stats":        stats,
                "current_z":    current_z,
                "current_corr": current_corr,
                "live_signal":  live_signal,
            })

        results.sort(key=lambda x: x["stats"].get("total_pnl", 0), reverse=True)

        # Correlation matrix
        returns = prices.pct_change().dropna()
        corr_matrix = returns.corr().round(3)
        corr_data = {
            "tickers": [t for t in corr_matrix.columns],
            "matrix":  [[float(v) for v in row] for row in corr_matrix.values.tolist()],
        }

        return jsonify({"ok": True, "pairs": results, "corr_matrix": corr_data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ── Serve dashboard ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("dashboard.html")

if __name__ == "__main__":
    print("\n🚀 IDX Quant Dashboard running at http://localhost:5000\n")
    app.run(debug=False, port=5000)