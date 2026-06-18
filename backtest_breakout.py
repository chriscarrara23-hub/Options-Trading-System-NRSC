#!/usr/bin/env python3
"""
Pullback-Momentum Breakout Options Backtest — SPY + QQQ, 2020-2025.

BULLISH signal (all 4 required):
  1. New 20-day closing high set within the last 5 trading days
  2. Price has since pulled back 1-5% from that 20-day high
  3. Today's close > yesterday's close AND above 9-day EMA
  4. Today's volume > 20-day average volume

BEARISH is the exact mirror (20-day low, 1-5% bounce, close below prev & EMA9, vol above avg).

Risk filters (unchanged from prior tests):
  - VIX 15-35
  - Proxy IV rank ≤ 50 (30-day realized vol percentile)
"""

import json
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TICKERS     = ['SPY', 'QQQ']
YEARS       = [2020, 2021, 2022, 2023, 2024, 2025]
VIX_LOW     = 15
VIX_HIGH    = 35
DTE         = 7
RISK_FREE   = 0.05
STOP_PCT    = -0.45
TAKE_HALF   = 0.50
FULL_EXIT   = 1.00
MAX_HOLD    = 7
IV_RANK_MAX = 50
COOLDOWN_N  = 3
TODAY       = datetime.today().strftime('%Y-%m-%d')
OUTFILE     = '/Users/noahrourke/trading-system/breakout_backtest_results.json'


# ── OPTION PRICING ────────────────────────────────────────────────────────────

def bs(S, K, T, r, sig, typ='call'):
    if T < 1e-6 or sig < 1e-6 or S <= 0 or K <= 0:
        iv = max(S-K, 0) if typ == 'call' else max(K-S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S/K) + (r + .5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    if typ == 'call':
        return max(S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2), 0.01)
    return max(K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), 0.01)


def simulate_exit(entry_px, fwd, sigma, direction):
    K    = entry_px
    typ  = 'call' if direction == 'bullish' else 'put'
    e0   = bs(entry_px, K, DTE/365, RISK_FREE, sigma, typ)
    half = False

    for day, S in enumerate(fwd[:MAX_HOLD], 1):
        T   = max((DTE - day) / 365, 0.001)
        cur = bs(float(S), K, T, RISK_FREE, sigma, typ)
        pnl = (cur - e0) / e0
        if pnl <= STOP_PCT:
            return float(S), day, 'STOP', pnl
        if pnl >= FULL_EXIT:
            return float(S), day, 'FULL_TARGET', pnl
        if not half and pnl >= TAKE_HALF:
            half = True
        if half and pnl < TAKE_HALF * 0.5:
            return float(S), day, 'TRAIL_EXIT', (TAKE_HALF + pnl) / 2

    if not fwd:
        return entry_px, 0, 'NO_DATA', 0.0
    day = min(MAX_HOLD, len(fwd))
    S   = float(fwd[day - 1])
    cur = bs(S, K, max((DTE - day)/365, 0.001), RISK_FREE, sigma, typ)
    return S, day, 'TIME_EXIT', (cur - e0) / e0


# ── VOLATILITY HELPERS ────────────────────────────────────────────────────────

def rv30_series(close):
    return np.log(close / close.shift(1)).rolling(30).std() * math.sqrt(252)


def get_iv_rank(rv_ser, as_of, lookback=252):
    idx = rv_ser.index.searchsorted(as_of, side='right') - 1
    if idx < lookback:
        return None
    curr = rv_ser.iloc[idx]
    hist = rv_ser.iloc[idx - lookback:idx].dropna()
    if pd.isna(curr) or len(hist) < 20:
        return None
    return float((hist < curr).sum() / len(hist) * 100)


def get_sigma(close, as_of):
    idx = close.index.searchsorted(as_of, side='right')
    sub = close.iloc[max(0, idx-31):idx]
    if len(sub) < 10:
        return None
    return float(np.log(sub / sub.shift(1)).dropna().std() * math.sqrt(252))


# ── SIGNAL INDICATORS ─────────────────────────────────────────────────────────

def add_breakout_signals(df):
    """
    Compute all signal components and attach to df.
    Returns df with 'bull_signal', 'bear_signal', and component columns added.
    """
    df   = df.copy()
    c    = df['Close']
    v    = df['Volume']
    ema9 = c.ewm(span=9, adjust=False).mean()
    vm20 = v.rolling(20).mean()

    # ── Condition 1 ───────────────────────────────────────────────────────────
    # New 20-day closing high: today's close >= max of prior 20 closes
    roll20_high = c.rolling(20).max()
    roll20_low  = c.rolling(20).min()
    is_20d_high = c >= roll20_high.shift(1)   # avoids using today in its own max
    is_20d_low  = c <= roll20_low.shift(1)

    # Did a new 20-day high/low occur within the last 5 bars (including today)?
    had_high_5d = is_20d_high.rolling(5).max().fillna(0).astype(bool)
    had_low_5d  = is_20d_low.rolling(5).max().fillna(0).astype(bool)

    # ── Condition 2 ───────────────────────────────────────────────────────────
    # Rolling 5-day peak/trough of Close (includes today)
    peak_5d   = c.rolling(5).max()
    trough_5d = c.rolling(5).min()

    # Pullback from peak (bull): today's close is 1-5% below the 5-day peak
    pullback = (peak_5d - c) / peak_5d.replace(0, np.nan)
    in_pullback_range = (pullback >= 0.01) & (pullback <= 0.05)

    # Bounce from trough (bear): today's close is 1-5% above the 5-day trough
    bounce = (c - trough_5d) / trough_5d.replace(0, np.nan)
    in_bounce_range = (bounce >= 0.01) & (bounce <= 0.05)

    # ── Condition 3 ───────────────────────────────────────────────────────────
    resuming_bull = (c > c.shift(1)) & (c > ema9)
    resuming_bear = (c < c.shift(1)) & (c < ema9)

    # ── Condition 4 ───────────────────────────────────────────────────────────
    vol_ok = v > vm20

    # ── Composite signals ─────────────────────────────────────────────────────
    df['bull_signal'] = had_high_5d & in_pullback_range & resuming_bull & vol_ok
    df['bear_signal'] = had_low_5d  & in_bounce_range  & resuming_bear & vol_ok

    # Store diagnostics
    df['pullback_pct'] = (pullback * 100).round(2)
    df['bounce_pct']   = (bounce   * 100).round(2)
    df['ema9']         = ema9.round(2)
    df['vol_ratio']    = (v / vm20).round(2)

    return df.dropna(subset=['bull_signal', 'bear_signal'])


# ── STATISTICS ────────────────────────────────────────────────────────────────

def compute_stats(trades):
    if not trades:
        return {}
    pnls   = [t['pnl_pct']/100 for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    max_cl = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        max_cl = max(max_cl, cur)

    eq = pk = 1.0; mdd = 0.0
    for p in pnls:
        eq *= (1 + p * 0.1)
        if eq > pk: pk = eq
        mdd = max(mdd, (pk - eq) / pk)

    arr    = np.array(pnls)
    sharpe = float(arr.mean() / arr.std() * math.sqrt(252)) if arr.std() > 1e-9 else 0.0

    bull = [t for t in trades if t['direction'] == 'bullish']
    bear = [t for t in trades if t['direction'] == 'bearish']

    return {
        'total_trades':      len(trades),
        'wins':              len(wins),
        'losses':            len(losses),
        'win_rate':          round(len(wins)/len(trades), 4),
        'avg_winner_pct':    round(np.mean(wins)*100,  2) if wins   else 0,
        'avg_loser_pct':     round(np.mean(losses)*100,2) if losses else 0,
        'max_consec_losses': max_cl,
        'max_drawdown_pct':  round(mdd*100, 2),
        'sharpe':            round(sharpe, 3),
        'bullish_trades':    len(bull),
        'bearish_trades':    len(bear),
        'bull_win_rate':     round(len([t for t in bull if t['pnl_pct']>0])/len(bull), 4) if bull else 0,
        'bear_win_rate':     round(len([t for t in bear if t['pnl_pct']>0])/len(bear), 4) if bear else 0,
    }


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def download_all():
    print("▶  Downloading data …")
    daily = {}
    for t in TICKERS:
        print(f"   {t} daily … ", end='', flush=True)
        df = yf.download(t, start='2019-01-01', end=TODAY,
                         interval='1d', auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        daily[t] = df
        print(f"{len(df)} bars")

    print(f"   VIX daily … ", end='', flush=True)
    vix = yf.download('^VIX', start='2019-01-01', end=TODAY,
                      interval='1d', auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    print(f"{len(vix)} bars")
    return daily, vix


# ── CORE BACKTEST ─────────────────────────────────────────────────────────────

def run_breakout_backtest(daily, vix_df):
    vix_close = vix_df['Close']
    rv_sers   = {t: rv30_series(daily[t]['Close']) for t in TICKERS}

    # Pre-compute signal frames (full history, then slice by year)
    signal_frames = {}
    for t in TICKERS:
        signal_frames[t] = add_breakout_signals(daily[t])

    all_trades   = []
    year_results = {}

    for year in YEARS:
        y0 = pd.Timestamp(f'{year}-01-01')
        y1 = pd.Timestamp(f'{year}-12-31')

        year_trades = []
        total_sig = iv_skip = cd_skip = 0
        raw_sigs  = []

        for ticker in TICKERS:
            sf    = signal_frames[ticker]
            sf_yr = sf[(sf.index >= y0) & (sf.index <= y1)]

            for ts, row in sf_yr.iterrows():
                is_bull = bool(row['bull_signal'])
                is_bear = bool(row['bear_signal'])
                if not is_bull and not is_bear:
                    continue
                direction = 'bullish' if is_bull else 'bearish'
                raw_sigs.append({
                    'ts':          ts,
                    'ticker':      ticker,
                    'direction':   direction,
                    'close':       float(row['Close']),
                    'pullback':    float(row['pullback_pct']) if is_bull else float(row['bounce_pct']),
                    'vol_ratio':   float(row['vol_ratio']),
                })

        raw_sigs.sort(key=lambda x: x['ts'])
        day_streak = {}

        for sig in raw_sigs:
            total_sig += 1
            t   = sig['ticker']
            ts  = sig['ts']
            sd  = pd.Timestamp(ts.date())
            c   = sig['close']

            # VIX filter
            vix_row = vix_close[vix_close.index <= sd]
            if vix_row.empty:
                continue
            vix_val = float(vix_row.iloc[-1])
            if pd.isna(vix_val) or not (VIX_LOW <= vix_val <= VIX_HIGH):
                continue

            # IV rank filter
            iv_rank = get_iv_rank(rv_sers[t], sd)
            if iv_rank is None or iv_rank > IV_RANK_MAX:
                iv_skip += 1
                continue

            # Cooldown
            date_key = ts.date()
            streak   = day_streak.get(date_key, 0)
            if streak >= COOLDOWN_N:
                cd_skip += 1
                continue

            # Sigma
            sigma = get_sigma(daily[t]['Close'], sd)
            if sigma is None or sigma <= 0:
                continue

            # Forward closes
            cd      = daily[t]['Close']
            fwd_idx = cd.index.searchsorted(sd, side='right')
            fwd     = list(cd.iloc[fwd_idx:fwd_idx + MAX_HOLD])
            if len(fwd) < 2:
                continue

            ex_px, days, ex_type, pnl = simulate_exit(c, fwd, sigma, sig['direction'])
            result = 'WIN' if pnl > 0 else 'LOSS'

            day_streak[date_key] = (streak + 1) if result == 'LOSS' else 0

            trade = {
                'date':          str(ts.date()),
                'ticker':        t,
                'direction':     sig['direction'],
                'pullback_pct':  sig['pullback'],
                'vol_ratio':     sig['vol_ratio'],
                'proxy_iv_rank': round(iv_rank, 1),
                'sigma':         round(sigma, 4),
                'entry_price':   round(c, 2),
                'exit_price':    round(ex_px, 2),
                'days_held':     days,
                'exit_type':     ex_type,
                'pnl_pct':       round(pnl * 100, 2),
                'result':        result,
                'year':          year,
            }
            year_trades.append(trade)
            all_trades.append(trade)

        stats = compute_stats(year_trades)
        wr    = stats.get('win_rate', 0)
        year_results[year] = {
            'year':           year,
            'total_signals':  total_sig,
            'iv_skips':       iv_skip,
            'cooldown_skips': cd_skip,
            'signals_taken':  len(year_trades),
            'win_rate':       round(wr, 4),
            'stats':          stats,
        }

    return year_results, all_trades


# ── REPORTING ─────────────────────────────────────────────────────────────────

def print_table(year_results, all_trades):
    print(f"\n{'='*72}")
    print("  BREAKOUT PULLBACK-MOMENTUM BACKTEST  |  SPY + QQQ  |  2020–2025")
    print(f"{'='*72}")
    hdr = (f"{'Year':<5}  {'Signals':>7}  {'IVSkip':>7}  {'CDSkip':>7}  "
           f"{'Taken':>6}  {'Win%':>6}  {'AvgW%':>6}  {'AvgL%':>6}  "
           f"{'MaxDD%':>7}  {'Sharpe':>7}")
    print(hdr)
    print("─"*72)

    sharpes  = []
    beats_55 = 0
    below_50 = []

    for year in YEARS:
        yr = year_results.get(year, {})
        s  = yr.get('stats', {})
        n  = yr.get('signals_taken', 0)
        wr = s.get('win_rate', 0)
        sh = s.get('sharpe', 0)
        if n > 0:
            sharpes.append(sh)
        if n > 0 and wr >= 0.55: beats_55 += 1
        if n > 0 and wr < 0.50:  below_50.append(year)
        flag = " ⚠" if (n > 0 and wr < 0.50) else (" ✓" if (n > 0 and wr >= 0.55) else "")

        print(f"{year:<5}  "
              f"{yr.get('total_signals',0):>7}  "
              f"{yr.get('iv_skips',0):>7}  "
              f"{yr.get('cooldown_skips',0):>7}  "
              f"{n:>6}  "
              f"{wr:>6.1%}  "
              f"{s.get('avg_winner_pct',0):>6.1f}  "
              f"{s.get('avg_loser_pct',0):>6.1f}  "
              f"{s.get('max_drawdown_pct',0):>7.1f}  "
              f"{s.get('sharpe',0):>7.2f}{flag}")

    print("─"*72)
    n_all  = len(all_trades)
    wr_all = len([t for t in all_trades if t['result']=='WIN']) / n_all if n_all else 0
    print(f"\nCombined  {n_all} trades | {wr_all:.1%} win rate | "
          f"{beats_55}/6 years ≥55% | below 50%: {below_50 or 'none'}")

    return wr_all, sharpes, beats_55


def print_three_way(yr_results, all_trades):
    """Three-way comparison: Breakout vs Original vs IBS Loose."""

    # ── Per-year win rates ────────────────────────────────────────────────────
    orig = {2020:72.7, 2021:31.7, 2022:42.9, 2023:65.8, 2024:52.8, 2025:34.0}
    ibs  = {2020:31.8, 2021:40.2, 2022:58.1, 2023:38.8, 2024:42.4, 2025:28.4}

    brk_wr  = len([t for t in all_trades if t['result']=='WIN']) / len(all_trades) if all_trades else 0
    orig_wr = 0.497
    ibs_wr  = 0.386

    brk_beats = sum(1 for y in YEARS
                    if yr_results[y]['signals_taken'] > 0
                    and yr_results[y]['win_rate'] >= 0.55)

    # Per-year win rates for std-dev
    brk_rates  = [yr_results[y]['win_rate']*100 for y in YEARS if yr_results[y]['signals_taken'] > 0]
    orig_rates = list(orig.values())
    ibs_rates  = list(ibs.values())

    # Combined Sharpe approximation (use per-trade pnl series)
    pnls = [t['pnl_pct']/100 for t in all_trades]
    arr  = np.array(pnls)
    brk_sharpe  = float(arr.mean()/arr.std()*math.sqrt(252)) if arr.std() > 1e-9 and len(arr) else 0

    # Hardcoded from prior runs
    orig_sharpe_avg = (10.73 - 1.21 - 3.76 + 11.87 + 6.31 - 2.21) / 6   # avg per-year Sharpe
    ibs_sharpe_avg  = (0.55 + 2.85 + 8.04 + 1.49 + 2.00 + 0.56) / 6

    print(f"\n{'='*72}")
    print("  THREE-WAY COMPARISON")
    print(f"{'='*72}")
    print(f"\n  {'Year':<5}  {'Original':>11}  {'IBS Loose':>11}  {'Breakout':>10}")
    print(f"  {'':5}  {'(win%)':>11}  {'(win%)':>11}  {'(win%)':>10}")
    print(f"  {'─'*46}")

    for year in YEARS:
        ow = orig.get(year, 0)
        iw = ibs.get(year, 0)
        bw = yr_results[year]['win_rate']*100
        bn = yr_results[year]['signals_taken']
        b_flag = " ✓" if bw >= 55 else (" ⚠" if (bn > 0 and bw < 50) else "")
        print(f"  {year:<5}  {ow:>10.1f}%  {iw:>10.1f}%  {bw:>9.1f}%{b_flag}")

    print(f"  {'─'*46}")
    print(f"  {'All':<5}  {orig_wr:>10.1%}  {ibs_wr:>10.1%}  {brk_wr:>9.1%}")

    print(f"\n  ── Summary metrics ──────────────────────────────────────────")
    metrics = [
        ("Total trades",    "318",                    "581",                f"{len(all_trades)}"),
        ("Combined win%",   f"{orig_wr:.1%}",         f"{ibs_wr:.1%}",      f"{brk_wr:.1%}"),
        ("Years ≥ 55%",    "2/6",                    "1/6",                f"{brk_beats}/6"),
        ("Win% std-dev (σ)",f"{np.std(orig_rates):.1f}pp",
                            f"{np.std(ibs_rates):.1f}pp",
                            f"{np.std(brk_rates):.1f}pp  ← lower = more consistent"),
        ("Avg annual Sharpe",f"{orig_sharpe_avg:.2f}", f"{ibs_sharpe_avg:.2f}", f"{brk_sharpe:.2f}"),
    ]
    for label, ov, iv, bv in metrics:
        print(f"  {label:<22}  {ov:>12}  {iv:>12}  {bv}")

    # ── Winner on each metric ─────────────────────────────────────────────────
    print(f"\n  ── Best on each metric ──────────────────────────────────────")
    winners = {
        "Highest combined win rate": ("Original (49.7%)"  if orig_wr > ibs_wr and orig_wr > brk_wr
                                      else ("IBS Loose (38.6%)" if ibs_wr > brk_wr else f"Breakout ({brk_wr:.1%})")),
        "Most consistent (lowest σ)": ("Original" if np.std(orig_rates) < np.std(ibs_rates) and np.std(orig_rates) < np.std(brk_rates)
                                       else ("IBS Loose" if np.std(ibs_rates) < np.std(brk_rates) else "Breakout")),
        "Best combined Sharpe":       ("Original" if orig_sharpe_avg > ibs_sharpe_avg and orig_sharpe_avg > brk_sharpe
                                       else ("IBS Loose" if ibs_sharpe_avg > brk_sharpe else "Breakout")),
    }
    for metric, winner in winners.items():
        print(f"  {metric:<30}  {winner}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("  BREAKOUT PULLBACK-MOMENTUM BACKTEST  |  SPY + QQQ  |  2020–2025")
    print("="*72)

    daily, vix_df = download_all()

    print("\n▶  Computing signals and running backtest …")
    yr_results, all_trades = run_breakout_backtest(daily, vix_df)

    brk_wr, sharpes, beats_55 = print_table(yr_results, all_trades)
    print_three_way(yr_results, all_trades)

    # Save
    output = {
        'generated':       datetime.now().isoformat(),
        'strategy':        'breakout_pullback_momentum',
        'total_trades':    len(all_trades),
        'overall_win_rate': round(brk_wr, 4),
        'years':           {str(y): yr_results[y] for y in YEARS},
        'last_20_trades':  sorted(all_trades, key=lambda t: t['date'])[-20:],
    }
    with open(OUTFILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Saved {OUTFILE}")


if __name__ == '__main__':
    main()
