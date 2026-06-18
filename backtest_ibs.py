#!/usr/bin/env python3
"""
IBS (Internal Bar Strength) mean-reversion options backtest.
IBS = (Close - Low) / (High - Low)

Bullish signal: IBS < threshold  → buy ATM call (closed near low, bounce candidate)
Bearish signal: IBS > 1-threshold → buy ATM put  (closed near high, pullback candidate)

Risk filters kept from original system:
  - VIX 15-35 (regime filter)
  - Proxy IV rank ≤ 50 (realized vol percentile filter)

No trend/momentum conditions added — IBS is the sole signal per academic spec.
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
OUTFILE     = '/Users/noahrourke/trading-system/ibs_backtest_results.json'


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
    """Simulate 7-DTE ATM option exit using forward daily closes."""
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
        'avg_winner_pct':    round(np.mean(wins)*100, 2)   if wins   else 0,
        'avg_loser_pct':     round(np.mean(losses)*100, 2) if losses else 0,
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

def run_ibs_backtest(daily, vix_df, threshold):
    """
    Run IBS backtest for both tickers, all years.
    threshold: IBS < threshold → bullish, IBS > (1-threshold) → bearish
    """
    vix_close = vix_df['Close']
    rv_sers   = {t: rv30_series(daily[t]['Close']) for t in TICKERS}

    all_trades   = []
    year_results = {}

    for year in YEARS:
        y0 = pd.Timestamp(f'{year}-01-01')
        y1 = pd.Timestamp(f'{year}-12-31')

        year_trades = []
        total_sig = iv_skip = cd_skip = 0
        raw_sigs  = []

        for ticker in TICKERS:
            df    = daily[ticker]
            df_yr = df[(df.index >= y0) & (df.index <= y1)].copy()

            # IBS per bar
            rng            = df_yr['High'] - df_yr['Low']
            df_yr = df_yr.copy()
            df_yr['ibs']   = (df_yr['Close'] - df_yr['Low']) / rng.replace(0, np.nan)

            for ts, row in df_yr.iterrows():
                ibs_val = row['ibs']
                if pd.isna(ibs_val):
                    continue
                if ibs_val < threshold:
                    direction = 'bullish'
                elif ibs_val > (1 - threshold):
                    direction = 'bearish'
                else:
                    continue

                raw_sigs.append({
                    'ts':        ts,
                    'ticker':    ticker,
                    'direction': direction,
                    'close':     float(row['Close']),
                    'ibs':       round(float(ibs_val), 4),
                })

        # Process in chronological order (both tickers interleaved)
        raw_sigs.sort(key=lambda x: x['ts'])
        day_streak = {}   # date → consecutive-loss count

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

            # Cooldown check
            date_key = ts.date()
            streak   = day_streak.get(date_key, 0)
            if streak >= COOLDOWN_N:
                cd_skip += 1
                continue

            # Sigma for Black-Scholes
            sigma = get_sigma(daily[t]['Close'], sd)
            if sigma is None or sigma <= 0:
                continue

            # Forward closes for exit simulation
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
                'ibs':           sig['ibs'],
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


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_table(year_results, all_trades, label, threshold):
    print(f"\n{'='*72}")
    print(f"  IBS {label.upper()}  |  bull: IBS < {threshold:.2f}   bear: IBS > {1-threshold:.2f}")
    print(f"{'='*72}")
    hdr = (f"{'Year':<5}  {'Signals':>7}  {'IVSkip':>7}  {'CDSkip':>7}  "
           f"{'Taken':>6}  {'Win%':>6}  {'AvgW%':>6}  {'AvgL%':>6}  "
           f"{'MaxDD%':>7}  {'Sharpe':>7}")
    print(hdr)
    print("─"*72)

    beats_55 = 0
    below_50 = []

    for year in YEARS:
        yr = year_results.get(year, {})
        s  = yr.get('stats', {})
        n  = yr.get('signals_taken', 0)
        wr = s.get('win_rate', 0)
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


# ── COMPARISON SUMMARY ────────────────────────────────────────────────────────

def print_comparison(results_by_label):
    print(f"\n{'='*72}")
    print("  COMPARISON: ORIGINAL 4-CONDITION SIGNAL  vs  IBS VARIANTS")
    print(f"{'='*72}")

    # Hardcoded original results (3/4 cond, 3-bar RSI, VIX 15-35, 318 trades)
    original = {
        2020: (11,  72.7), 2021: (41,  31.7), 2022: (7,  42.9),
        2023: (123, 65.8), 2024: (36,  52.8), 2025: (100, 34.0),
    }
    orig_total = 318; orig_wr = 49.7

    print(f"\n  {'Year':<5}  {'Original':>10}  {'IBS Loose':>11}  {'IBS Strict':>11}")
    print(f"  {'':5}  {'(trades/win%)':>10}  {'(trades/win%)':>11}  {'(trades/win%)':>11}")
    print(f"  {'─'*52}")

    for year in YEARS:
        on, ow = original.get(year, (0, 0))
        loose  = results_by_label['loose']['year_results'][year]
        strict = results_by_label['strict']['year_results'][year]
        ln, lw = loose['signals_taken'], loose['win_rate']*100
        sn, sw = strict['signals_taken'], strict['win_rate']*100
        print(f"  {year:<5}  {on:>4}  {ow:>5.1f}%   {ln:>4}  {lw:>5.1f}%   {sn:>4}  {sw:>5.1f}%")

    print(f"  {'─'*52}")

    # Totals
    for label, data in results_by_label.items():
        beats = sum(1 for y in YEARS
                    if data['year_results'][y]['signals_taken'] > 0
                    and data['year_results'][y]['win_rate'] >= 0.55)
        print(f"  IBS {label:6s}: {data['total_trades']:4d} trades | "
              f"{data['overall_win_rate']:.1%} win rate | {beats}/6 years ≥55%")

    print(f"  Original:   {orig_total:4d} trades | {orig_wr/100:.1%} win rate | "
          f"{sum(1 for _,w in original.values() if w>=55)}/6 years ≥55%")

    print(f"\n  Consistency note (std-dev of annual win rates):")
    for label, data in results_by_label.items():
        rates = [data['year_results'][y]['win_rate']*100
                 for y in YEARS if data['year_results'][y]['signals_taken'] > 0]
        std = np.std(rates) if rates else 0
        print(f"  IBS {label:6s}: σ={std:.1f}pp across {len(rates)} active years")
    orig_rates = [w for _,w in original.values()]
    print(f"  Original:   σ={np.std(orig_rates):.1f}pp across 6 years")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("  IBS MEAN-REVERSION OPTIONS BACKTEST  |  SPY + QQQ  |  2020–2025")
    print("="*72)

    daily, vix_df = download_all()

    run_configs = [('loose', 0.20), ('strict', 0.10)]
    results_by_label = {}
    all_output = {'generated': datetime.now().isoformat()}

    for label, threshold in run_configs:
        print(f"\n▶  Running IBS {label} (threshold={threshold}) …")
        yr, trades = run_ibs_backtest(daily, vix_df, threshold)
        print_table(yr, trades, label, threshold)

        overall_wr = (len([t for t in trades if t['result']=='WIN']) / len(trades)
                      if trades else 0)
        results_by_label[label] = {
            'threshold':       threshold,
            'total_trades':    len(trades),
            'overall_win_rate': round(overall_wr, 4),
            'year_results':    yr,
        }
        all_output[label] = {
            **results_by_label[label],
            'year_results': {str(y): yr[y] for y in YEARS},
            'last_20_trades': sorted(trades, key=lambda t: t['date'])[-20:],
        }

    print_comparison(results_by_label)

    with open(OUTFILE, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\n✓ Saved {OUTFILE}")


if __name__ == '__main__':
    main()
