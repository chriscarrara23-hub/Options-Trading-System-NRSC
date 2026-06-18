#!/usr/bin/env python3
"""
One-parameter-at-a-time sensitivity analysis on the original 4-condition strategy.

Baseline: 3/4 conditions required, 3-bar RSI, VIX 15-35, EMA21, MACD expansion,
          volume > 20-day avg, -45% stop, +50%/+100% targets.

Each parameter group is tested in isolation — nothing else changes.
No combined "best" parameters are assembled at the end.
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
TICKERS    = ['SPY', 'QQQ']
YEARS      = [2020, 2021, 2022, 2023, 2024, 2025]
TODAY      = datetime.today().strftime('%Y-%m-%d')
OUTFILE    = '/Users/noahrourke/trading-system/sensitivity_results.json'
COOLDOWN_N = 3
IV_RANK_MAX = 50

BASELINE = dict(
    rsi_low   = 30,   rsi_high  = 50,
    vix_low   = 15,   vix_high  = 35,
    vol_mult  = 1.0,
    stop_pct  = -0.45,
    take_half = 0.50,
    full_exit = 1.00,
    min_score = 3,
)

# ── PARAMETER VARIANTS (one group at a time) ──────────────────────────────────
VARIANT_GROUPS = {
    'RSI range': [
        ('RSI 25-55 (wider)',    dict(rsi_low=25, rsi_high=55)),
        ('RSI 30-50 (baseline)', dict()),
        ('RSI 35-45 (narrower)', dict(rsi_low=35, rsi_high=45)),
    ],
    'VIX range': [
        ('VIX 15-35 (baseline)', dict()),
        ('VIX 18-32',            dict(vix_low=18, vix_high=32)),
        ('VIX 20-30 (tight)',    dict(vix_low=20, vix_high=30)),
    ],
    'Volume threshold': [
        ('>1.0x avg (baseline)', dict()),
        ('>1.2x avg (stricter)', dict(vol_mult=1.2)),
        ('>1.5x avg (strictest)',dict(vol_mult=1.5)),
    ],
    'Stop loss': [
        ('-45% (baseline)',      dict()),
        ('-35% (tighter)',       dict(stop_pct=-0.35)),
        ('-55% (wider)',         dict(stop_pct=-0.55)),
    ],
    'Profit targets': [
        ('+50%/+100% (baseline)',dict()),
        ('+40%/+80% (tighter)',  dict(take_half=0.40, full_exit=0.80)),
        ('+60%/+120% (wider)',   dict(take_half=0.60, full_exit=1.20)),
    ],
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, n=14):
    d  = s.diff()
    ag = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def macd_hist(s, f=12, sl=26, sig=9):
    m = ema(s, f) - ema(s, sl)
    return m - ema(m, sig)

def add_indicators(df):
    df = df.copy()
    c, v  = df['Close'], df['Volume']
    df['ema21']      = ema(c, 21)
    df['rsi']        = rsi_series(c)
    df['mhist']      = macd_hist(c)
    df['volma']      = v.rolling(20).mean()
    r, m = df['rsi'], df['mhist']
    df['rsi_up3']    = (r > r.shift(1)) & (r.shift(1) > r.shift(2))
    df['rsi_dn3']    = (r < r.shift(1)) & (r.shift(1) < r.shift(2))
    df['mhist_bull'] = (m > 0) & (m > m.shift(1))
    df['mhist_bear'] = (m < 0) & (m < m.shift(1))
    return df.dropna()

def bs(S, K, T, r, sig, typ='call'):
    if T < 1e-6 or sig < 1e-6 or S <= 0 or K <= 0:
        iv = max(S-K, 0) if typ == 'call' else max(K-S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S/K) + (r + .5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    if typ == 'call':
        return max(S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2), 0.01)
    return max(K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), 0.01)

def simulate_exit(entry_px, fwd, sigma, direction, p):
    K    = entry_px
    typ  = 'call' if direction == 'bullish' else 'put'
    DTE  = 7
    RISK = 0.05
    e0   = bs(entry_px, K, DTE/365, RISK, sigma, typ)
    half = False
    for day, S in enumerate(fwd[:7], 1):
        T   = max((DTE - day) / 365, 0.001)
        cur = bs(float(S), K, T, RISK, sigma, typ)
        pnl = (cur - e0) / e0
        if pnl <= p['stop_pct']:
            return float(S), day, 'STOP', pnl
        if pnl >= p['full_exit']:
            return float(S), day, 'FULL_TARGET', pnl
        if not half and pnl >= p['take_half']:
            half = True
        if half and pnl < p['take_half'] * 0.5:
            return float(S), day, 'TRAIL_EXIT', (p['take_half'] + pnl) / 2
    if not fwd:
        return entry_px, 0, 'NO_DATA', 0.0
    day = min(7, len(fwd))
    S   = float(fwd[day-1])
    cur = bs(S, K, max((7-day)/365, 0.001), RISK, sigma, typ)
    return S, day, 'TIME_EXIT', (cur - e0) / e0

def rv30_series(close):
    return np.log(close / close.shift(1)).rolling(30).std() * math.sqrt(252)

def get_iv_rank(rv_ser, as_of, lookback=252):
    idx = rv_ser.index.searchsorted(as_of, side='right') - 1
    if idx < lookback:
        return None
    curr = rv_ser.iloc[idx]
    hist = rv_ser.iloc[idx-lookback:idx].dropna()
    if pd.isna(curr) or len(hist) < 20:
        return None
    return float((hist < curr).sum() / len(hist) * 100)

def get_sigma(close, as_of):
    idx = close.index.searchsorted(as_of, side='right')
    sub = close.iloc[max(0, idx-31):idx]
    if len(sub) < 10:
        return None
    return float(np.log(sub / sub.shift(1)).dropna().std() * math.sqrt(252))

def combined_stats(trades):
    if not trades:
        return 0, 0, 0
    pnls = [t['pnl_pct']/100 for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr   = wins / len(pnls)
    arr  = np.array(pnls)
    sh   = float(arr.mean()/arr.std()*math.sqrt(252)) if arr.std() > 1e-9 else 0.0
    beats = {}
    by_year = {}
    for t in trades:
        yr = t['year']
        by_year.setdefault(yr, []).append(t['pnl_pct'])
    beats = sum(1 for yr, ps in by_year.items()
                if len(ps) > 0 and sum(1 for p in ps if p > 0)/len(ps) >= 0.55)
    return wr, sh, beats

def per_year_wr(trades):
    by_year = {}
    for t in trades:
        by_year.setdefault(t['year'], []).append(t['pnl_pct'])
    return {yr: (sum(1 for p in ps if p > 0)/len(ps) if ps else 0)
            for yr, ps in by_year.items()}


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def download_all():
    print("▶  Downloading data (once, shared across all variants) …")
    hourly = {}
    for t in TICKERS:
        print(f"   {t} 1H … ", end='', flush=True)
        h = yf.download(t, period='730d', interval='1h', auto_adjust=True, progress=False)
        h = flatten(h)
        h.index = pd.to_datetime(h.index)
        if h.index.tz:
            h.index = h.index.tz_convert('US/Eastern').tz_localize(None)
        hourly[t] = h
        print(f"{len(h)} bars [{h.index[0].date() if len(h) else 'none'} → {h.index[-1].date() if len(h) else 'none'}]")

    daily = {}
    for t in TICKERS:
        print(f"   {t} 1D … ", end='', flush=True)
        d = yf.download(t, start='2019-01-01', end=TODAY, interval='1d', auto_adjust=True, progress=False)
        d = flatten(d)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        daily[t] = d
        print(f"{len(d)} bars")

    print(f"   VIX 1D … ", end='', flush=True)
    vix = yf.download('^VIX', start='2019-01-01', end=TODAY, interval='1d', auto_adjust=True, progress=False)
    vix = flatten(vix)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    print(f"{len(vix)} bars")
    return hourly, daily, vix


# ── INDICATOR PRE-COMPUTE ─────────────────────────────────────────────────────

def build_frames(hourly, daily):
    """Compute indicators on all data once. Indicators don't change between variants."""
    h_ind = {t: add_indicators(hourly[t]) if len(hourly[t]) >= 60 else pd.DataFrame()
             for t in TICKERS}
    d_ind = {t: add_indicators(daily[t]) for t in TICKERS}
    return h_ind, d_ind


# ── PARAMETRIC SIGNAL + BACKTEST ──────────────────────────────────────────────

def gen_signals_for_year(year, ticker, h_ind, d_ind, spy_d, vix_close, p):
    """Generate raw signals for one year+ticker with the given parameter set."""
    y0 = pd.Timestamp(f'{year}-01-01')
    y1 = pd.Timestamp(f'{year}-12-31 23:59:59')

    h    = h_ind.get(ticker, pd.DataFrame())
    d    = d_ind.get(ticker, pd.DataFrame())
    h_yr = h[(h.index >= y0) & (h.index <= y1)] if len(h) else pd.DataFrame()
    d_yr = d[(d.index >= y0) & (d.index <= y1)] if len(d) else pd.DataFrame()

    h_dates = set(h_yr.index.date) if len(h_yr) else set()
    if h_dates:
        d_yr = d_yr[[dt.date() not in h_dates for dt in d_yr.index]]

    sigs = []
    for frame in [h_yr, d_yr]:
        if frame.empty:
            continue
        for ts, row in frame.iterrows():
            sd = pd.Timestamp(ts.date())

            # Regime + VIX
            spy_row = spy_d[spy_d.index <= sd]
            vix_row = vix_close[vix_close.index <= sd]
            if spy_row.empty or vix_row.empty:
                continue
            vix_val = float(vix_row.iloc[-1])
            if pd.isna(vix_val) or not (p['vix_low'] <= vix_val <= p['vix_high']):
                continue
            spy_above = float(spy_row['Close'].iloc[-1]) > float(spy_row['ema50'].iloc[-1])

            c   = float(row['Close'])
            r   = float(row['rsi'])
            e21 = float(row['ema21'])
            v   = float(row['Volume'])
            vm  = float(row['volma'])
            ru3 = bool(row['rsi_up3'])
            rd3 = bool(row['rsi_dn3'])
            mb  = bool(row['mhist_bull'])
            mbe = bool(row['mhist_bear'])

            vol_thresh = vm * p['vol_mult']

            # RSI mirrors: bull [low, high], bear [100-high, 100-low]
            bull_rsi = p['rsi_low']  <= r <= p['rsi_high']
            bear_rsi = (100 - p['rsi_high']) <= r <= (100 - p['rsi_low'])

            bull = sum([c > e21, bull_rsi and ru3, mb, v > vol_thresh])
            bear = sum([c < e21, bear_rsi and rd3, mbe, v > vol_thresh])

            if spy_above and bull >= p['min_score']:
                direction = 'bullish'
            elif not spy_above and bear >= p['min_score']:
                direction = 'bearish'
            else:
                continue

            sigs.append({'ts': ts, 'sd': sd, 'ticker': ticker,
                         'direction': direction, 'close': c})
    return sigs


def run_variant(hourly, daily, vix_df, h_ind, d_ind, spy_d, rv_sers, overrides):
    """Run full backtest with baseline params overridden by 'overrides'. Returns all_trades."""
    p         = {**BASELINE, **overrides}
    vix_close = vix_df['Close']
    all_trades = []

    for year in YEARS:
        raw_sigs = []
        for t in TICKERS:
            raw_sigs += gen_signals_for_year(year, t, h_ind, d_ind, spy_d, vix_close, p)
        raw_sigs.sort(key=lambda x: x['ts'])

        day_streak = {}
        for sig in raw_sigs:
            t   = sig['ticker']
            ts  = sig['ts']
            sd  = sig['sd']
            c   = sig['close']

            iv_rank = get_iv_rank(rv_sers[t], sd)
            if iv_rank is None or iv_rank > IV_RANK_MAX:
                continue

            date_key = ts.date()
            streak   = day_streak.get(date_key, 0)
            if streak >= COOLDOWN_N:
                continue

            sigma = get_sigma(daily[t]['Close'], sd)
            if sigma is None or sigma <= 0:
                continue

            cd      = daily[t]['Close']
            fwd_idx = cd.index.searchsorted(sd, side='right')
            fwd     = list(cd.iloc[fwd_idx:fwd_idx+7])
            if len(fwd) < 2:
                continue

            ex_px, days, ex_type, pnl = simulate_exit(c, fwd, sigma, sig['direction'], p)
            result = 'WIN' if pnl > 0 else 'LOSS'
            day_streak[date_key] = (streak + 1) if result == 'LOSS' else 0

            all_trades.append({'date': str(ts.date()), 'ticker': t,
                               'direction': sig['direction'], 'pnl_pct': round(pnl*100, 2),
                               'result': result, 'year': year, 'exit_type': ex_type})
    return all_trades


# ── REPORTING ─────────────────────────────────────────────────────────────────

def print_variant_table(group_name, results):
    """
    results: list of (label, trades_list)
    Prints: label | total trades | combined win% | combined Sharpe | years≥55%
    """
    print(f"\n  {'─'*62}")
    print(f"  Parameter: {group_name}")
    print(f"  {'─'*62}")
    print(f"  {'Variant':<28}  {'Trades':>7}  {'Win%':>6}  {'Sharpe':>8}  {'≥55% yrs':>9}")
    print(f"  {'─'*62}")

    for label, trades in results:
        wr, sh, beats = combined_stats(trades)
        n = len(trades)
        flag = " ← BASELINE" if "(baseline)" in label else ""
        print(f"  {label:<28}  {n:>7}  {wr:>6.1%}  {sh:>8.2f}  {beats:>6}/6{flag}")

    print(f"  {'─'*62}")


def sensitivity_assessment(group_name, results):
    """
    Returns (is_sensitive, best_label, best_wr, note).
    Sensitive = win rate range > 3pp or any variant improves baseline by > 5pp.
    """
    baseline_wr = None
    best_wr     = -1
    best_label  = ''
    wrs         = []

    for label, trades in results:
        wr, sh, _ = combined_stats(trades)
        wrs.append(wr)
        if "(baseline)" in label:
            baseline_wr = wr
        if wr > best_wr:
            best_wr, best_label = wr, label

    wr_range    = (max(wrs) - min(wrs)) * 100
    improvement = (best_wr - baseline_wr) * 100 if baseline_wr is not None else 0
    is_sensitive = wr_range > 3.0

    return is_sensitive, best_label, best_wr, wr_range, improvement


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*68)
    print("  SENSITIVITY ANALYSIS  |  SPY + QQQ  |  ONE PARAMETER AT A TIME")
    print("="*68)

    hourly, daily, vix_df = download_all()

    print("\n▶  Pre-computing indicators …")
    h_ind, d_ind = build_frames(hourly, daily)
    spy_d = daily['SPY'].copy()
    spy_d['ema50'] = ema(spy_d['Close'], 50)
    rv_sers = {t: rv30_series(daily[t]['Close']) for t in TICKERS}
    print("   Done.")

    # ── Step 1: Sanity check — baseline must reproduce ~318 trades / 49.7% ──
    print("\n▶  Step 1: Sanity check — running baseline …")
    baseline_trades = run_variant(hourly, daily, vix_df, h_ind, d_ind,
                                  spy_d, rv_sers, {})
    bl_wr, bl_sh, bl_beats = combined_stats(baseline_trades)
    n_bl = len(baseline_trades)
    match = "✓ MATCHES" if 310 <= n_bl <= 326 else "⚠ MISMATCH"
    print(f"   Baseline: {n_bl} trades | {bl_wr:.1%} win rate | Sharpe {bl_sh:.2f} | {bl_beats}/6 yrs ≥55%")
    print(f"   Expected: ~318 trades / ~49.7% win rate → {match}")

    # ── Step 2: Run all parameter groups ─────────────────────────────────────
    print("\n▶  Step 2: Running sensitivity variants …\n")

    all_group_results = {}
    assessment_rows   = []

    for group_name, variants in VARIANT_GROUPS.items():
        group_results = []
        for label, overrides in variants:
            trades = run_variant(hourly, daily, vix_df, h_ind, d_ind,
                                 spy_d, rv_sers, overrides)
            group_results.append((label, trades))
        all_group_results[group_name] = group_results
        print_variant_table(group_name, group_results)
        is_sens, best_lbl, best_wr, wr_range, improvement = sensitivity_assessment(
            group_name, group_results)
        assessment_rows.append((group_name, is_sens, best_lbl, best_wr, wr_range, improvement))

    # ── Step 3: Per-year breakdown for each group ─────────────────────────────
    print("\n" + "="*68)
    print("  PER-YEAR WIN RATES BY PARAMETER GROUP")
    print("="*68)

    for group_name, group_results in all_group_results.items():
        print(f"\n  {group_name}")
        yr_hdr = "  " + f"{'Variant':<28}" + "".join(f"  {y}" for y in YEARS)
        print(yr_hdr)
        print("  " + "─"*66)
        for label, trades in group_results:
            yr_wrs = per_year_wr(trades)
            cells  = "".join(f"  {yr_wrs.get(y, 0):5.1%}" for y in YEARS)
            baseline_tag = " ←" if "(baseline)" in label else ""
            print(f"  {label:<28}{cells}{baseline_tag}")

    # ── Step 4: Honest assessment ─────────────────────────────────────────────
    print("\n" + "="*68)
    print("  PARAMETER SENSITIVITY ASSESSMENT")
    print("="*68)
    print(f"\n  {'Parameter':<22}  {'Sensitive?':>11}  {'Win% range':>12}  {'Best improvement':>18}")
    print(f"  {'─'*70}")

    for group_name, is_sens, best_lbl, best_wr, wr_range, improvement in assessment_rows:
        sens_str = "YES — matters" if is_sens else "no — stable"
        imp_str  = f"+{improvement:.1f}pp ({best_lbl[:20]})"
        print(f"  {group_name:<22}  {sens_str:>11}  {wr_range:>10.1f}pp  {imp_str:>18}")

    print()
    print("  ── Detailed findings ────────────────────────────────────────────")

    for group_name, is_sens, best_lbl, best_wr, wr_range, improvement in assessment_rows:
        print(f"\n  [{group_name}]")
        group_results = all_group_results[group_name]
        trades_by_label = {lbl: t for lbl, t in group_results}

        if group_name == 'RSI range':
            wrs = [(lbl, combined_stats(t)[0]) for lbl, t in group_results]
            print(f"  Win rates: {', '.join(f'{l}: {w:.1%}' for l,w in wrs)}")
            if is_sens:
                print(f"  Sensitive — the RSI window meaningfully changes which pullbacks qualify.")
                print(f"  Tighter range (35-45) filters more noise but also reduces trade count.")
            else:
                print(f"  Not sensitive — RSI range shifts produce similar overall win rates.")
                print(f"  The 3-bar direction check matters more than the exact range boundaries.")

        elif group_name == 'VIX range':
            wrs = [(lbl, combined_stats(t)[0]*100, len(t)) for lbl, t in group_results]
            print(f"  Win rates: {', '.join(f'{l}: {w:.1f}% ({n} trades)' for l,w,n in wrs)}")
            if is_sens:
                print(f"  Sensitive — tighter VIX windows admit fewer days, changing sample composition.")
                print(f"  Note: fewer trades means noisier estimates, not necessarily a real edge.")
            else:
                print(f"  Not sensitive — the strategy works similarly across the tested VIX ranges.")

        elif group_name == 'Volume threshold':
            wrs = [(lbl, combined_stats(t)[0]*100, len(t)) for lbl, t in group_results]
            print(f"  Win rates: {', '.join(f'{l}: {w:.1f}% ({n} trades)' for l,w,n in wrs)}")
            if is_sens:
                print(f"  Sensitive — volume confirmation level meaningfully changes results.")
            else:
                print(f"  Not sensitive — any volume above the 20-day avg qualifies equally well.")
                print(f"  Higher volume thresholds reduce trade count without proportional improvement.")

        elif group_name == 'Stop loss':
            wrs = [(lbl, combined_stats(t)[0]*100, combined_stats(t)[1]) for lbl, t in group_results]
            print(f"  Win rates + Sharpe: {', '.join(f'{l}: {w:.1f}% / {s:.2f}' for l,w,s in wrs)}")
            print(f"  Stop placement affects loss severity, not which trades are taken.")
            print(f"  Tighter stops increase loss frequency; wider stops increase loss size.")
            if is_sens:
                print(f"  Sensitive — Sharpe changes materially. Exit discipline matters here.")
            else:
                print(f"  Not sensitive — win rate is similar; Sharpe may differ more than win rate.")

        elif group_name == 'Profit targets':
            wrs = [(lbl, combined_stats(t)[0]*100, combined_stats(t)[1]) for lbl, t in group_results]
            print(f"  Win rates + Sharpe: {', '.join(f'{l}: {w:.1f}% / {s:.2f}' for l,w,s in wrs)}")
            print(f"  Target levels determine how large winners grow. Lower targets lock in gains")
            print(f"  early (higher win rate possible); higher targets cut fewer winners short.")
            if is_sens:
                print(f"  Sensitive — target level materially changes win rate or Sharpe.")
            else:
                print(f"  Not sensitive — the exit rules produce similar outcomes across tested levels.")

    # ── Overfit warning ───────────────────────────────────────────────────────
    overfit_flags = [(g, best_lbl, imp) for g, _, best_lbl, _, _, imp in assessment_rows if imp > 5.0]
    print("\n  ── Overfit risk flags ───────────────────────────────────────────")
    if overfit_flags:
        for g, lbl, imp in overfit_flags:
            print(f"  ⚠ [{g}] '{lbl}' shows +{imp:.1f}pp improvement on this dataset.")
            print(f"    That is a meaningful jump, but it was found by scanning the same")
            print(f"    2020-2025 data used to build the strategy. It should NOT be treated")
            print(f"    as confirmed edge without out-of-sample validation.")
    else:
        print(f"  No single parameter change produced a jump > 5pp in combined win rate.")
        print(f"  This is a healthy sign: results are not driven by one magic number.")

    print("\n  ── Bottom line ──────────────────────────────────────────────────")
    sensitive   = [g for g, is_s, *_ in assessment_rows if is_s]
    insensitive = [g for g, is_s, *_ in assessment_rows if not is_s]
    if sensitive:
        print(f"  Parameters that matter (>3pp range): {', '.join(sensitive)}")
    if insensitive:
        print(f"  Parameters that don't (≤3pp range):  {', '.join(insensitive)}")
    print(f"\n  These results were produced by one-parameter-at-a-time testing.")
    print(f"  No combined 'best' configuration has been assembled — that would")
    print(f"  curve-fit this exact dataset and produce results that won't hold.")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'generated':        datetime.now().isoformat(),
        'baseline_trades':  n_bl,
        'baseline_win_rate': round(bl_wr, 4),
        'baseline_sharpe':   round(bl_sh, 3),
        'groups': {},
    }
    for group_name, group_results in all_group_results.items():
        output['groups'][group_name] = []
        for label, trades in group_results:
            wr, sh, beats = combined_stats(trades)
            output['groups'][group_name].append({
                'label':    label,
                'trades':   len(trades),
                'win_rate': round(wr, 4),
                'sharpe':   round(sh, 3),
                'beats_55': beats,
                'by_year':  {str(y): round(v, 4) for y, v in per_year_wr(trades).items()},
            })

    with open(OUTFILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Saved {OUTFILE}")


if __name__ == '__main__':
    main()
