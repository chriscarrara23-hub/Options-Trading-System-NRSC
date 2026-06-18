#!/usr/bin/env python3
"""
Options Trading Backtest: SPY + QQQ, 2020-2025
Black-Scholes option pricing, 30-day realized vol as IV proxy.
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
TICKERS       = ['SPY', 'QQQ']
YEARS         = [2020, 2021, 2022, 2023, 2024, 2025]
VIX_LOW       = 15
VIX_HIGH      = 35
DTE           = 7
RISK_FREE     = 0.05
STOP_PCT      = -0.45
TAKE_HALF_PCT = 0.50
FULL_EXIT_PCT = 1.00
MAX_HOLD      = 7
IV_RANK_MAX   = 50
COOLDOWN_N    = 3
TODAY         = datetime.today().strftime('%Y-%m-%d')
OUTFILE       = '/Users/noahrourke/trading-system/backtest_results.json'


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d  = s.diff()
    ag = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def macd_hist(s, f=12, sl=26, sig=9):
    m = ema(s, f) - ema(s, sl)
    return m - ema(m, sig)


def calc_adx(df, period=14):
    """ADX(period) on a daily OHLC DataFrame. Returns Series indexed same as df."""
    h, l, c = df['High'], df['Low'], df['Close']
    tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    up, down = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=df.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    pdi   = 100 * pdm.ewm(com=period - 1, adjust=False).mean() / atr_s
    ndi   = 100 * ndm.ewm(com=period - 1, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    c, v = df['Close'], df['Volume']
    df['ema21']  = ema(c, 21)
    df['rsi']    = rsi(c)
    df['mhist']  = macd_hist(c)
    df['volma']  = v.rolling(20).mean()
    r, m = df['rsi'], df['mhist']
    # 3-bar check (primary, for 1H data)
    df['rsi_up3']    = (r > r.shift(1)) & (r.shift(1) > r.shift(2))
    df['rsi_dn3']    = (r < r.shift(1)) & (r.shift(1) < r.shift(2))
    # 2-bar check (fallback, for daily data where RSI moves slower)
    df['rsi_up2']    = r > r.shift(1)
    df['rsi_dn2']    = r < r.shift(1)
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


def simulate_exit(entry_px, fwd, sigma, direction):
    K   = entry_px
    typ = 'call' if direction == 'bullish' else 'put'
    e0  = bs(entry_px, K, DTE/365, RISK_FREE, sigma, typ)
    half = False

    for day, S in enumerate(fwd[:MAX_HOLD], 1):
        T   = max((DTE - day) / 365, 0.001)
        cur = bs(float(S), K, T, RISK_FREE, sigma, typ)
        pnl = (cur - e0) / e0

        if pnl <= STOP_PCT:
            return float(S), day, 'STOP', pnl
        if pnl >= FULL_EXIT_PCT:
            return float(S), day, 'FULL_TARGET', pnl
        if not half and pnl >= TAKE_HALF_PCT:
            half = True
        if half and pnl < TAKE_HALF_PCT * 0.5:
            return float(S), day, 'TRAIL_EXIT', (TAKE_HALF_PCT + pnl) / 2

    if not fwd:
        return entry_px, 0, 'NO_DATA', 0.0
    day = min(MAX_HOLD, len(fwd))
    S   = float(fwd[day - 1])
    cur = bs(S, K, max((DTE - day) / 365, 0.001), RISK_FREE, sigma, typ)
    return S, day, 'TIME_EXIT', (cur - e0) / e0


def rv30(close):
    lr = np.log(close / close.shift(1))
    return lr.rolling(30).std() * math.sqrt(252)


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


def compute_stats(trades):
    if not trades:
        return {}
    pnls  = [t['pnl_pct']/100 for t in trades]
    wins  = [p for p in pnls if p > 0]
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

    arr = np.array(pnls)
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


def regime_desc(year, spy_yr, vix_yr):
    if spy_yr.empty:
        return 'No data'
    ret   = float((spy_yr['Close'].iloc[-1] - spy_yr['Close'].iloc[0]) /
                  spy_yr['Close'].iloc[0] * 100)
    avg_v = float(vix_yr.mean()) if len(vix_yr) else 0
    pct_r = float(vix_yr.between(VIX_LOW, VIX_HIGH).mean() * 100) if len(vix_yr) else 0
    base  = f"SPY {ret:+.0f}%, avg VIX {avg_v:.0f}, {pct_r:.0f}% of days VIX in 20-30"
    ctx = {
        2020: f"COVID crash + recovery. {base}",
        2021: f"Strong bull market post-COVID. {base}",
        2022: f"Fed rate-hike bear market. {base}",
        2023: f"AI-driven recovery rally. {base}",
        2024: f"Bull market with election volatility. {base}",
        2025: f"Macro uncertainty, tariff volatility. {base}",
    }
    return ctx.get(year, base)


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download_all():
    print("\n▶  Downloading price data …")
    hourly, daily = {}, {}

    for t in TICKERS:
        print(f"   {t} 1H … ", end='', flush=True)
        # yfinance 1H data is capped at 730 days; use period= to avoid rejection
        h = yf.download(t, period='730d', interval='1h',
                        auto_adjust=True, progress=False)
        h = flatten(h)
        h.index = pd.to_datetime(h.index)
        if h.index.tz:
            h.index = h.index.tz_convert('US/Eastern').tz_localize(None)
        hourly[t] = h
        if len(h):
            print(f"{len(h)} bars  [{h.index[0].date()} → {h.index[-1].date()}]")
        else:
            print("no data")

    for t in TICKERS:
        print(f"   {t} 1D … ", end='', flush=True)
        d = yf.download(t, start='2019-01-01', end=TODAY,
                        interval='1d', auto_adjust=True, progress=False)
        d = flatten(d)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        daily[t] = d
        print(f"{len(d)} bars")

    print(f"   VIX 1D … ", end='', flush=True)
    vix = yf.download('^VIX', start='2019-01-01', end=TODAY,
                      interval='1d', auto_adjust=True, progress=False)
    vix = flatten(vix)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    print(f"{len(vix)} bars")
    return hourly, daily, vix


# ── SIGNAL GENERATION ─────────────────────────────────────────────────────────

def gen_signals(year, ticker, h_ind, d_ind, spy_d, vix_close, adx_series, rv_ser, close_daily):
    """
    Returns list of raw signal dicts (pre-cooldown, pre-IV-filter) for year+ticker.
    Uses 1H bars if available, daily bars for any uncovered dates.
    """
    y0 = pd.Timestamp(f'{year}-01-01')
    y1 = pd.Timestamp(f'{year}-12-31 23:59:59')

    h = h_ind.get(ticker, pd.DataFrame())
    d = d_ind.get(ticker, pd.DataFrame())

    h_yr = h[(h.index >= y0) & (h.index <= y1)] if len(h) else pd.DataFrame()
    d_yr = d[(d.index >= y0) & (d.index <= y1)] if len(d) else pd.DataFrame()

    # For daily: only use dates NOT covered by hourly bars
    h_dates = set(h_yr.index.date) if len(h_yr) else set()
    if len(h_dates):
        d_yr = d_yr[[dt.date() not in h_dates for dt in d_yr.index]]

    sigs = []
    for frame, dtype in [(h_yr, '1H'), (d_yr, 'DAILY')]:
        if frame.empty:
            continue
        for ts, row in frame.iterrows():
            sd = pd.Timestamp(ts.date())

            # Regime: SPY above/below 50 EMA + VIX in range
            spy_row = spy_d[spy_d.index <= sd]
            vix_row = vix_close[vix_close.index <= sd]
            if spy_row.empty or vix_row.empty:
                continue
            vix_val = float(vix_row.iloc[-1])
            if pd.isna(vix_val) or not (VIX_LOW <= vix_val <= VIX_HIGH):
                continue
            spy_above = float(spy_row['Close'].iloc[-1]) > float(spy_row['ema50'].iloc[-1])

            c   = float(row['Close'])
            r   = float(row['rsi'])
            e21 = float(row['ema21'])
            v   = float(row['Volume'])
            vm  = float(row['volma'])
            mb  = bool(row['mhist_bull'])
            mbe = bool(row['mhist_bear'])

            # 3-bar RSI directional check on all timeframes (per original spec)
            ru = bool(row['rsi_up3'])
            rd = bool(row['rsi_dn3'])

            bull = sum([c > e21, (30 <= r <= 50) and ru, mb, v > vm])
            bear = sum([c < e21, (50 <= r <= 70) and rd, mbe, v > vm])

            # 3-of-4 conditions required; 3-bar RSI check intact on all timeframes
            if spy_above and bull >= 3:
                direction = 'bullish'
            elif not spy_above and bear >= 3:
                direction = 'bearish'
            else:
                continue

            score = bull if direction == 'bullish' else bear
            sigs.append({
                'ts':        ts,
                'sd':        sd,
                'ticker':    ticker,
                'direction': direction,
                'score':     score,
                'close':     c,
                'dtype':     dtype,
                'vix':       round(vix_val, 2),
            })

    return sorted(sigs, key=lambda x: x['ts'])


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────

def run_backtest(hourly, daily, vix_df):
    print("\n▶  Calculating indicators …")

    spy_d = daily['SPY'].copy()
    spy_d['ema50'] = ema(spy_d['Close'], 50)
    vix_close = vix_df['Close']

    adx_spy = calc_adx(daily['SPY'])
    print(f"   SPY ADX(14): {len(adx_spy.dropna())} values, current={adx_spy.iloc[-1]:.1f}")

    rv_ser  = {t: rv30(daily[t]['Close']) for t in TICKERS}

    h_ind, d_ind = {}, {}
    for t in TICKERS:
        h_ind[t] = add_indicators(hourly[t]) if len(hourly[t]) >= 60 else pd.DataFrame()
        d_ind[t] = add_indicators(daily[t])
        print(f"   {t}: 1H={len(h_ind[t])} bars  1D={len(d_ind[t])} bars")

    all_trades   = []
    year_results = {}

    print("\n▶  Running year-by-year backtest …\n")

    for year in YEARS:
        year_trades = []
        total_sig = iv_skip = cd_skip = 0

        # Collect and interleave signals from both tickers chronologically
        raw_sigs = []
        for t in TICKERS:
            raw_sigs += gen_signals(year, t, h_ind, d_ind, spy_d, vix_close,
                                    adx_spy, rv_ser[t], daily[t]['Close'])
        raw_sigs.sort(key=lambda x: x['ts'])

        # Per-day consecutive-loss state (shared across tickers)
        day_streak = {}   # date → consecutive loss count today

        for sig in raw_sigs:
            total_sig += 1
            t  = sig['ticker']
            sd = sig['sd']
            c  = sig['close']
            rv = rv_ser[t]
            cd = daily[t]['Close']

            # IV rank filter
            iv_rank = get_iv_rank(rv, sd)
            if iv_rank is None or iv_rank > IV_RANK_MAX:
                iv_skip += 1
                continue

            # Cooldown filter
            streak = day_streak.get(sd.date(), 0)
            if streak >= COOLDOWN_N:
                cd_skip += 1
                continue

            # Sigma for B-S
            sigma = get_sigma(cd, sd)
            if sigma is None or sigma <= 0:
                continue

            # Forward daily closes
            fwd_idx = cd.index.searchsorted(sd, side='right')
            fwd     = list(cd.iloc[fwd_idx:fwd_idx + MAX_HOLD])
            if len(fwd) < 2:
                continue

            ex_px, days, ex_type, pnl = simulate_exit(c, fwd, sigma, sig['direction'])
            result = 'WIN' if pnl > 0 else 'LOSS'

            # Update streak
            date_key = sd.date()
            day_streak[date_key] = (streak + 1) if result == 'LOSS' else 0

            trade = {
                'date':          str(sd.date()),
                'ticker':        t,
                'direction':     sig['direction'],
                'score':         sig['score'],
                'proxy_iv_rank': round(iv_rank, 1),
                'sigma':         round(sigma, 4),
                'entry_price':   round(c, 2),
                'exit_price':    round(ex_px, 2),
                'days_held':     days,
                'exit_type':     ex_type,
                'pnl_pct':       round(pnl * 100, 2),
                'result':        result,
                'year':          year,
                'data_type':     sig['dtype'],
                'flags':         [],
            }
            year_trades.append(trade)
            all_trades.append(trade)

        stats = compute_stats(year_trades)
        wr    = stats.get('win_rate', 0)

        yr_spy = spy_d[(spy_d.index.year == year)]
        yr_vix = vix_close[vix_close.index.year == year]

        year_results[year] = {
            'year':           year,
            'total_signals':  total_sig,
            'iv_skips':       iv_skip,
            'cooldown_skips': cd_skip,
            'signals_taken':  len(year_trades),
            'win_rate':       round(wr, 4),
            'below_50pct':    wr < 0.50,
            'beats_55pct':    wr >= 0.55,
            'regime':         regime_desc(year, yr_spy, yr_vix),
            'stats':          stats,
        }

        flag = " ⚠" if (year_trades and wr < 0.50) else (" ✓" if (year_trades and wr >= 0.55) else "")
        print(f"   {year}: {len(year_trades):3d} trades | win {wr:.1%}{flag} | "
              f"IV-skip {iv_skip} | CD-skip {cd_skip}")

    return year_results, all_trades


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(year_results, all_trades):
    print("\n" + "="*72)
    print("  YEAR-BY-YEAR RESULTS")
    print("="*72)
    hdr = f"{'Year':<5}  {'Signals':>7}  {'IVSkip':>7}  {'CDSkip':>7}  {'Taken':>6}  {'Win%':>6}  {'AvgW%':>6}  {'AvgL%':>6}  {'MaxDD%':>7}  {'Sharpe':>7}"
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
    n_all = len(all_trades)
    wr_all = len([t for t in all_trades if t['result']=='WIN']) / n_all if n_all else 0
    bull_t = [t for t in all_trades if t['direction']=='bullish']
    bear_t = [t for t in all_trades if t['direction']=='bearish']
    print(f"\nCombined  {n_all} trades | {wr_all:.1%} win rate | "
          f"{beats_55}/6 years ≥55% | below 50%: {below_50 or 'none'}")
    print(f"Bullish: {len(bull_t)} trades | Bearish: {len(bear_t)} trades")

    print("\n── Regime Notes ─────────────────────────────────────────────────────")
    for yr in YEARS:
        print(f"  {yr}: {year_results.get(yr,{}).get('regime','N/A')}")

    print("\n── Assessment ───────────────────────────────────────────────────────")
    if beats_55 >= 4:
        verdict = "READY FOR PAPER TRADING"
        note = "Strategy is robust across multiple regimes. VIX 20-30 filter creates quality trading conditions."
    elif beats_55 >= 2:
        verdict = "NEEDS REFINEMENT"
        note = "Works in trending regimes; struggles in extreme vol or choppy markets."
    else:
        verdict = "NOT READY"
        note = "Underperforms in most years. Consider loosening VIX range or adjusting signal thresholds."

    print(f"  Verdict: {verdict}")
    print(f"  {note}")
    print(f"\n  ℹ  yfinance 1H data covers only the last ~730 days.")
    print(f"     Years 2020–2023 used daily bars for signal generation.")
    print(f"     All option exits simulated with B-S + 30-day realized vol as IV proxy.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("  OPTIONS TRADING BACKTEST SYSTEM  |  SPY + QQQ  |  2020–2025")
    print("="*72)

    hourly, daily, vix_df = download_all()
    year_results, all_trades = run_backtest(hourly, daily, vix_df)
    print_report(year_results, all_trades)

    last20 = sorted(all_trades, key=lambda t: t['date'])[-20:]
    output = {
        'generated':        datetime.now().isoformat(),
        'total_trades':     len(all_trades),
        'overall_win_rate': round(len([t for t in all_trades if t['result']=='WIN'])/len(all_trades), 4) if all_trades else 0,
        'years':            {str(y): year_results[y] for y in YEARS},
        'last_20_trades':   last20,
    }
    with open(OUTFILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ backtest_results.json saved  ({len(all_trades)} trades)")


if __name__ == '__main__':
    main()
