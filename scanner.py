#!/usr/bin/env python3
"""
Live options signal scanner: SPY + QQQ, runs every 60 min during market hours.
Checks signal ruleset, news risk via Claude, calendar flags, sends Discord alerts.
"""

import json
import math
import os
import time
import warnings
from datetime import datetime, date, timedelta

import anthropic
import numpy as np
import pandas as pd
import pytz
import requests
import schedule
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm

warnings.filterwarnings('ignore')
load_dotenv()

from paper_execution import (execute_paper_trade, daily_summary as _paper_daily_summary,
                             monitor_positions, reconcile_positions_on_startup)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS           = ['SPY', 'QQQ']
VIX_LOW           = 15
VIX_HIGH          = 35
DTE               = 7
RISK_FREE         = 0.05
IV_RANK_MAX       = 50
COOLDOWN_N        = 3
DISCORD_WEBHOOK   = os.getenv('DISCORD_WEBHOOK_URL')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SCANNER_LOG       = '/Users/noahrourke/trading-system/scanner_log.json'
TRADE_LOG         = '/Users/noahrourke/trading-system/trade_log.json'
ET                = pytz.timezone('US/Eastern')

# Hardcoded FOMC decision dates 2025-2026
FOMC_DATES = {
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_val(s, n=14):
    d  = s.diff()
    ag = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    r  = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    return float(r.iloc[-1]) if not r.empty else 50.0


def macd_hist_val(s, f=12, sl=26, sig=9):
    m = ema(s, f) - ema(s, sl)
    h = m - ema(m, sig)
    return float(h.iloc[-1]), float(h.iloc[-2]) if len(h) >= 2 else 0.0


def bs(S, K, T, r, sig, typ='call'):
    if T < 1e-6 or sig < 1e-6:
        iv = max(S-K, 0) if typ == 'call' else max(K-S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S/K) + (r + .5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    if typ == 'call':
        return max(S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2), 0.01)
    return max(K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), 0.01)


def atm_strike(price):
    """Round to nearest dollar for ATM strike."""
    return round(price)


def expiry_date(dte=7):
    """Next available expiry ~DTE days out (Friday if possible)."""
    d = date.today() + timedelta(days=dte)
    # Roll forward to Friday if needed
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


# ── MARKET HOURS ──────────────────────────────────────────────────────────────

def is_market_hours():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


# ── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_1h_data(ticker, bars=60):
    """Fetch recent 1H bars for indicator calculation."""
    df = yf.download(ticker, period='10d', interval='1h',
                     auto_adjust=True, progress=False)
    df = flatten(df)
    df.index = pd.to_datetime(df.index)
    if df.index.tz:
        df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
    return df.tail(bars)


def fetch_daily_data(ticker, bars=300):
    df = yf.download(ticker, period='400d', interval='1d',
                     auto_adjust=True, progress=False)
    df = flatten(df)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.tail(bars)


def fetch_vix():
    df = yf.download('^VIX', period='5d', interval='1d',
                     auto_adjust=True, progress=False)
    df = flatten(df)
    return float(df['Close'].iloc[-1]) if len(df) else None


# ── IV RANK PROXY ─────────────────────────────────────────────────────────────

def proxy_iv_rank(daily_close):
    """30-day RV percentile rank over trailing 252 days."""
    if len(daily_close) < 282:
        return None
    lr   = np.log(daily_close / daily_close.shift(1))
    rv30 = lr.rolling(30).std() * math.sqrt(252)
    curr = float(rv30.iloc[-1])
    hist = rv30.iloc[-252:-1].dropna()
    if pd.isna(curr) or len(hist) < 20:
        return None
    return float((hist < curr).sum() / len(hist) * 100)


def sigma_30d(daily_close):
    lr = np.log(daily_close / daily_close.shift(1)).dropna()
    return float(lr.tail(30).std() * math.sqrt(252))


# ── SIGNAL EVALUATION ─────────────────────────────────────────────────────────

def evaluate_signal(ticker):
    """
    Evaluate full signal ruleset for a ticker.
    Returns dict with score, direction, indicator values, or None if no signal.
    """
    h1  = fetch_1h_data(ticker)
    d1  = fetch_daily_data(ticker)
    spy = fetch_daily_data('SPY')

    if len(h1) < 30 or len(d1) < 60 or len(spy) < 60:
        return None

    vix = fetch_vix()
    if vix is None or not (VIX_LOW <= vix <= VIX_HIGH):
        return None

    # Regime
    spy_close = spy['Close']
    spy_ema50 = float(ema(spy_close, 50).iloc[-1])
    spy_above = float(spy_close.iloc[-1]) > spy_ema50

    # 1H indicators
    c1h = h1['Close']
    v1h = h1['Volume']

    close    = float(c1h.iloc[-1])
    ema21    = float(ema(c1h, 21).iloc[-1])
    rsi_now  = rsi_val(c1h)
    rsi_prev = rsi_val(c1h.iloc[:-1])
    rsi_pp   = rsi_val(c1h.iloc[:-2])
    mhist_now, mhist_prev = macd_hist_val(c1h)
    vol_now  = float(v1h.iloc[-1])
    vol_ma   = float(v1h.rolling(20).mean().iloc[-1])

    rsi_up3 = rsi_now > rsi_prev > rsi_pp
    rsi_dn3 = rsi_now < rsi_prev < rsi_pp
    mhist_bull = (mhist_now > 0) and (mhist_now > mhist_prev)
    mhist_bear = (mhist_now < 0) and (mhist_now < mhist_prev)
    vol_ok = vol_now > vol_ma

    bull_conds = [close > ema21, (30 <= rsi_now <= 50) and rsi_up3, mhist_bull, vol_ok]
    bear_conds = [close < ema21, (50 <= rsi_now <= 70) and rsi_dn3, mhist_bear, vol_ok]

    bull_score = sum(bull_conds)
    bear_score = sum(bear_conds)

    if spy_above and bull_score >= 3:
        direction = 'bullish'
        score = bull_score
    elif not spy_above and bear_score >= 3:
        direction = 'bearish'
        score = bear_score
    else:
        return None

    # Proxy IV rank
    iv_rank = proxy_iv_rank(d1['Close'])
    if iv_rank is None or iv_rank > IV_RANK_MAX:
        return None

    return {
        'ticker':    ticker,
        'direction': direction,
        'score':     score,
        'close':     round(close, 2),
        'rsi':       round(rsi_now, 1),
        'ema21':     round(ema21, 2),
        'mhist':     round(mhist_now, 4),
        'vol_ratio': round(vol_now / vol_ma, 2),
        'vix':       round(vix, 1),
        'iv_rank':   round(iv_rank, 1),
        'sigma':     round(sigma_30d(d1['Close']), 4),
        'strike':    atm_strike(close),
        'expiry':    expiry_date(DTE),
        'option':    'CALL' if direction == 'bullish' else 'PUT',
    }


# ── NEWS RISK (Claude) ────────────────────────────────────────────────────────

def check_news_risk(ticker):
    """Fetch headlines and rate risk via Claude API. Returns (risk, reason)."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        headlines = [item.get('content', {}).get('title', '') or item.get('title', '')
                     for item in news[:10]]
        headlines = [h for h in headlines if h]
        if not headlines:
            return 'LOW', 'No recent news found'

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=200,
            system=(
                f'You are a pre-trade risk filter. Given these recent news headlines for {ticker}, '
                'identify any of: earnings announcements or guidance, Federal Reserve decisions, '
                'analyst upgrades/downgrades, macro data releases, geopolitical events. '
                'Rate risk as LOW, MEDIUM, or HIGH. '
                'Respond only in JSON with no preamble: {"risk": string, "reason": string}'
            ),
            messages=[{'role': 'user', 'content': '\n'.join(headlines)}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        data = json.loads(raw)
        return data.get('risk', 'LOW'), data.get('reason', '')
    except Exception as e:
        return 'LOW', f'News check error: {e}'


# ── CALENDAR CHECKS ───────────────────────────────────────────────────────────

def check_earnings(ticker):
    """Flag if earnings fall within next 7 days."""
    try:
        t = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is None or cal.empty:
            return False
        cal.index = pd.to_datetime(cal.index).tz_localize(None)
        window_end = pd.Timestamp(date.today() + timedelta(days=7))
        window_start = pd.Timestamp(date.today())
        upcoming = cal[(cal.index >= window_start) & (cal.index <= window_end)]
        return not upcoming.empty
    except Exception:
        return False


def check_fed(days_ahead=7):
    """Flag if an FOMC decision date falls within next N days."""
    today = date.today()
    for d in range(days_ahead + 1):
        if today + timedelta(days=d) in FOMC_DATES:
            return True
    return False


# ── DISCORD ───────────────────────────────────────────────────────────────────

def send_discord(message):
    if not DISCORD_WEBHOOK:
        print(f'[Discord] {message}')
        return
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK, json={'content': message}, timeout=10)
            if r.status_code == 429:
                # Discord rate limit — honour retry_after before next attempt
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Discord 429 rate-limit, retrying in {wait:.1f}s (attempt {attempt + 1}/3) …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return   # delivered
        except Exception as e:
            print(f'Discord error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                time.sleep(1)
    print(f'Discord: gave up after 3 attempts — message: {message[:80]}…')


# ── COOLDOWN CHECK ────────────────────────────────────────────────────────────

def today_consecutive_losses():
    """Read trade_log.json and return today's consecutive-loss count."""
    try:
        with open(TRADE_LOG) as f:
            trades = json.load(f)
        today_str = str(date.today())
        today_t   = [t for t in trades if t.get('date') == today_str]
        count = 0
        for t in reversed(today_t):
            if t.get('result') == 'LOSS':
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


# ── LOGGING ───────────────────────────────────────────────────────────────────

def log_scan(entry):
    try:
        with open(SCANNER_LOG) as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append(entry)
    with open(SCANNER_LOG, 'w') as f:
        json.dump(log, f, indent=2, default=str)


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────

def run_scan():
    if not is_market_hours():
        print(f'[{datetime.now(ET).strftime("%H:%M ET")}] Outside market hours, skipping.')
        return

    now_str = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    print(f'\n[{now_str}] Running scan …')

    # Exit monitor runs every cycle regardless of cooldown status.
    # Cooldown only blocks new entries — it never blocks managing existing positions.
    monitor_positions(send_discord)

    # Cooldown check first (shared across all tickers)
    consec_losses = today_consecutive_losses()
    if consec_losses >= COOLDOWN_N:
        msg = '🛑 Trading paused — 3 consecutive losses today. Review log before tomorrow.'
        send_discord(msg)
        print(f'  COOLDOWN: {msg}')
        log_scan({'timestamp': now_str, 'event': 'COOLDOWN', 'losses': consec_losses})
        return

    # Calendar flags (shared)
    fed_flag = check_fed()

    for ticker in TICKERS:
        print(f'  Checking {ticker} …', end=' ', flush=True)

        sig = evaluate_signal(ticker)

        log_entry = {
            'timestamp': now_str,
            'ticker':    ticker,
            'signal':    sig is not None,
            'score':     sig['score'] if sig else None,
            'iv_rank':   sig['iv_rank'] if sig else None,
            'news_risk': None,
            'alert_sent': False,
        }

        if sig is None:
            print('no signal')
            log_scan(log_entry)
            continue

        print(f"SIGNAL {sig['option']} score={sig['score']}/4 iv_rank={sig['iv_rank']}%")

        # News risk
        news_risk, news_reason = check_news_risk(ticker)
        log_entry['news_risk'] = news_risk

        # Earnings flag
        earnings_flag = check_earnings(ticker)

        # Build alert
        prefix = '⚠️ ' if news_risk in ('MEDIUM', 'HIGH') else ''
        fed_note = ' | ⚠️ FOMC in 7d' if fed_flag else ''
        earn_note = ' | ⚠️ Earnings in 7d' if earnings_flag else ''

        # Paper trade — only when news risk is not HIGH
        if news_risk == 'HIGH':
            exec_note = '⏭️ Paper trade skipped: news risk HIGH'
        else:
            exec_note = execute_paper_trade(sig, discord_fn=send_discord)

        message = (
            f"{prefix}{ticker} {sig['option']} SIGNAL "
            f"· Score {sig['score']}/4 "
            f"· Proxy IV Rank {sig['iv_rank']}% "
            f"· News: {news_risk} — {news_reason}"
            f"{fed_note}{earn_note} "
            f"· Strike: ATM {sig['strike']} Exp {sig['expiry']} "
            f"· Stop: -45% · Targets: +50% then +100% "
            f"| {exec_note}"
        )

        send_discord(message)
        log_entry['alert_sent'] = True
        log_entry['exec_note']  = exec_note
        log_entry['message']    = message
        print(f'  → Alert sent: {message[:80]}…')

        log_scan(log_entry)


# ── DAILY SUMMARY (4:05 PM ET, once per day) ─────────────────────────────────

_summary_sent_date = None

def check_daily_summary():
    """Called every 60 min; fires the EOD summary once after 16:05 ET on weekdays."""
    global _summary_sent_date
    now   = datetime.now(ET)
    today = now.date()
    if (now.weekday() < 5
            and now.hour == 16 and now.minute >= 5
            and _summary_sent_date != today):
        _summary_sent_date = today
        _paper_daily_summary(send_discord)


# ── TEST MESSAGE ──────────────────────────────────────────────────────────────

def send_test_message():
    send_discord('✅ Trading system is live and scanning')
    print('Test message sent to Discord.')


# ── SCHEDULER ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  OPTIONS SCANNER  |  SPY + QQQ')
    print('  Runs every 60 min, 9:30–16:00 ET, Mon–Fri')
    print('=' * 60)

    # Reconcile open Alpaca positions against state file before anything else.
    # Guards against ephemeral filesystem loss on Railway redeploy.
    reconcile_positions_on_startup(send_discord)

    schedule.every(60).minutes.do(run_scan)
    schedule.every(60).minutes.do(check_daily_summary)

    # Run immediately on start
    run_scan()
    check_daily_summary()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        send_test_message()
    else:
        main()
