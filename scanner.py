#!/usr/bin/env python3
"""
VS Intraday Scanner — Volume Spike, Variation B (validated 2023-2025).
Tickers: SPY, QQQ, AAPL, NVDA, TSLA
Scan: every 15 min, 9:30am–4:00pm ET, Mon–Fri

Signal flow (two-scan):
  Scan N   : spike detected on bars[-1]  → Discord "👀 waiting" + store pending
  Scan N+1 : bars[-2] = spike bar, bars[-1] = confirmation bar → enter at market
"""

import json
import os
import time
from datetime import date, datetime

import pytz
import requests
import schedule
from dotenv import load_dotenv

load_dotenv()

from paper_execution import (
    execute_vs_trade,
    monitor_positions,
    daily_summary        as _paper_daily_summary,
    reconcile_positions_on_startup,
    today_consecutive_losses,
    load_state,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS         = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK_URL')
ALPACA_KEY      = os.getenv('ALPACA_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET')
DATA_URL        = 'https://data.alpaca.markets'
ET              = pytz.timezone('US/Eastern')
_DIR            = os.path.dirname(os.path.abspath(__file__))
SCANNER_LOG     = os.path.join(_DIR, 'scanner_log.json')
PENDING_FILE    = os.path.join(_DIR, 'pending_signals.json')

# Variation B parameters
VOL_MA_LEN    = 20
VOL_MULT      = 3.0    # bar volume > 3.0× 20-bar rolling average
MOVE_PCT      = 0.005  # |close - open| / open > 0.5%
TGT_MULT      = 1.5    # target = entry ± 1.5× spike_move
MIN_STOP_DIST = 0.01   # skip if stop distance < $0.01
MAX_POSITIONS = 3
COOLDOWN_N    = 3


# ── DISCORD ───────────────────────────────────────────────────────────────────

def send_discord(message):
    if not DISCORD_WEBHOOK:
        print(f'[Discord] {message}')
        return
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK, json={'content': message}, timeout=10)
            if r.status_code == 429:
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Discord 429, retrying in {wait:.1f}s …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return
        except Exception as e:
            print(f'Discord error (attempt {attempt+1}/3): {e}')
            if attempt < 2:
                time.sleep(1)


# ── MARKET HOURS ──────────────────────────────────────────────────────────────

def is_market_hours():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def _is_eod():
    now = datetime.now(ET)
    return now.hour == 15 and now.minute >= 50


def _can_enter():
    """No new entries at or after 3:30pm ET."""
    now = datetime.now(ET)
    return not (now.hour > 15 or (now.hour == 15 and now.minute >= 30))


# ── DATA FETCH ────────────────────────────────────────────────────────────────

def _data_headers():
    return {'APCA-API-KEY-ID': ALPACA_KEY, 'APCA-API-SECRET-KEY': ALPACA_SECRET}


def fetch_bars(ticker, limit=25):
    """Return last `limit` completed 15-min bars from Alpaca IEX feed."""
    try:
        r = requests.get(
            f'{DATA_URL}/v2/stocks/{ticker}/bars',
            headers=_data_headers(),
            params={'timeframe': '15Min', 'feed': 'iex', 'limit': limit},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get('bars', [])
    except Exception as e:
        print(f'  [fetch_bars {ticker}] {e}')
        return []


# ── PENDING SIGNAL STATE ──────────────────────────────────────────────────────

def _load_pending():
    try:
        with open(PENDING_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_pending(pending):
    with open(PENDING_FILE, 'w') as f:
        json.dump(pending, f, indent=2, default=str)


# ── VS SIGNAL DETECTION ───────────────────────────────────────────────────────

def detect_spike(bars):
    """
    Check if bars[-1] (most recent completed bar) is a volume spike.
    Returns spike dict or None.

    Only processes bars from today to avoid false signals from yesterday's data
    when the scan fires before the first 15-min bar of the day completes.
    """
    if len(bars) < VOL_MA_LEN + 1:
        return None

    spike_bar = bars[-1]

    # Skip stale bars from a previous session (compare in ET)
    try:
        ts_str   = spike_bar['t'].replace('Z', '+00:00')
        dt_et    = datetime.fromisoformat(ts_str).astimezone(ET)
        bar_date = dt_et.date()
        if bar_date != datetime.now(ET).date():
            return None
    except Exception:
        pass   # can't parse — allow through, worst case a stale spike gets discarded on conf check

    # 20-bar volume MA (excludes the spike bar itself)
    vols = [b['v'] for b in bars[-(VOL_MA_LEN + 1):-1]]
    vol_ma = sum(vols) / len(vols) if vols else 0
    if vol_ma == 0:
        return None

    vol_ratio = spike_bar['v'] / vol_ma
    if vol_ratio < VOL_MULT:
        return None

    o, c = spike_bar['o'], spike_bar['c']
    if o == 0:
        return None
    move = abs(c - o) / o
    if move < MOVE_PCT:
        return None

    direction  = 'long' if c > o else 'short'
    spike_move = abs(c - o)

    return {
        'ts':         spike_bar['t'],
        'open':       o,
        'close':      c,
        'high':       spike_bar['h'],
        'low':        spike_bar['l'],
        'vol_ratio':  round(vol_ratio, 2),
        'spike_move': round(spike_move, 4),
        'spike_pct':  round(move * 100, 2),
        'direction':  direction,
    }


def build_entry_sig(ticker, pending_spike, conf_bar):
    """
    Build sig dict for execute_vs_trade() from the pending spike + confirmation bar.
    conf_bar is bars[-1] at the time of the confirmation scan.
    Returns sig dict or None if stop is degenerate.
    """
    entry_px  = conf_bar['c']   # enter at confirmation bar close
    direction = pending_spike['direction']

    stop_px   = pending_spike['low'] if direction == 'long' else pending_spike['high']
    stop_dist = abs(entry_px - stop_px)
    if stop_dist < MIN_STOP_DIST:
        return None

    spike_move = pending_spike['spike_move']
    target_px  = (entry_px + TGT_MULT * spike_move if direction == 'long'
                  else entry_px - TGT_MULT * spike_move)

    return {
        'ticker':         ticker,
        'direction':      direction,
        'entry_price':    round(entry_px, 4),
        'stop_price':     round(stop_px, 4),
        'target_price':   round(target_px, 4),
        'spike_bar_high': pending_spike['high'],
        'spike_bar_low':  pending_spike['low'],
        'spike_move':     spike_move,
        'vol_ratio':      pending_spike['vol_ratio'],
    }


# ── FILE INIT ─────────────────────────────────────────────────────────────────

def _init_data_files():
    defaults = {
        SCANNER_LOG:                                   [],
        PENDING_FILE:                                  {},
        os.path.join(_DIR, 'trade_log.json'):          [],
        os.path.join(_DIR, 'paper_trades.json'):       [],
        os.path.join(_DIR, 'position_state.json'):     {},
    }
    for path, empty in defaults.items():
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(empty, f)
            print(f'[init] Created {os.path.basename(path)}')


def _log_scan(entry):
    try:
        try:
            with open(SCANNER_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        if len(log) > 500:
            log = log[-500:]
        with open(SCANNER_LOG, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log_scan] write failed: {e}')


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────

def run_scan():
    market_hrs = is_market_hours()
    now_str    = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')

    if not market_hrs:
        print(f'[{now_str}] Outside market hours, skipping.')
        return

    print(f'\n[{now_str}] VS scan …')

    # ── Step 1: Monitor exits (runs every scan, regardless of cooldown) ───────
    try:
        monitor_positions(send_discord, scan_log_fn=_log_scan)
    except Exception as e:
        print(f'  [monitor_positions] ERROR — {type(e).__name__}: {e}')
        _log_scan({'timestamp': now_str, 'event': 'MONITOR_ERROR',
                   'error': f'{type(e).__name__}: {e}'})

    # ── Step 2: EOD — clear pending signals, no new entries ──────────────────
    if _is_eod():
        _save_pending({})
        print('  EOD — pending signals cleared.')
        _log_scan({'timestamp': now_str, 'event': 'EOD_CLEAR'})
        return

    # ── Step 3: Cooldown check ───────────────────────────────────────────────
    consec = today_consecutive_losses()
    if consec >= COOLDOWN_N:
        msg = '🛑 Trading paused — 3 consecutive losses today. Review before tomorrow.'
        send_discord(msg)
        print(f'  COOLDOWN: {consec} consecutive losses')
        _log_scan({'timestamp': now_str, 'event': 'COOLDOWN', 'losses': consec})
        return

    state                = load_state()
    pending              = _load_pending()
    new_pending          = {}   # rebuilt each scan; only carries forward freshly detected spikes
    can_enter            = _can_enter()
    open_positions_at_scan = len(state)
    ticker_results       = []

    for ticker in TICKERS:
        try:
            bars = fetch_bars(ticker, limit=25)
            if len(bars) < VOL_MA_LEN + 2:
                print(f'  {ticker}: only {len(bars)} bars — skip')
                ticker_results.append({'ticker': ticker, 'result': 'skipped',
                                       'reason': f'insufficient_bars ({len(bars)})'})
                continue

            # ── 5a. Process pending spike → confirmation → entry ──────────────
            if ticker in pending:
                spike = pending[ticker]
                _log_scan({'timestamp': now_str, 'event': 'PENDING_CONFIRMATION',
                           'ticker': ticker, 'waiting_since': spike.get('ts', '?')})

                # Verify the spike bar is now bars[-2] (exactly 1 bar old)
                if bars[-2]['t'] == spike['ts']:
                    if ticker in state:
                        # Position was opened by another signal in the meantime — discard
                        print(f'  {ticker}: pending but position now open — discard')
                        ticker_results.append({'ticker': ticker, 'result': 'pending_confirmation',
                                               'confirmed': False,
                                               'reason': 'position_already_open'})
                    elif len(state) >= MAX_POSITIONS:
                        # Max positions reached between spike detection and confirmation
                        print(f'  {ticker}: pending confirmation skipped — max positions')
                        send_discord(f'⏭️ SIGNAL SKIPPED | {ticker} | Max 3 positions open')
                        _log_scan({'timestamp': now_str, 'ticker': ticker,
                                   'event': 'CONF_SKIPPED_MAX_POS'})
                        ticker_results.append({'ticker': ticker, 'result': 'pending_confirmation',
                                               'confirmed': False, 'reason': 'max_positions'})
                    else:
                        sig = build_entry_sig(ticker, spike, bars[-1])
                        if sig is None:
                            print(f'  {ticker}: degenerate stop — discard')
                            ticker_results.append({'ticker': ticker,
                                                   'result': 'pending_confirmation',
                                                   'confirmed': False,
                                                   'reason': 'degenerate_stop'})
                        else:
                            dir_label = 'LONG' if sig['direction'] == 'long' else 'SHORT'
                            print(f'  {ticker}: entering {dir_label} @ ${sig["entry_price"]:.2f} '
                                  f'(confirmation of {spike["vol_ratio"]}x spike)')
                            result = execute_vs_trade(sig, discord_fn=send_discord,
                                                      scan_log_fn=_log_scan)
                            print(f'  {ticker}: {result}')
                            # Reload state in case another ticker was just opened
                            state = load_state()
                            ticker_results.append({'ticker': ticker,
                                                   'result': 'pending_confirmation',
                                                   'confirmed': True, 'direction': dir_label,
                                                   'entry_price': sig['entry_price']})
                else:
                    # bars[-2].t ≠ spike.ts → spike is > 1 bar old, stale
                    print(f'  {ticker}: pending spike stale (spike_ts={spike["ts"]}, '
                          f'bars[-2].t={bars[-2]["t"]}) — discard')
                    _log_scan({'timestamp': now_str, 'ticker': ticker, 'event': 'STALE_SPIKE'})
                    ticker_results.append({'ticker': ticker, 'result': 'skipped',
                                           'reason': 'stale_spike'})
                # Either way, don't carry forward this pending entry
                continue

            # ── 5b. Detect new spike on bars[-1] — gated by entry cutoff ────
            if not can_enter:
                ticker_results.append({'ticker': ticker, 'result': 'skipped',
                                       'reason': 'past_entry_cutoff_3:30pm'})
                continue

            if ticker in state:
                print(f'  {ticker}: position open')
                ticker_results.append({'ticker': ticker, 'result': 'position_open'})
                continue

            if len(state) >= MAX_POSITIONS:
                ticker_results.append({'ticker': ticker, 'result': 'skipped',
                                       'reason': 'max_positions'})
                continue

            spike = detect_spike(bars)
            if spike:
                dir_label = 'LONG' if spike['direction'] == 'long' else 'SHORT'
                print(f'  {ticker}: SPIKE {dir_label} {spike["vol_ratio"]}x '
                      f'{spike["spike_pct"]}% — pending confirmation')
                send_discord(
                    f'👀 VS SIGNAL | {ticker} {dir_label} '
                    f'| Spike: {spike["spike_pct"]:.1f}% move, {spike["vol_ratio"]:.1f}x volume '
                    f'| Waiting for confirmation bar'
                )
                new_pending[ticker] = spike
                _log_scan({'timestamp': now_str, 'event': 'SPIKE_DETECTED',
                           'ticker': ticker, 'direction': dir_label,
                           'vol_ratio': spike['vol_ratio'],
                           'spike_move_pct': spike['spike_pct']})
                ticker_results.append({'ticker': ticker, 'result': 'spike_detected',
                                       'direction': dir_label,
                                       'vol_ratio': spike['vol_ratio'],
                                       'spike_move_pct': spike['spike_pct']})
            else:
                print(f'  {ticker}: no signal')
                ticker_results.append({'ticker': ticker, 'result': 'no_signal'})

        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            print(f'  {ticker}: SCAN ERROR — {err}')
            _log_scan({'timestamp': now_str, 'event': 'TICKER_ERROR',
                       'ticker': ticker, 'error': err})
            ticker_results.append({'ticker': ticker, 'result': 'error', 'reason': err})

    _save_pending(new_pending)

    _log_scan({
        'timestamp':            now_str,
        'event':                'SCAN',
        'market_hours':         True,
        'tickers_checked':      TICKERS,
        'ticker_results':       ticker_results,
        'pending_from_previous': list(pending.keys()),
        'open_positions':       open_positions_at_scan,
    })


# ── DAILY SUMMARY ─────────────────────────────────────────────────────────────

_summary_sent_date = None

def check_daily_summary():
    """Fire EOD summary once after 16:05 ET on weekdays."""
    global _summary_sent_date
    now   = datetime.now(ET)
    today = now.date()
    if (now.weekday() < 5
            and now.hour == 16 and now.minute >= 5
            and _summary_sent_date != today):
        _summary_sent_date = today
        _paper_daily_summary(send_discord)


# ── SCHEDULER ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print('  VS INTRADAY SCANNER  |  SPY  QQQ  AAPL  NVDA  TSLA')
    print('  Variation B: vol>3.0×  target=1.5×  shares  $125 risk')
    print('  Scan every 15 min, 9:30–16:00 ET, Mon–Fri')
    print('=' * 62)

    _init_data_files()
    reconcile_positions_on_startup(send_discord)

    schedule.every(15).minutes.do(run_scan)
    schedule.every(15).minutes.do(check_daily_summary)

    send_discord(
        '✅ VS Intraday system live | SPY QQQ AAPL NVDA TSLA '
        '| Variation B: 3.0× vol, 1.5× target, shares'
    )

    # Run immediately on startup
    run_scan()
    check_daily_summary()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        send_discord('✅ VS Trading system is live and scanning')
        print('Test message sent.')
    else:
        main()
