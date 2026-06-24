#!/usr/bin/env python3
"""
VS intraday share trading — Alpaca PAPER API only.
Trades shares (long/short) on Volume Spike signals. No options.
$125 risk/trade, $500 max notional, Variation B parameters.
"""

import json
import math
import os
from datetime import date, datetime

import pytz
import requests
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PAPER_BASE_URL  = 'https://paper-api.alpaca.markets'   # PAPER only — never changed
ALPACA_KEY      = os.getenv('ALPACA_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET')
ET              = pytz.timezone('US/Eastern')

TICKERS         = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
SIMULATED_ACCOUNT = 5_000.0
RISK_PER_TRADE    = 125.0
MAX_NOTIONAL      = 500.0     # hard cap per trade regardless of stop distance
DRAWDOWN_LIMIT    = 0.10      # pause if paper equity drops >10% from start
MAX_POSITIONS     = 3
COOLDOWN_N        = 3

_DIR             = os.path.dirname(os.path.abspath(__file__))
PAPER_TRADES_LOG = os.path.join(_DIR, 'paper_trades.json')
PAPER_START_FILE = os.path.join(_DIR, 'paper_account_start.json')
TRADE_LOG        = os.path.join(_DIR, 'trade_log.json')
POSITION_STATE   = os.path.join(_DIR, 'position_state.json')


# ── ALPACA HELPERS ─────────────────────────────────────────────────────────────

def _headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type':        'application/json',
    }


def get_account():
    r = requests.get(f'{PAPER_BASE_URL}/v2/account', headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_position(ticker):
    """Return live Alpaca position for ticker, or None if not open."""
    try:
        r = requests.get(f'{PAPER_BASE_URL}/v2/positions/{ticker}',
                         headers=_headers(), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_all_positions():
    r = requests.get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def _place_order(ticker, qty, side):
    """Market order. side: 'buy' or 'sell'."""
    payload = {
        'symbol':        ticker,
        'qty':           str(qty),
        'side':          side,
        'type':          'market',
        'time_in_force': 'day',
    }
    r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                      headers=_headers(), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ── STATE ──────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        with open(POSITION_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict):
    with open(POSITION_STATE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ── STARTING EQUITY ───────────────────────────────────────────────────────────

def _load_start_equity():
    try:
        with open(PAPER_START_FILE) as f:
            return float(json.load(f)['start_equity'])
    except Exception:
        acct   = get_account()
        equity = float(acct['equity'])
        with open(PAPER_START_FILE, 'w') as f:
            json.dump({'start_equity': equity,
                       'recorded_at':  datetime.now().isoformat()}, f, indent=2)
        return equity


# ── LOGGING ───────────────────────────────────────────────────────────────────

def _log(entry):
    """Append to paper_trades.json (full execution detail)."""
    try:
        with open(PAPER_TRADES_LOG) as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append(entry)
    with open(PAPER_TRADES_LOG, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def _log_trade(date_str, result, ticker, pnl):
    """Append to trade_log.json (cooldown tracking: result = 'WIN' or 'LOSS')."""
    try:
        with open(TRADE_LOG) as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append({'date': date_str, 'result': result, 'ticker': ticker, 'pnl': round(pnl, 2)})
    with open(TRADE_LOG, 'w') as f:
        json.dump(log, f, indent=2, default=str)


# ── COOLDOWN ──────────────────────────────────────────────────────────────────

def today_consecutive_losses():
    """Return number of trailing consecutive losses today (from trade_log.json)."""
    try:
        with open(TRADE_LOG) as f:
            trades = json.load(f)
        today = str(date.today())
        count = 0
        for t in reversed([x for x in trades if x.get('date') == today]):
            if t.get('result') == 'LOSS':
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


# ── ENTRY ─────────────────────────────────────────────────────────────────────

def execute_vs_trade(sig, discord_fn=None):
    """
    Place entry share order for a confirmed VS signal.

    sig keys required:
        ticker, direction ('long'|'short'), entry_price, stop_price,
        target_price, spike_bar_high, spike_bar_low, spike_move, vol_ratio
    Returns a short status string.
    """
    ticker    = sig['ticker']
    direction = sig['direction']
    now_str   = datetime.now(ET).isoformat()
    today_str = str(date.today())

    def _skip(reason):
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'SKIPPED', 'reason': reason})
        return f'⏭️ Skipped: {reason}'

    def _block(reason):
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'BLOCKED', 'reason': reason})
        return f'🚫 Blocked: {reason}'

    try:
        # 1. Cooldown
        if today_consecutive_losses() >= COOLDOWN_N:
            return _skip('3 consecutive losses — cooldown active')

        # 2. No second position in same ticker
        state = load_state()
        if ticker in state:
            return _skip(f'position already open in {ticker}')

        # 3. Max concurrent positions
        if len(state) >= MAX_POSITIONS:
            return _skip('max 3 positions open')

        # 4. Drawdown guard
        acct         = get_account()
        equity       = float(acct['equity'])
        start_equity = _load_start_equity()
        if equity < start_equity * (1 - DRAWDOWN_LIMIT):
            msg = (f'📉 Paper account down >10% '
                   f'(${start_equity:,.0f} → ${equity:,.0f}) — trading paused.')
            if discord_fn:
                discord_fn(msg)
            return _block(f'equity ${equity:.0f} below 90% of start ${start_equity:.0f}')

        # 5. Size: shares = floor($125 / stop_dist), capped at $500 notional
        entry_px  = sig['entry_price']
        stop_px   = sig['stop_price']
        stop_dist = abs(entry_px - stop_px)
        if stop_dist < 0.01:
            return _skip('stop distance < $0.01 (degenerate)')

        shares_by_risk    = max(1, math.floor(RISK_PER_TRADE / stop_dist))
        shares_by_notional= max(1, math.floor(MAX_NOTIONAL / entry_px))
        shares            = min(shares_by_risk, shares_by_notional)
        risk_actual       = stop_dist * shares

        # 6. Place market order
        side  = 'buy' if direction == 'long' else 'sell'
        order = _place_order(ticker, shares, side)
        order_id = order.get('id')

        # 7. Persist to state
        state[ticker] = {
            'ticker':         ticker,
            'direction':      direction,
            'entry_price':    entry_px,
            'stop_price':     stop_px,
            'target_price':   sig['target_price'],
            'spike_bar_high': sig['spike_bar_high'],
            'spike_bar_low':  sig['spike_bar_low'],
            'spike_move':     sig['spike_move'],
            'entry_time':     now_str,
            'shares':         shares,
            'entry_order_id': order_id,
            'side':           side,
        }
        _save_state(state)

        _log({
            'timestamp':    now_str,
            'ticker':       ticker,
            'direction':    direction,
            'entry_price':  entry_px,
            'stop_price':   stop_px,
            'target_price': sig['target_price'],
            'shares':       shares,
            'risk':         round(risk_actual, 2),
            'order_id':     order_id,
            'status':       'SUBMITTED',
        })

        # 8. Discord entry alert
        dir_label = 'LONG' if direction == 'long' else 'SHORT'
        if discord_fn:
            discord_fn(
                f'🚀 ENTRY | {ticker} {dir_label} '
                f'| Entry: ${entry_px:.2f} | Stop: ${stop_px:.2f} '
                f'| Target: ${sig["target_price"]:.2f} '
                f'| Shares: {shares} | Risk: ${risk_actual:.0f}'
            )

        return f'Submitted: {side} {shares} {ticker} @ market'

    except Exception as e:
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'ERROR', 'reason': str(e)})
        return f'⚠️ Execution error: {e}'


# ── EXIT ──────────────────────────────────────────────────────────────────────

def _close_position(ticker, entry, reason, current_price, discord_fn=None):
    """
    Place market close order and log P&L.
    reason: 'TARGET' | 'STOP' | 'EOD'
    Returns True on success.
    """
    now_str   = datetime.now(ET).isoformat()
    today_str = str(date.today())
    direction = entry['direction']
    entry_px  = entry['entry_price']
    shares    = entry['shares']

    try:
        close_side = 'buy' if direction == 'short' else 'sell'
        _place_order(ticker, shares, close_side)

        # P&L
        if direction == 'long':
            pnl = (current_price - entry_px) * shares
            pct = (current_price - entry_px) / entry_px * 100
        else:
            pnl = (entry_px - current_price) * shares
            pct = (entry_px - current_price) / entry_px * 100

        result = 'WIN' if pnl > 0 else 'LOSS'

        # Discord alert — exact format from spec
        if reason == 'TARGET':
            msg = (f'✅ TARGET HIT | {ticker} '
                   f'| Exit: ${current_price:.2f} '
                   f'| P&L: +${abs(pnl):.0f} (+{abs(pct):.1f}%)')
        elif reason == 'STOP':
            msg = (f'🔴 STOP HIT | {ticker} '
                   f'| Exit: ${current_price:.2f} '
                   f'| P&L: -${abs(pnl):.0f} (-{abs(pct):.1f}%)')
        else:  # EOD
            sign = '+' if pnl >= 0 else ''
            msg = (f'⏰ EOD CLOSE | {ticker} '
                   f'| Exit: ${current_price:.2f} '
                   f'| P&L: {sign}${pnl:.0f} ({sign}{pct:.1f}%)')

        if discord_fn:
            discord_fn(msg)

        _log({
            'timestamp':   now_str,
            'ticker':      ticker,
            'direction':   direction,
            'entry_price': entry_px,
            'exit_price':  round(current_price, 4),
            'shares':      shares,
            'pnl':         round(pnl, 2),
            'pct':         round(pct, 2),
            'reason':      reason,
            'status':      'CLOSED',
        })
        _log_trade(today_str, result, ticker, pnl)
        return True

    except Exception as e:
        err = f'⚠️ EXIT FAILED | {ticker} | {reason} | {e} | Manual close required'
        if discord_fn:
            discord_fn(err)
        _log({'timestamp': now_str, 'ticker': ticker,
              'event': f'{reason}_FAILED', 'error': str(e), 'status': 'ERROR'})
        return False


# ── POSITION MONITOR ──────────────────────────────────────────────────────────

def monitor_positions(discord_fn=None):
    """
    Called on every 15-min scan. Checks all open positions against stop/target/EOD.
    Exit priority: EOD → TARGET → STOP.
    """
    state = load_state()
    if not state:
        return

    now    = datetime.now(ET)
    is_eod = now.hour == 15 and now.minute >= 50
    to_remove = []

    for ticker, entry in list(state.items()):
        pos = get_position(ticker)
        if pos is None:
            # Alpaca doesn't have it — orphaned state, remove and alert
            to_remove.append(ticker)
            if discord_fn:
                discord_fn(f'⚠️ {ticker} not found in Alpaca — removed from tracking '
                           f'(closed manually, order rejected, or account reset)')
            _log({'timestamp': datetime.now(ET).isoformat(), 'ticker': ticker,
                  'event': 'POSITION_NOT_FOUND', 'status': 'REMOVED'})
            continue

        current_price = float(pos['current_price'])
        direction     = entry['direction']
        stop_px       = entry['stop_price']
        target_px     = entry['target_price']

        reason = None
        if is_eod:
            reason = 'EOD'
        elif direction == 'long':
            if current_price >= target_px:
                reason = 'TARGET'
            elif current_price <= stop_px:
                reason = 'STOP'
        else:  # short
            if current_price <= target_px:
                reason = 'TARGET'
            elif current_price >= stop_px:
                reason = 'STOP'

        if reason:
            ok = _close_position(ticker, entry, reason, current_price, discord_fn)
            if ok:
                to_remove.append(ticker)

    for ticker in to_remove:
        state.pop(ticker, None)
    _save_state(state)


# ── STARTUP RECONCILIATION ────────────────────────────────────────────────────

def reconcile_positions_on_startup(discord_fn=None):
    """
    Re-adopt open Alpaca equity positions not tracked in position_state.json.
    Guards against Railway ephemeral filesystem wiping state on redeploy.
    Stop/target are estimated (±1% / ±1.5%) when spike data is unavailable.
    """
    try:
        alpaca_positions = get_all_positions()
        state            = load_state()
        adopted          = []

        for pos in alpaca_positions:
            symbol = pos['symbol']
            if symbol in state:
                continue
            if pos.get('asset_class') != 'us_equity':
                continue
            if symbol not in TICKERS:
                continue

            direction = 'long' if pos.get('side') == 'long' else 'short'
            entry_px  = float(pos.get('avg_entry_price') or pos.get('current_price', 0))
            shares    = abs(int(float(pos.get('qty', 1))))

            # Approximate stop/target when spike data is lost
            if direction == 'long':
                stop_est   = round(entry_px * 0.99, 4)
                target_est = round(entry_px * 1.015, 4)
            else:
                stop_est   = round(entry_px * 1.01, 4)
                target_est = round(entry_px * 0.985, 4)

            state[symbol] = {
                'ticker':         symbol,
                'direction':      direction,
                'entry_price':    entry_px,
                'stop_price':     stop_est,
                'target_price':   target_est,
                'spike_bar_high': None,
                'spike_bar_low':  None,
                'spike_move':     None,
                'entry_time':     str(date.today()),
                'shares':         shares,
                'entry_order_id': None,
                'side':           'buy' if direction == 'long' else 'sell',
                'reconciled':     True,
            }
            adopted.append(symbol)

        if adopted:
            _save_state(state)
            msg = (f'🔄 Startup reconciliation: {len(adopted)} position(s) re-adopted: '
                   f'{", ".join(adopted)}\n'
                   f'Stop/target estimated (±1%/±1.5% from entry) — spike data unavailable. '
                   f'Update manually if original levels differ.')
            if discord_fn:
                discord_fn(msg)
            print(f'[reconcile] Adopted: {adopted}')
        else:
            print(f'[reconcile] OK — {len(state)} tracked / {len(alpaca_positions)} open in Alpaca')

    except Exception as e:
        print(f'[reconcile] Error (scanner will start anyway): {e}')


# ── DAILY SUMMARY ─────────────────────────────────────────────────────────────

def daily_summary(discord_fn):
    """Post EOD summary to Discord. Scheduled 16:05 ET by scanner.py."""
    try:
        acct         = get_account()
        equity       = float(acct['equity'])
        start_equity = _load_start_equity()
        total_ret    = (equity - start_equity) / start_equity * 100
        today_str    = str(date.today())

        try:
            with open(PAPER_TRADES_LOG) as f:
                paper_log = json.load(f)
            entered_today = sum(
                1 for p in paper_log
                if p.get('timestamp', '').startswith(today_str) and p.get('status') == 'SUBMITTED'
            )
            closed_today = [
                p for p in paper_log
                if p.get('timestamp', '').startswith(today_str) and p.get('status') == 'CLOSED'
            ]
        except Exception:
            entered_today = 0
            closed_today  = []

        wins      = sum(1 for p in closed_today if p.get('pnl', 0) > 0)
        losses    = sum(1 for p in closed_today if p.get('pnl', 0) <= 0)
        pnl_today = sum(p.get('pnl', 0) for p in closed_today)

        pnl_lines = ''
        if closed_today:
            pnl_lines = '\nClosed today:\n' + '\n'.join(
                f'  {p.get("ticker","?")} {p.get("reason","?")} ${p.get("pnl",0):+.2f}'
                for p in closed_today
            )

        open_str = ', '.join(load_state().keys()) or 'none'

        discord_fn(
            f'📊 **Daily Summary — {today_str}**\n'
            f'Entered: {entered_today}  |  Closed: {len(closed_today)}'
            f'  (W: {wins} / L: {losses})\n'
            f'Today P&L: ${pnl_today:+.2f}'
            f'{pnl_lines}\n'
            f'Open positions: {open_str}\n'
            f'Paper equity: ${equity:,.2f}\n'
            f'Total return since start: {total_ret:+.2f}%'
        )

    except Exception as e:
        discord_fn(f'⚠️ Daily summary error: {e}')


# ── CONNECTION TEST ────────────────────────────────────────────────────────────

def test_connection():
    print('─' * 60)
    print('  ALPACA PAPER — VS SHARE TRADING  CONNECTION TEST')
    print(f'  Base URL: {PAPER_BASE_URL}')
    print('─' * 60)
    acct = get_account()
    print(f'  Status          : {acct["status"]}')
    print(f'  Equity          : ${float(acct["equity"]):,.2f}')
    print(f'  Cash            : ${float(acct["cash"]):,.2f}')
    print(f'  Shorting enabled: {acct.get("shorting_enabled")}')
    positions = get_all_positions()
    print(f'  Open positions  : {len(positions)}')
    for p in positions:
        print(f'    {p["symbol"]:<8} {p.get("side","?")} qty={p["qty"]}  '
              f'price=${float(p.get("current_price",0)):.2f}')
    print('  ✓ Connection OK')
    print('─' * 60)


if __name__ == '__main__':
    test_connection()
