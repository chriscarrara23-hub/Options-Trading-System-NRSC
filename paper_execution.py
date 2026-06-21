#!/usr/bin/env python3
"""
Paper trading execution — Alpaca PAPER API only.

Base URL is hard-coded to paper-api.alpaca.markets.
This module never references the live trading endpoint.
"""

import json
import os
import time
from datetime import date, datetime

import pytz
import requests
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PAPER_BASE_URL        = 'https://paper-api.alpaca.markets'   # PAPER only
ALPACA_KEY            = os.getenv('ALPACA_KEY')
ALPACA_SECRET         = os.getenv('ALPACA_SECRET')
PAPER_ACCOUNT_BALANCE = 5_000.0   # change this to rescale position sizes
RISK_PER_TRADE_PCT    = 0.025     # 2.5% → $125 at $5k balance
DRAWDOWN_LIMIT        = 0.10      # block new trades if equity drops >10% from start
ET                    = pytz.timezone('US/Eastern')

_DIR             = os.path.dirname(os.path.abspath(__file__))
PAPER_TRADES_LOG = os.path.join(_DIR, 'paper_trades.json')
PAPER_START_FILE = os.path.join(_DIR, 'paper_account_start.json')
TRADE_LOG        = os.path.join(_DIR, 'trade_log.json')


# ── ALPACA REST HELPERS ────────────────────────────────────────────────────────

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


def get_position(symbol):
    """Return open position dict or None."""
    try:
        r = requests.get(f'{PAPER_BASE_URL}/v2/positions/{symbol}',
                         headers=_headers(), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _place_notional_order(symbol, side, notional):
    """Market day order sized by dollar notional."""
    payload = {
        'symbol':        symbol,
        'notional':      str(round(notional, 2)),
        'side':          side,
        'type':          'market',
        'time_in_force': 'day',
    }
    r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                      headers=_headers(), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def _place_qty_order(symbol, side, qty):
    """Market day order for an exact share count (used for test only)."""
    payload = {
        'symbol':        symbol,
        'qty':           str(qty),
        'side':          side,
        'type':          'market',
        'time_in_force': 'day',
    }
    r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                      headers=_headers(), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def _cancel_order(order_id):
    r = requests.delete(f'{PAPER_BASE_URL}/v2/orders/{order_id}',
                        headers=_headers(), timeout=10)
    return r.status_code in (200, 204)


def close_position(symbol):
    """Market order to close all shares of a position."""
    r = requests.delete(f'{PAPER_BASE_URL}/v2/positions/{symbol}',
                        headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


# ── STARTING EQUITY BASELINE ───────────────────────────────────────────────────

def _load_start_equity():
    """Return starting equity; initialise from live account on first call."""
    try:
        with open(PAPER_START_FILE) as f:
            return float(json.load(f)['start_equity'])
    except Exception:
        acct   = get_account()
        equity = float(acct['equity'])
        with open(PAPER_START_FILE, 'w') as f:
            json.dump({
                'start_equity': equity,
                'recorded_at':  datetime.now().isoformat(),
            }, f, indent=2)
        return equity


# ── TRADE LOGGING ──────────────────────────────────────────────────────────────

def _log(entry):
    try:
        with open(PAPER_TRADES_LOG) as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append(entry)
    with open(PAPER_TRADES_LOG, 'w') as f:
        json.dump(log, f, indent=2, default=str)


# ── COOLDOWN (reuses trade_log logic) ─────────────────────────────────────────

def _today_consecutive_losses():
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


# ── MAIN EXECUTION ENTRY POINT ────────────────────────────────────────────────

def execute_paper_trade(sig, discord_fn=None):
    """
    Attempt to paper-trade a signal returned by scanner.evaluate_signal().
    Returns a one-line string to append to the Discord alert.
    """
    ticker    = sig['ticker']
    direction = sig['direction']
    side      = 'buy' if direction == 'bullish' else 'sell'
    now_str   = datetime.now(ET).isoformat()

    def _skip(reason):
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'SKIPPED', 'reason': reason})
        return f'⏭️ Paper trade skipped: {reason}'

    def _block(reason):
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'BLOCKED', 'reason': reason})
        return f'🚫 Paper trade blocked: {reason}'

    try:
        # 1. Same-day cooldown
        if _today_consecutive_losses() >= 3:
            return _skip('3 consecutive losses today — cooldown active')

        # 2. Account health
        acct         = get_account()
        equity       = float(acct['equity'])
        start_equity = _load_start_equity()
        if equity < start_equity * (1 - DRAWDOWN_LIMIT):
            msg = (f'📉 Paper account down >10% from start '
                   f'(${start_equity:,.0f} → ${equity:,.0f}) — '
                   f'trading paused, review needed.')
            if discord_fn:
                discord_fn(msg)
            return _block(f'equity ${equity:.2f} below 90% of start ${start_equity:.2f}')

        # 3. No adding to existing positions
        pos = get_position(ticker)
        if pos is not None:
            return _skip(f'position already open ({pos.get("qty", "?")} shares)')

        # 4. Shorting availability for bearish signals
        if side == 'sell' and not acct.get('shorting_enabled', True):
            return _skip('shorting not enabled on this paper account')

        # 5. Size and submit
        dollar_risk = PAPER_ACCOUNT_BALANCE * RISK_PER_TRADE_PCT
        order       = _place_notional_order(ticker, side, dollar_risk)
        est_shares  = round(dollar_risk / sig['close'], 4)

        _log({
            'timestamp':  now_str,
            'ticker':     ticker,
            'direction':  direction,
            'side':       side,
            'notional':   dollar_risk,
            'est_shares': est_shares,
            'price':      sig['close'],
            'order_id':   order.get('id'),
            'status':     'SUBMITTED',
        })

        action = 'bought' if side == 'buy' else 'shorted'
        return f'🤖 Paper trade executed: {action} ~{est_shares:.4f} shares (${dollar_risk:.0f})'

    except Exception as e:
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'ERROR', 'reason': str(e)})
        return f'⚠️ Paper execution error: {e}'


# ── DAILY SUMMARY ─────────────────────────────────────────────────────────────

def daily_summary(discord_fn):
    """Post EOD paper account summary to Discord. Wired to 16:05 ET in scanner."""
    try:
        acct         = get_account()
        equity       = float(acct['equity'])
        start_equity = _load_start_equity()
        total_ret    = (equity - start_equity) / start_equity * 100

        today_str = str(date.today())

        # Trade log stats
        try:
            with open(TRADE_LOG) as f:
                all_trades = json.load(f)
            today_trades = [t for t in all_trades if t.get('date') == today_str]
            wins   = sum(1 for t in today_trades if t.get('result') == 'WIN')
            losses = sum(1 for t in today_trades if t.get('result') == 'LOSS')
        except Exception:
            wins = losses = 0

        # Paper execution count today
        try:
            with open(PAPER_TRADES_LOG) as f:
                paper_log = json.load(f)
            paper_today = sum(
                1 for p in paper_log
                if p.get('timestamp', '').startswith(today_str)
                and p.get('status') == 'SUBMITTED'
            )
        except Exception:
            paper_today = 0

        msg = (
            f'📊 **Daily Summary — {today_str}**\n'
            f'Paper trades executed: {paper_today}\n'
            f'Trade log today: {wins}W / {losses}L\n'
            f'Paper account equity: ${equity:,.2f}\n'
            f'Total return since start: {total_ret:+.2f}%'
        )
        discord_fn(msg)

    except Exception as e:
        discord_fn(f'⚠️ Daily summary error: {e}')


# ── CONNECTION TEST ────────────────────────────────────────────────────────────

def test_connection():
    """
    Verify paper API auth, place 1-share SPY order, then cancel or close it.
    Safe to run during market hours or when closed.
    """
    print('─' * 62)
    print('  ALPACA PAPER API — CONNECTION TEST')
    print('  Base URL:', PAPER_BASE_URL)
    print('─' * 62)

    print('\n1. Fetching account …')
    acct = get_account()
    print(f'   Account ID  : {acct["id"]}')
    print(f'   Status      : {acct["status"]}')
    print(f'   Equity      : ${float(acct["equity"]):,.2f}')
    print(f'   Cash        : ${float(acct["cash"]):,.2f}')
    print(f'   Shorting    : {acct.get("shorting_enabled")}')
    print(f'   Pattern Day : {acct.get("pattern_day_trader")}')

    print('\n2. Checking for existing SPY position …')
    pos = get_position('SPY')
    if pos:
        print(f'   Existing position: {pos["qty"]} shares. Skipping test order.')
        print('\n   ✓ Auth confirmed via account + position fetch.')
        return

    print('   No existing position — placing 1-share market order …')
    order = _place_qty_order('SPY', 'buy', 1)
    oid   = order['id']
    print(f'   Order ID    : {oid}')
    print(f'   Symbol      : {order["symbol"]}')
    print(f'   Side        : {order["side"]}')
    print(f'   Qty         : {order["qty"]}')
    print(f'   Status      : {order["status"]}')

    print('\n3. Waiting 3s …')
    time.sleep(3)

    # Fetch updated order state
    r = requests.get(f'{PAPER_BASE_URL}/v2/orders/{oid}',
                     headers=_headers(), timeout=10)
    updated = r.json() if r.ok else {}
    filled_status = updated.get('status', order['status'])
    print(f'   Order status after wait: {filled_status}')

    print('\n4. Cleaning up …')
    if filled_status in ('filled', 'partially_filled'):
        try:
            close = close_position('SPY')
            print(f'   Position closed — order {close.get("id", "n/a")}')
        except Exception as e:
            print(f'   Close position error: {e}')
    else:
        cancelled = _cancel_order(oid)
        print(f'   Order cancelled: {cancelled}')

    print('\n   ✓ Test complete. Paper API authenticated and functional.')
    print('─' * 62)


if __name__ == '__main__':
    test_connection()
