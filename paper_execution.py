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

_DIR              = os.path.dirname(os.path.abspath(__file__))
PAPER_TRADES_LOG  = os.path.join(_DIR, 'paper_trades.json')
PAPER_START_FILE  = os.path.join(_DIR, 'paper_account_start.json')
TRADE_LOG         = os.path.join(_DIR, 'trade_log.json')
POSITION_STATE    = os.path.join(_DIR, 'position_state.json')

# Exit thresholds — must match backtest constants exactly
STOP_PCT      = -0.45   # close all
TAKE_HALF_PCT =  0.50   # sell half
TRAIL_PCT     =  0.25   # close remaining after partial (trail floor)
FULL_EXIT_PCT =  1.00   # close all remaining
MAX_HOLD_DAYS =  7      # force-close after this many calendar days


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


def get_all_positions():
    """Return list of all open Alpaca positions (empty list if none)."""
    r = requests.get(f'{PAPER_BASE_URL}/v2/positions',
                     headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()  # list of position dicts


# ── POSITION STATE ────────────────────────────────────────────────────────────
# Persists entry metadata that Alpaca doesn't track: entry date, direction,
# and whether the partial (+50%) exit has already been taken for each position.

def _load_state() -> dict:
    try:
        with open(POSITION_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict):
    with open(POSITION_STATE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


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


# ── EXIT ORDER HELPERS ────────────────────────────────────────────────────────

def _exit_all(symbol, exit_type, pnl_pct, dollar_pnl, price, qty, discord_fn):
    """Close 100% of the remaining position via Alpaca. Returns 'OK' or 'ERROR'."""
    now_str = datetime.now(ET).isoformat()
    try:
        close_position(symbol)

        _EMOJI = {
            'STOP':        '🛑',
            'FULL_TARGET': '✅',
            'TRAIL_STOP':  '🔶',
            'TIME_EXIT':   '⏱️',
        }
        _LABEL = {
            'STOP':        f'STOP {STOP_PCT*100:.0f}%',
            'FULL_TARGET': f'FULL EXIT +{FULL_EXIT_PCT*100:.0f}%',
            'TRAIL_STOP':  f'TRAIL STOP +{TRAIL_PCT*100:.0f}%',
            'TIME_EXIT':   f'TIME EXIT ({MAX_HOLD_DAYS}d)',
        }
        sign = '+' if dollar_pnl >= 0 else ''
        msg = (f"{_EMOJI.get(exit_type, '📤')} {_LABEL.get(exit_type, exit_type)} "
               f"| {symbol} | Closed {qty:.4f} shares @ ${price:.2f} "
               f"| P&L: {sign}${dollar_pnl:.2f} ({sign}{pnl_pct*100:.1f}%)")
        if discord_fn:
            discord_fn(msg)
        _log({'timestamp': now_str, 'ticker': symbol, 'event': exit_type,
              'qty': qty, 'price': price, 'pnl_pct': round(pnl_pct, 4),
              'dollar_pnl': round(dollar_pnl, 2), 'status': 'CLOSED'})
        return 'OK'

    except Exception as e:
        err = (f'⚠️ EXIT ORDER FAILED | {symbol} | {exit_type} '
               f'| {e} | Manual close required')
        if discord_fn:
            discord_fn(err)
        _log({'timestamp': now_str, 'ticker': symbol, 'event': f'{exit_type}_FAILED',
              'error': str(e), 'status': 'ERROR'})
        return 'ERROR'


def _exit_partial(symbol, half_qty, close_side, pnl_pct, dollar_pnl, price, discord_fn):
    """Sell exactly half_qty shares (partial +50% exit). Returns 'OK' or 'ERROR'."""
    now_str = datetime.now(ET).isoformat()
    try:
        payload = {
            'symbol':        symbol,
            'qty':           str(round(half_qty, 4)),
            'side':          close_side,   # 'sell' for long, 'buy' for short
            'type':          'market',
            'time_in_force': 'day',
        }
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=10)
        r.raise_for_status()

        sign = '+' if dollar_pnl >= 0 else ''
        msg = (f'🎯 HALF EXIT +{TAKE_HALF_PCT*100:.0f}% | {symbol} '
               f'| Sold {half_qty:.4f} shares @ ${price:.2f} '
               f'| {sign}${dollar_pnl:.2f} ({sign}{pnl_pct*100:.1f}%) '
               f'| Trailing remaining half (floor +{TRAIL_PCT*100:.0f}%)')
        if discord_fn:
            discord_fn(msg)
        _log({'timestamp': now_str, 'ticker': symbol, 'event': 'PARTIAL_EXIT',
              'qty': half_qty, 'price': price, 'pnl_pct': round(pnl_pct, 4),
              'dollar_pnl': round(dollar_pnl, 2), 'status': 'PARTIAL_CLOSED'})
        return 'OK'

    except Exception as e:
        err = (f'⚠️ PARTIAL EXIT FAILED | {symbol} '
               f'| {e} | Manual intervention required')
        if discord_fn:
            discord_fn(err)
        _log({'timestamp': now_str, 'ticker': symbol, 'event': 'PARTIAL_EXIT_FAILED',
              'error': str(e), 'status': 'ERROR'})
        return 'ERROR'


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

        # Write entry metadata for exit monitor — Alpaca doesn't track entry
        # date or partial-exit state, so we persist it ourselves.
        state = _load_state()
        state[ticker] = {
            'entry_date':    str(date.today()),
            'entry_price':   sig['close'],
            'direction':     direction,
            'side':          side,
            'partial_taken': False,
            'order_id':      order.get('id'),
        }
        _save_state(state)

        action = 'bought' if side == 'buy' else 'shorted'
        return f'🤖 Paper trade executed: {action} ~{est_shares:.4f} shares (${dollar_risk:.0f})'

    except Exception as e:
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'ERROR', 'reason': str(e)})
        return f'⚠️ Paper execution error: {e}'


# ── POSITION MONITOR ──────────────────────────────────────────────────────────

def monitor_positions(discord_fn=None):
    """
    Called on every hourly scan (inside market hours).
    Checks every tracked position against all five exit conditions in priority
    order and places the appropriate Alpaca order automatically.

    Exit priority (highest to lowest):
      1. Stop loss      : unrealized P&L ≤ -45%  → close ALL remaining
      2. Full target    : unrealized P&L ≥ +100% → close ALL remaining
      3. Trail stop     : partial already taken AND P&L ≤ +25% → close ALL remaining
      4. Partial exit   : unrealized P&L ≥ +50%, partial NOT yet taken → sell HALF
      5. Time exit      : calendar days since entry ≥ 7 → close ALL remaining

    If two conditions are simultaneously true (e.g. price past +100% on day 7),
    the higher-priority condition fires and the lower one is skipped — no
    double-execution is possible because each branch is an elif.
    """
    state = _load_state()
    if not state:
        return

    today     = date.today()
    now_str   = datetime.now(ET).isoformat()
    to_remove = []

    for ticker, entry in list(state.items()):
        pos = get_position(ticker)

        # Position missing from Alpaca — closed manually or order never filled
        if pos is None:
            to_remove.append(ticker)
            msg = (f'⚠️ {ticker} position not found in Alpaca '
                   f'— removed from tracking (closed manually or order lapsed)')
            if discord_fn:
                discord_fn(msg)
            _log({'timestamp': now_str, 'ticker': ticker,
                  'event': 'POSITION_NOT_FOUND', 'status': 'REMOVED'})
            continue

        pnl_pct     = float(pos['unrealized_plpc'])   # e.g. 0.50 = +50%
        dollar_pnl  = float(pos['unrealized_pl'])
        qty         = float(pos['qty'])
        price       = float(pos['current_price'])
        alpaca_side = pos['side']                      # 'long' or 'short'
        close_side  = 'sell' if alpaca_side == 'long' else 'buy'

        entry_date    = date.fromisoformat(entry['entry_date'])
        days_held     = (today - entry_date).days
        partial_taken = entry.get('partial_taken', False)

        # ── Priority 1: Stop loss ─────────────────────────────────────────────
        if pnl_pct <= STOP_PCT:
            result = _exit_all(ticker, 'STOP', pnl_pct, dollar_pnl, price, qty, discord_fn)
            if result == 'OK':
                to_remove.append(ticker)

        # ── Priority 2: Full target ───────────────────────────────────────────
        elif pnl_pct >= FULL_EXIT_PCT:
            result = _exit_all(ticker, 'FULL_TARGET', pnl_pct, dollar_pnl, price, qty, discord_fn)
            if result == 'OK':
                to_remove.append(ticker)

        # ── Priority 3: Trail stop (only active after partial taken) ──────────
        elif partial_taken and pnl_pct <= TRAIL_PCT:
            result = _exit_all(ticker, 'TRAIL_STOP', pnl_pct, dollar_pnl, price, qty, discord_fn)
            if result == 'OK':
                to_remove.append(ticker)

        # ── Priority 4: Partial exit (+50%, first time only) ─────────────────
        elif pnl_pct >= TAKE_HALF_PCT and not partial_taken:
            half_qty    = round(qty / 2, 4)
            half_dollar = dollar_pnl / 2
            result = _exit_partial(ticker, half_qty, close_side,
                                   pnl_pct, half_dollar, price, discord_fn)
            if result == 'OK':
                entry['partial_taken'] = True
                state[ticker] = entry    # update in-memory state before saving

        # ── Priority 5: Time exit (7-day force-close) ─────────────────────────
        elif days_held >= MAX_HOLD_DAYS:
            result = _exit_all(ticker, 'TIME_EXIT', pnl_pct, dollar_pnl, price, qty, discord_fn)
            if result == 'OK':
                to_remove.append(ticker)

    # Remove fully-closed positions from state
    for ticker in to_remove:
        state.pop(ticker, None)
    _save_state(state)


# ── STARTUP RECONCILIATION ────────────────────────────────────────────────────

def reconcile_positions_on_startup(discord_fn=None):
    """
    Call once in main() before the scheduler loop starts.

    Fetches every open position from Alpaca and creates a conservative
    state entry for any that are not already in position_state.json.
    This guards against the ephemeral Railway filesystem wiping state on
    redeploy while a position remains open.

    Conservative defaults for adopted positions:
      entry_date    = today  (7-day timer resets — safe, not dangerous)
      entry_price   = Alpaca's avg_entry_price (correct cost basis)
      partial_taken = False  (may re-trigger a partial if price is still
                              above +50%, but won't double-close since the
                              monitor uses Alpaca's live qty each cycle)
      reconciled    = True   (audit flag so you can see in paper_trades.json)
    """
    try:
        alpaca_positions = get_all_positions()
        state            = _load_state()
        adopted          = []

        for pos in alpaca_positions:
            symbol = pos['symbol']
            if symbol in state:
                continue  # already tracked — leave existing state untouched

            alpaca_side = pos['side']                       # 'long' or 'short'
            entry_price = float(
                pos.get('avg_entry_price') or pos.get('current_price', 0)
            )

            state[symbol] = {
                'entry_date':    str(date.today()),
                'entry_price':   entry_price,
                'direction':     'bullish' if alpaca_side == 'long' else 'bearish',
                'side':          'buy'     if alpaca_side == 'long' else 'sell',
                'partial_taken': False,
                'order_id':      None,
                'reconciled':    True,
            }
            adopted.append(symbol)

        if adopted:
            _save_state(state)
            msg = (
                f'🔄 Startup reconciliation: {len(adopted)} orphaned position(s) '
                f'adopted from Alpaca — {", ".join(adopted)}.\n'
                f'entry_date reset to today; partial_taken=False. '
                f'Verify manually if a partial exit was already taken before restart.'
            )
            if discord_fn:
                discord_fn(msg)
            print(f'[reconcile] Adopted: {adopted}')
        else:
            tracked = len(state)
            open_n  = len(alpaca_positions)
            print(f'[reconcile] OK — {tracked} tracked / {open_n} open in Alpaca')

    except Exception as e:
        # Never let reconciliation failure prevent the scanner from starting
        print(f'[reconcile] Error (scanner will start anyway): {e}')


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
