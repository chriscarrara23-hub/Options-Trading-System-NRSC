#!/usr/bin/env python3
"""
Paper trading execution — Alpaca PAPER API only.
Trades real options contracts (long calls for bullish, long puts for bearish).
Base URL is hard-coded to paper-api.alpaca.markets — never changed.
"""

import json
import os
import re
from datetime import date, datetime, timedelta

import pytz
import requests
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PAPER_BASE_URL    = 'https://paper-api.alpaca.markets'  # PAPER only
ALPACA_KEY        = os.getenv('ALPACA_KEY')
ALPACA_SECRET     = os.getenv('ALPACA_SECRET')
ET                = pytz.timezone('US/Eastern')

# Size trades as if managing a $5k real account (2.5% = $125 risk budget).
# The paper account's $100k buying power absorbs orders without rejection.
SIMULATED_ACCOUNT  = 5_000.0
RISK_PER_TRADE_PCT = 0.025
RISK_BUDGET        = SIMULATED_ACCOUNT * RISK_PER_TRADE_PCT   # $125
DRAWDOWN_LIMIT     = 0.10

# Contract selection
TARGET_DTE        = 7
MIN_DTE           = 4
MAX_DTE           = 8
MIN_CONTRACT_COST = 75     # 0.6× budget floor — avoids deep junk OTM
MAX_CONTRACT_COST = 250    # 2.0× budget cap   — catches QQQ far-OTM when affordable

# Exit thresholds (match backtest constants)
STOP_PCT         = -0.45
TAKE_HALF_PCT    =  0.50   # partial snapshot trigger (no actual sell — only 1 contract)
FULL_EXIT_PCT    =  1.00
BREAKEVEN_STOP   =  0.00   # trail floor after partial snapshot
OPTIONS_MAX_HOLD =  6      # day-6 time exit before Alpaca expiry-day risk mgmt at 3:30pm

_DIR             = os.path.dirname(os.path.abspath(__file__))
PAPER_TRADES_LOG = os.path.join(_DIR, 'paper_trades.json')
PAPER_START_FILE = os.path.join(_DIR, 'paper_account_start.json')
TRADE_LOG        = os.path.join(_DIR, 'trade_log.json')
POSITION_STATE   = os.path.join(_DIR, 'position_state.json')


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
    """Return open position dict for any symbol (equity or OCC option) or None."""
    try:
        r = requests.get(f'{PAPER_BASE_URL}/v2/positions/{symbol}',
                         headers=_headers(), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_all_positions():
    """Return list of all open Alpaca positions (shares + options)."""
    r = requests.get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def _sell_option(option_symbol, qty=1):
    """Place a market sell-to-close order for a long options position."""
    payload = {
        'symbol':        option_symbol,
        'qty':           str(qty),
        'side':          'sell',
        'type':          'market',
        'time_in_force': 'day',
    }
    r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                      headers=_headers(), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ── OCC SYMBOL PARSER ─────────────────────────────────────────────────────────

_OCC_RE = re.compile(r'^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$')


def _parse_occ(symbol):
    """Parse OCC option symbol. Returns dict with underlying/type/strike/expiration or None."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    yy, mm, dd = m.group(2), m.group(3), m.group(4)
    return {
        'underlying':  m.group(1),
        'option_type': 'call' if m.group(5) == 'C' else 'put',
        'strike':      int(m.group(6)) / 1000,
        'expiration':  f'20{yy}-{mm}-{dd}',
    }


# ── CONTRACT SELECTION ─────────────────────────────────────────────────────────

def select_contract(ticker, direction, current_price):
    """
    Fetch the live Alpaca options chain and return the best contract for this signal.

    Selection rules:
      1. Expiration closest to 7 DTE within the 4–8 DTE window.
      2. Strike where 1-contract cost (premium × 100) is $75–$200.
      3. Among affordable strikes, prefer the one closest to ATM.

    Returns a dict {symbol, strike, expiry, premium, cost, dte, atm_diff} or None.
    """
    option_type = 'call' if direction == 'bullish' else 'put'
    today       = date.today()

    params = {
        'underlying_symbols':  ticker,
        'expiration_date_gte': str(today + timedelta(days=MIN_DTE)),
        'expiration_date_lte': str(today + timedelta(days=MAX_DTE + 2)),
        'type':                option_type,
        'limit':               500,
    }
    r = requests.get(f'{PAPER_BASE_URL}/v2/options/contracts',
                     headers=_headers(), params=params, timeout=10)
    r.raise_for_status()
    contracts = r.json().get('option_contracts', [])

    if not contracts:
        return None

    # Build map of expiration → DTE, keep only those in the 4–8 DTE window
    valid_exps = {}
    for c in contracts:
        dte = (date.fromisoformat(c['expiration_date']) - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            valid_exps[c['expiration_date']] = dte

    if not valid_exps:
        return None

    # Pick expiration closest to TARGET_DTE; break ties in favour of longer DTE
    best_exp = min(valid_exps, key=lambda e: (abs(valid_exps[e] - TARGET_DTE), -valid_exps[e]))

    # Score candidates at this expiration
    candidates = []
    for c in contracts:
        if c['expiration_date'] != best_exp:
            continue
        raw = c.get('close_price')
        if not raw:
            continue
        try:
            premium = float(raw)
        except (ValueError, TypeError):
            continue
        if premium <= 0:
            continue

        strike = float(c['strike_price'])
        cost   = premium * 100
        if MIN_CONTRACT_COST <= cost <= MAX_CONTRACT_COST:
            candidates.append({
                'symbol':   c['symbol'],
                'strike':   strike,
                'premium':  premium,
                'cost':     cost,
                'dte':      valid_exps[best_exp],
                'expiry':   best_exp,
                'atm_diff': abs(strike - current_price),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x['atm_diff'])
    return candidates[0]


# ── POSITION STATE ─────────────────────────────────────────────────────────────

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
    try:
        with open(PAPER_START_FILE) as f:
            return float(json.load(f)['start_equity'])
    except Exception:
        acct   = get_account()
        equity = float(acct['equity'])
        with open(PAPER_START_FILE, 'w') as f:
            json.dump({'start_equity': equity, 'recorded_at': datetime.now().isoformat()}, f, indent=2)
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


# ── COOLDOWN ──────────────────────────────────────────────────────────────────

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


# ── EXIT HELPER ───────────────────────────────────────────────────────────────

_EXIT_EMOJI = {
    'STOP':        '🔴',
    'FULL_TARGET': '✅',
    'TRAIL_STOP':  '🔶',
    'TIME_EXIT':   '⏰',
}

_EXIT_LABEL = {
    'STOP':        f'STOP {STOP_PCT*100:.0f}%',
    'FULL_TARGET': f'FULL EXIT +{FULL_EXIT_PCT*100:.0f}%',
    'TRAIL_STOP':  'TRAIL STOP (breakeven)',
    'TIME_EXIT':   f'TIME EXIT (day {OPTIONS_MAX_HOLD})',
}


def _exit_option(option_symbol, exit_type, entry_premium, pos, discord_fn):
    """
    Market sell-to-close the options position. pos is the live Alpaca position dict.
    Returns 'OK' or 'ERROR'.
    """
    now_str         = datetime.now(ET).isoformat()
    current_premium = float(pos['current_price'])
    dollar_pnl      = float(pos['unrealized_pl'])
    pnl_pct         = float(pos['unrealized_plpc'])
    sign            = '+' if dollar_pnl >= 0 else ''

    try:
        _sell_option(option_symbol)

        msg = (
            f"{_EXIT_EMOJI.get(exit_type, '📤')} {_EXIT_LABEL.get(exit_type, exit_type)} "
            f"| {option_symbol} "
            f"| Entry ${entry_premium:.2f} → Exit ${current_premium:.2f} "
            f"| P&L: {sign}${dollar_pnl:.2f} ({sign}{pnl_pct*100:.1f}%)"
        )
        if discord_fn:
            discord_fn(msg)

        _log({
            'timestamp':     now_str,
            'symbol':        option_symbol,
            'event':         exit_type,
            'entry_premium': entry_premium,
            'exit_premium':  current_premium,
            'dollar_pnl':    round(dollar_pnl, 2),
            'pnl_pct':       round(pnl_pct, 4),
            'status':        'CLOSED',
        })
        return 'OK'

    except Exception as e:
        err = (f'⚠️ EXIT ORDER FAILED | {option_symbol} | {exit_type} '
               f'| {e} | Manual close required')
        if discord_fn:
            discord_fn(err)
        _log({'timestamp': now_str, 'symbol': option_symbol,
              'event': f'{exit_type}_FAILED', 'error': str(e), 'status': 'ERROR'})
        return 'ERROR'


# ── MAIN EXECUTION ENTRY POINT ────────────────────────────────────────────────

def execute_paper_trade(sig, discord_fn=None):
    """
    Attempt to enter a paper options trade for a scanner signal.
    Returns a one-line string appended to the Discord alert in scanner.py.
    """
    ticker    = sig['ticker']
    direction = sig['direction']
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
        # 1. Cooldown
        if _today_consecutive_losses() >= 3:
            return _skip('3 consecutive losses today — cooldown active')

        # 2. Entry guard: no doubling into the same underlying
        state = _load_state()
        if any(e.get('underlying') == ticker for e in state.values()):
            return _skip(f'position already open in {ticker}')

        # 3. Account health vs paper equity
        acct         = get_account()
        equity       = float(acct['equity'])
        start_equity = _load_start_equity()
        if equity < start_equity * (1 - DRAWDOWN_LIMIT):
            msg = (f'📉 Paper account down >10% from start '
                   f'(${start_equity:,.0f} → ${equity:,.0f}) — trading paused.')
            if discord_fn:
                discord_fn(msg)
            return _block(f'equity ${equity:.2f} below 90% of start ${start_equity:.2f}')

        # 4. Select contract from live chain
        contract = select_contract(ticker, direction, sig['close'])
        if contract is None:
            # Skip logged to paper_trades.json; reason appears in scanner signal alert
            return _skip(f'no contract in ${MIN_CONTRACT_COST}–${MAX_CONTRACT_COST} range')

        option_symbol = contract['symbol']
        premium       = contract['premium']
        strike        = contract['strike']
        expiry        = contract['expiry']
        cost          = contract['cost']

        # 5. Place buy-to-open order (always long — calls for bullish, puts for bearish)
        payload = {
            'symbol':        option_symbol,
            'qty':           '1',
            'side':          'buy',
            'type':          'market',
            'time_in_force': 'day',
        }
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=10)
        r.raise_for_status()
        order = r.json()

        # 6. Persist state — key is the OCC option symbol
        state[option_symbol] = {
            'underlying':                ticker,
            'direction':                 direction,
            'option_type':               'call' if direction == 'bullish' else 'put',
            'strike':                    strike,
            'expiration':                expiry,
            'entry_premium':             premium,
            'contracts':                 1,
            'underlying_price_at_entry': sig['close'],
            'entry_date':                str(date.today()),
            'partial_taken':             False,
            'realized_pct':              None,
            'order_id':                  order.get('id'),
        }
        _save_state(state)

        _log({
            'timestamp':     now_str,
            'ticker':        ticker,
            'symbol':        option_symbol,
            'direction':     direction,
            'strike':        strike,
            'expiration':    expiry,
            'entry_premium': premium,
            'cost':          cost,
            'order_id':      order.get('id'),
            'status':        'SUBMITTED',
        })

        option_label = 'CALL' if direction == 'bullish' else 'PUT'
        entry_msg = (
            f'🎯 OPTIONS TRADE | {option_label} {ticker} ${strike:.0f} exp {expiry} '
            f'| Premium: ${premium:.2f} | Cost: ${cost:.0f} '
            f'| Underlying: ${sig["close"]:.2f}'
        )
        if discord_fn:
            discord_fn(entry_msg)

        return f'🤖 Options order submitted: {option_symbol} × 1 (${cost:.0f})'

    except Exception as e:
        _log({'timestamp': now_str, 'ticker': ticker, 'direction': direction,
              'status': 'ERROR', 'reason': str(e)})
        return f'⚠️ Paper execution error: {e}'


# ── POSITION MONITOR ──────────────────────────────────────────────────────────

def monitor_positions(discord_fn=None):
    """
    Called on every hourly scan (inside market hours).

    Exit priority (options-specific):
      1. Stop loss       : pnl_pct ≤ -45%              → market sell-to-close
      2. Full target     : pnl_pct ≥ +100%             → market sell-to-close
      3. Trail stop      : partial_taken AND pnl_pct ≤ 0 (breakeven) → sell-to-close
      4. Partial snapshot: pnl_pct ≥ +50%, not yet taken → record; raise stop to BE
      5. Time exit       : days_held ≥ 6               → market sell-to-close
    """
    state = _load_state()
    if not state:
        return

    today     = date.today()
    now_str   = datetime.now(ET).isoformat()
    to_remove = []

    for option_symbol, entry in list(state.items()):
        pos = get_position(option_symbol)

        if pos is None:
            to_remove.append(option_symbol)
            msg = (f'⚠️ {option_symbol} not found in Alpaca — removed from tracking '
                   f'(expired, closed manually, or order never filled)')
            if discord_fn:
                discord_fn(msg)
            _log({'timestamp': now_str, 'symbol': option_symbol,
                  'event': 'POSITION_NOT_FOUND', 'status': 'REMOVED'})
            continue

        pnl_pct       = float(pos['unrealized_plpc'])
        entry_premium = entry.get('entry_premium', float(pos['avg_entry_price']))
        entry_date    = date.fromisoformat(entry['entry_date'])
        days_held     = (today - entry_date).days
        partial_taken = entry.get('partial_taken', False)

        # ── Priority 1: Stop loss ─────────────────────────────────────────────
        if pnl_pct <= STOP_PCT:
            result = _exit_option(option_symbol, 'STOP', entry_premium, pos, discord_fn)
            if result == 'OK':
                to_remove.append(option_symbol)

        # ── Priority 2: Full target ───────────────────────────────────────────
        elif pnl_pct >= FULL_EXIT_PCT:
            result = _exit_option(option_symbol, 'FULL_TARGET', entry_premium, pos, discord_fn)
            if result == 'OK':
                to_remove.append(option_symbol)

        # ── Priority 3: Trail stop at breakeven (only after partial snapshot) ─
        elif partial_taken and pnl_pct <= BREAKEVEN_STOP:
            result = _exit_option(option_symbol, 'TRAIL_STOP', entry_premium, pos, discord_fn)
            if result == 'OK':
                to_remove.append(option_symbol)

        # ── Priority 4: Partial snapshot (+50%, first time only) ─────────────
        elif pnl_pct >= TAKE_HALF_PCT and not partial_taken:
            entry['partial_taken'] = True
            entry['realized_pct']  = round(pnl_pct, 4)
            state[option_symbol]   = entry

            msg = (f'🎯 PARTIAL TARGET +50% | {option_symbol} '
                   f'| Stop raised to breakeven | Holding for +100%')
            if discord_fn:
                discord_fn(msg)
            _log({'timestamp': now_str, 'symbol': option_symbol,
                  'event': 'PARTIAL_SNAPSHOT', 'pnl_pct': round(pnl_pct, 4),
                  'status': 'HOLDING'})

        # ── Priority 5: Day-6 time exit ───────────────────────────────────────
        elif days_held >= OPTIONS_MAX_HOLD:
            result = _exit_option(option_symbol, 'TIME_EXIT', entry_premium, pos, discord_fn)
            if result == 'OK':
                to_remove.append(option_symbol)

    for symbol in to_remove:
        state.pop(symbol, None)
    _save_state(state)


# ── STARTUP RECONCILIATION ────────────────────────────────────────────────────

def reconcile_positions_on_startup(discord_fn=None):
    """
    Re-adopt open Alpaca options positions not tracked in position_state.json.
    Guards against Railway ephemeral filesystem wiping state on redeploy.
    """
    try:
        alpaca_positions = get_all_positions()
        state            = _load_state()
        adopted          = []

        for pos in alpaca_positions:
            symbol = pos['symbol']
            if symbol in state:
                continue
            if pos.get('asset_class') != 'us_option':
                continue
            parsed = _parse_occ(symbol)
            if parsed is None:
                continue

            entry_premium = float(pos.get('avg_entry_price') or pos.get('current_price', 0))
            state[symbol] = {
                'underlying':                parsed['underlying'],
                'direction':                 'bullish' if parsed['option_type'] == 'call' else 'bearish',
                'option_type':               parsed['option_type'],
                'strike':                    parsed['strike'],
                'expiration':                parsed['expiration'],
                'entry_premium':             entry_premium,
                'contracts':                 1,
                'underlying_price_at_entry': None,
                'entry_date':                str(date.today()),
                'partial_taken':             False,
                'realized_pct':              None,
                'order_id':                  None,
                'reconciled':                True,
            }
            adopted.append(symbol)

        if adopted:
            _save_state(state)
            msg = (f'🔄 Startup reconciliation: {len(adopted)} orphaned option(s) adopted '
                   f'— {", ".join(adopted)}.\n'
                   f'entry_date reset to today; partial_taken=False. '
                   f'Verify if a partial snapshot was already recorded before restart.')
            if discord_fn:
                discord_fn(msg)
            print(f'[reconcile] Adopted: {adopted}')
        else:
            print(f'[reconcile] OK — {len(state)} tracked / {len(alpaca_positions)} open in Alpaca')

    except Exception as e:
        print(f'[reconcile] Error (scanner will start anyway): {e}')


# ── DAILY SUMMARY ─────────────────────────────────────────────────────────────

def daily_summary(discord_fn):
    """Post EOD paper account summary to Discord. Wired to 16:05 ET in scanner."""
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

        pnl_lines = ''
        if closed_today:
            pnl_lines = '\nClosed today:\n' + '\n'.join(
                f'  {p.get("symbol", "?")} {p.get("event", "?")} '
                f'{p.get("pnl_pct", 0) * 100:+.1f}% (${p.get("dollar_pnl", 0):+.2f})'
                for p in closed_today
            )

        open_positions = list(_load_state().keys())
        open_str = ', '.join(open_positions) if open_positions else 'none'

        msg = (
            f'📊 **Daily Summary — {today_str}**\n'
            f'Options entered today: {entered_today}\n'
            f'Options closed today: {len(closed_today)}{pnl_lines}\n'
            f'Open positions: {open_str}\n'
            f'Paper account equity: ${equity:,.2f}\n'
            f'Total return since start: {total_ret:+.2f}%'
        )
        discord_fn(msg)

    except Exception as e:
        discord_fn(f'⚠️ Daily summary error: {e}')


# ── CONNECTION TEST ────────────────────────────────────────────────────────────

def test_connection():
    """Verify paper API auth and account status. No orders placed."""
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
    print(f'   Options lvl : {acct.get("options_approved_level")}')
    print(f'   Opt BP      : ${float(acct.get("options_buying_power", 0)):,.2f}')

    print('\n2. Open positions …')
    positions = get_all_positions()
    if positions:
        for p in positions:
            print(f'   {p["symbol"]:<28} {p.get("asset_class","?"):<12} qty={p["qty"]}')
    else:
        print('   No open positions.')

    print('\n   ✓ Auth confirmed.')
    print('─' * 62)


if __name__ == '__main__':
    test_connection()
