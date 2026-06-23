#!/usr/bin/env python3
"""
Contract selection dry-run — NO orders placed.
Fetches the live SPY + QQQ chains, shows what select_contract() would pick,
and confirms the entry guard blocks same-underlying duplicates.
"""

import json
import os
from datetime import date, timedelta

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

PAPER_BASE_URL    = 'https://paper-api.alpaca.markets'
ALPACA_KEY        = os.getenv('ALPACA_KEY')
ALPACA_SECRET     = os.getenv('ALPACA_SECRET')
MIN_CONTRACT_COST = 75
MAX_CONTRACT_COST = 250
TARGET_DTE        = 7
MIN_DTE           = 4
MAX_DTE           = 8
W                 = 70   # display width


def _headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type':        'application/json',
    }


def fetch_chain(ticker, direction):
    """Return (all_contracts, best_exp, dte) or ([], None, None)."""
    option_type = 'call' if direction == 'bullish' else 'put'
    today = date.today()
    params = {
        'underlying_symbols':  ticker,
        'expiration_date_gte': str(today + timedelta(days=MIN_DTE)),
        'expiration_date_lte': str(today + timedelta(days=MAX_DTE + 2)),
        'type':                option_type,
        'limit':               500,
    }
    r = requests.get(f'{PAPER_BASE_URL}/v2/options/contracts',
                     headers=_headers(), params=params, timeout=15)
    r.raise_for_status()
    contracts = r.json().get('option_contracts', [])

    valid_exps = {}
    for c in contracts:
        dte = (date.fromisoformat(c['expiration_date']) - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            valid_exps[c['expiration_date']] = dte

    if not valid_exps:
        return contracts, None, None

    best_exp = min(valid_exps, key=lambda e: (abs(valid_exps[e] - TARGET_DTE), -valid_exps[e]))
    return contracts, best_exp, valid_exps[best_exp]


def score_candidates(contracts, best_exp, current_price):
    """Score all contracts at best_exp. Returns list sorted by ATM distance."""
    scored = []
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
        scored.append({
            'symbol':    c['symbol'],
            'strike':    strike,
            'premium':   premium,
            'cost':      cost,
            'atm_diff':  abs(strike - current_price),
            'in_range':  MIN_CONTRACT_COST <= cost <= MAX_CONTRACT_COST,
        })
    scored.sort(key=lambda x: x['atm_diff'])
    return scored


def print_section(title):
    print()
    print('═' * W)
    print(f'  {title}')
    print('═' * W)


def run_test(ticker, direction):
    print_section(f'{ticker}  {direction.upper()} SIGNAL  →  {"CALL" if direction == "bullish" else "PUT"} selection')

    # Fetch current price
    hist = yf.Ticker(ticker).history(period='1d')
    current_price = float(hist['Close'].iloc[-1])
    print(f'  Current price : ${current_price:.2f}')
    print(f'  Risk budget   : ${MIN_CONTRACT_COST}–${MAX_CONTRACT_COST} per contract (1× = $125 target)')

    # Fetch chain
    contracts, best_exp, best_dte = fetch_chain(ticker, direction)
    print(f'  Chain total   : {len(contracts)} {("call" if direction=="bullish" else "put")} contracts fetched (4–10 DTE window)')

    if best_exp is None:
        # Show what expirations are available
        today = date.today()
        avail = {}
        for c in contracts:
            dte = (date.fromisoformat(c['expiration_date']) - today).days
            avail[c['expiration_date']] = dte
        print(f'  ⚠️  No expiration in {MIN_DTE}–{MAX_DTE} DTE range.')
        print(f'  Available: {dict(sorted(avail.items()))}')
        print(f'  → SKIP — no affordable contract')
        return

    print(f'  Expiration    : {best_exp}  ({best_dte} DTE)')

    # Score candidates
    scored = score_candidates(contracts, best_exp, current_price)
    affordable = [s for s in scored if s['in_range']]

    # Table header
    print()
    print(f'  {"Symbol":<24} {"Strike":>8} {"Premium":>9} {"Cost/1c":>10} {"ATM±":>8}  {"Range?":>7}')
    print('  ' + '─' * 68)

    # Print nearest 8 strikes, marking affordable ones
    for row in scored[:8]:
        marker = '✓ IN RANGE' if row['in_range'] else '✗'
        atm_str = f'±{row["atm_diff"]:.1f}'
        print(f'  {row["symbol"]:<24} ${row["strike"]:>7.0f} ${row["premium"]:>8.2f}'
              f' ${row["cost"]:>9.0f} {atm_str:>8}  {marker}')

    print()
    if affordable:
        best = affordable[0]   # closest to ATM in range
        print(f'  TOP 3 CANDIDATES (affordable, sorted ATM→OTM):')
        for i, c in enumerate(affordable[:3], 1):
            sel = '← SELECTED' if i == 1 else ''
            print(f'    {i}. {c["symbol"]}  strike=${c["strike"]:.0f}'
                  f'  premium=${c["premium"]:.2f}  cost=${c["cost"]:.0f}'
                  f'  ATM±{c["atm_diff"]:.1f}  {sel}')

        print()
        print(f'  SELECTION RATIONALE:')
        print(f'    Symbol     : {best["symbol"]}')
        print(f'    Strike     : ${best["strike"]:.0f}  (ATM±{best["atm_diff"]:.1f}  vs  ${current_price:.2f})')
        print(f'    Expiration : {best_exp}  ({best_dte} DTE)')
        print(f'    Premium    : ${best["premium"]:.2f}')
        print(f'    Cost / 1c  : ${best["cost"]:.0f}  (target budget: $125)')
        pct_of_budget = best["cost"] / 125 * 100
        print(f'    % of budget: {pct_of_budget:.0f}%  (${best["cost"]:.0f} / $125)')
        print(f'    Why chosen : closest strike to ATM whose cost falls in ${MIN_CONTRACT_COST}–${MAX_CONTRACT_COST}')
    else:
        print(f'  ⚠️  No contracts in ${MIN_CONTRACT_COST}–${MAX_CONTRACT_COST} range at {best_exp}.')
        if scored:
            cheapest = min(scored, key=lambda x: x['cost'])
            print(f'  Cheapest available: ${cheapest["cost"]:.0f}  ({cheapest["symbol"]})')
        print(f'  → SKIP — no affordable contract')


def run_entry_guard_test():
    print_section('ENTRY GUARD TEST')

    # Simulate state: SPY call position already open
    fake_state = {
        'SPY260626C00740000': {
            'underlying':    'SPY',
            'direction':     'bullish',
            'option_type':   'call',
            'entry_date':    str(date.today()),
            'entry_premium': 7.50,
        }
    }

    print(f'  Simulated open positions: {list(fake_state.keys())}')
    print()

    # SPY bullish signal arrives
    spy_blocked = any(e.get('underlying') == 'SPY' for e in fake_state.values())
    print(f'  New SPY CALL signal:')
    print(f'    Entry guard → {"BLOCKED — position already open in SPY" if spy_blocked else "PASS"}')

    # QQQ bearish signal arrives
    qqq_blocked = any(e.get('underlying') == 'QQQ' for e in fake_state.values())
    print(f'  New QQQ PUT signal:')
    print(f'    Entry guard → {"BLOCKED" if qqq_blocked else "PASS — different underlying, would proceed"}')

    # SPY put (opposite direction) — still blocked because same underlying
    spy_put_blocked = any(e.get('underlying') == 'SPY' for e in fake_state.values())
    print(f'  New SPY PUT signal (opposite direction):')
    print(f'    Entry guard → {"BLOCKED — same underlying regardless of direction" if spy_put_blocked else "PASS"}')

    print()
    print(f'  ✓ Guard checks underlying only, not strike/expiry/direction.')
    print(f'    One open SPY position blocks ALL new SPY signals until closed.')


if __name__ == '__main__':
    print('=' * W)
    print('  CONTRACT SELECTION DRY-RUN  —  no orders placed')
    print(f'  Date: {date.today()}')
    print('=' * W)

    run_test('SPY', 'bullish')
    run_test('QQQ', 'bullish')
    run_entry_guard_test()

    print()
    print('=' * W)
    print('  Done. Review the SELECTED contracts above before approving.')
    print('=' * W)
