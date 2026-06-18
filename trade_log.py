#!/usr/bin/env python3
"""
Trade logging utility: log, summarize, and export options trades.
"""

import csv
import json
from datetime import date, datetime
from typing import List, Optional

TRADE_LOG = '/Users/noahrourke/trading-system/trade_log.json'
TRADE_CSV = '/Users/noahrourke/trading-system/trade_log.csv'


def _load() -> List[dict]:
    try:
        with open(TRADE_LOG) as f:
            return json.load(f)
    except Exception:
        return []


def _save(trades: List[dict]):
    with open(TRADE_LOG, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def log_trade(
    ticker: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    result: str,
    exit_type: str,
    days_held: int,
    score: int,
    proxy_iv_rank: float,
    sigma: float,
    entry_date: Optional[str] = None,
    flags: Optional[list] = None,
):
    """Append a completed trade to trade_log.json."""
    trades = _load()
    trades.append({
        'date':          entry_date or str(date.today()),
        'ticker':        ticker,
        'direction':     direction,
        'entry_price':   round(entry_price, 2),
        'exit_price':    round(exit_price, 2),
        'pnl_pct':       round(pnl_pct, 2),
        'result':        result,
        'exit_type':     exit_type,
        'days_held':     days_held,
        'score':         score,
        'proxy_iv_rank': round(proxy_iv_rank, 1),
        'sigma':         round(sigma, 4),
        'flags':         flags or [],
        'logged_at':     datetime.now().isoformat(),
    })
    _save(trades)


def summary():
    """Print win rate, avg P&L, streak, total return, best/worst trade."""
    trades = _load()
    if not trades:
        print('No trades logged yet.')
        return

    pnls   = [t['pnl_pct'] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Current consecutive-loss streak (from most recent backward)
    streak = 0
    for t in reversed(trades):
        if t.get('result') == 'LOSS':
            streak += 1
        else:
            break

    # Approx total return (assuming 10% of portfolio risked per trade)
    total_return = sum(p * 0.10 for p in pnls)

    best  = max(trades, key=lambda t: t['pnl_pct'])
    worst = min(trades, key=lambda t: t['pnl_pct'])

    print('═' * 52)
    print('  TRADE LOG SUMMARY')
    print('═' * 52)
    print(f'  Total trades:         {len(trades)}')
    print(f'  Win rate:             {len(wins)/len(trades):.1%}  ({len(wins)}W / {len(losses)}L)')
    if wins:
        print(f'  Avg winner:           +{sum(wins)/len(wins):.1f}%')
    if losses:
        print(f'  Avg loser:            {sum(losses)/len(losses):.1f}%')
    print(f'  Current loss streak:  {streak}')
    print(f'  Approx total return:  {total_return:+.1f}%')
    print(f'  Best trade:   {best["ticker"]}  {best["date"]}  +{best["pnl_pct"]:.1f}%  ({best["exit_type"]})')
    print(f'  Worst trade:  {worst["ticker"]}  {worst["date"]}  {worst["pnl_pct"]:.1f}%  ({worst["exit_type"]})')
    print('═' * 52)


def today_losses() -> int:
    """Return count of losses entered today — used for scanner cooldown."""
    trades   = _load()
    today    = str(date.today())
    count    = 0
    for t in reversed([x for x in trades if x.get('date') == today]):
        if t.get('result') == 'LOSS':
            count += 1
        else:
            break
    return count


def export_csv():
    """Write all trades to trade_log.csv."""
    trades = _load()
    if not trades:
        print('No trades to export.')
        return
    with open(TRADE_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)
    print(f'Exported {len(trades)} trades → {TRADE_CSV}')


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    {'summary': summary, 'csv': export_csv, 'losses': lambda: print(today_losses())}.get(cmd, summary)()
