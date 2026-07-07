#!/usr/bin/env python3
"""
credit_spread_strat.py — 7DTE SPY Put Credit Spread Strategy.

STANDALONE: no imports from scanner.py or paper_execution.py.
DORMANT until ACTIVE = True.

Alpaca API: raw requests only (no SDK; requirements.txt: requests==2.32.5).
  POST /v2/orders  order_class='mleg'  for multi-leg spread orders.

State files (all independent of scanner.py's files):
  credit_spread_positions.json — open positions + daily summary flag
  credit_spread_state.json     — weekly_realized_loss, cooldown_active, week_start_date
  credit_spread_trade_log.json — permanent trade history (never truncated)
  credit_spread_log.json       — rolling 500-entry scan log

Deploy as a second Railway worker:
  Add to Procfile:  credit_spread: python3 credit_spread_strat.py
"""

import json
import math
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import requests
import schedule
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm

try:
    import psycopg2
except ImportError:
    psycopg2 = None

load_dotenv()

# ── ACTIVATION FLAG ────────────────────────────────────────────────────────────
ACTIVE = True   # Set to True to enable live order placement

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK_URL')
ALPACA_KEY      = os.getenv('ALPACA_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET')
PAPER_BASE_URL  = 'https://paper-api.alpaca.markets'
DATA_URL        = 'https://data.alpaca.markets'
DATABASE_URL    = os.getenv('DATABASE_URL')
ET              = pytz.timezone('US/Eastern')
_DIR            = os.path.dirname(os.path.abspath(__file__))

# Strategy parameters
TARGET_DTE        = 7
DTE_MIN           = 6
DTE_MAX           = 8
TARGET_DELTA      = 0.20
DELTA_TOLERANCE   = 0.05     # reject if no strike within 0.05 of target delta
SPREAD_WIDTH      = 5.0
MIN_CREDIT        = 0.25
MAX_POSITIONS     = 2
TICKERS           = ['SPY', 'QQQ']   # underlyings traded; shared 2-position limit
PROFIT_TARGET_PCT = 0.50
STOP_LOSS_PCT     = 2.00
ORDER_FILL_TIMEOUT = 300
WEEKLY_LOSS_LIMIT = 1_000.0
RISK_FREE_RATE    = 0.045
VIX_IVR_WINDOW    = 252
MIN_IVR           = 30.0
MAX_VIX           = 35.0
SMA_PERIOD        = 20

# ── MACRO EVENT CALENDAR ───────────────────────────────────────────────────────
# Update these each January using official sources:
#   FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI:  bls.gov/schedule/news_release/cpi.htm
#   GDP:  bea.gov/news/schedule
# Jobs Report (NFP) is always the first Friday of each month — computed in code.

FOMC_DAYS = {
    # 2026 — both Day 1 and Day 2 of each meeting
    '2026-01-28', '2026-01-29',
    '2026-03-18', '2026-03-19',
    '2026-04-29', '2026-04-30',
    '2026-06-10', '2026-06-11',
    '2026-07-29', '2026-07-30',
    '2026-09-16', '2026-09-17',
    '2026-11-04', '2026-11-05',
    '2026-12-09', '2026-12-10',
}

CPI_DAYS = {
    # 2026 — BLS CPI release dates (prior month's data, ~12 days after month end)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-13', '2026-06-11', '2026-07-14', '2026-08-12',
    '2026-09-09', '2026-10-14', '2026-11-12', '2026-12-10',
}

GDP_DAYS = {
    # 2026 — BEA advance GDP estimates (~30 days after each quarter end)
    '2026-01-29',   # Q4 2025
    '2026-04-29',   # Q1 2026
    '2026-07-30',   # Q2 2026
    '2026-10-29',   # Q3 2026
}

# Timing
ENTRY_HOUR_START, ENTRY_MIN_START =  9, 45
ENTRY_HOUR_END,   ENTRY_MIN_END   = 15, 30
SUMMARY_HOUR,     SUMMARY_MIN     = 15, 35

# Files
POSITIONS_FILE = os.path.join(_DIR, 'credit_spread_positions.json')
WEEKLY_FILE    = os.path.join(_DIR, 'credit_spread_state.json')
TRADE_LOG_FILE = os.path.join(_DIR, 'credit_spread_trade_log.json')
SCAN_LOG_FILE  = os.path.join(_DIR, 'credit_spread_log.json')
LOG_MAX        = 500


# ── ALPACA HEADERS ─────────────────────────────────────────────────────────────

def _headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type':        'application/json',
    }


def _data_headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
    }


def _alpaca_get(url, **kwargs):
    """Alpaca GET with 429 backoff retry (max 3 attempts). Returns Response or None."""
    kwargs.setdefault('timeout', 10)
    for attempt in range(3):
        try:
            r = requests.get(url, **kwargs)
            if r.status_code == 429:
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Alpaca 429, retrying in {wait:.1f}s …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            print(f'Alpaca GET error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                time.sleep(1)
    return None


# ── DATABASE ───────────────────────────────────────────────────────────────────

_DB = None  # module-level psycopg2 connection
_DB_LAST_ALERT_AT = None  # datetime of the last "DB unreachable" alert; None means no active failure
DB_ALERT_REMINDER_INTERVAL = timedelta(hours=1)


def _get_db():
    """Return a live psycopg2 connection, or None if DATABASE_URL is not set."""
    global _DB, _DB_LAST_ALERT_AT
    if not DATABASE_URL or psycopg2 is None:
        return None
    try:
        if _DB is None or _DB.closed:
            _DB = psycopg2.connect(DATABASE_URL)
        if _DB_LAST_ALERT_AT is not None:
            _DB_LAST_ALERT_AT = None
            _discord('✅ DB reconnected — persistent storage active again')
        return _DB
    except Exception as e:
        print(f'[db] Connection failed: {e}')
        return None


def _alert_db_unreachable():
    """
    Fire the DB-unreachable Discord alert immediately on first failure, then
    re-fire at most once per DB_ALERT_REMINDER_INTERVAL for as long as the DB
    stays unreachable (e.g. on later _load_weekly() calls during run_scan()).
    Only called when DATABASE_URL is actually set — a deliberately DB-less
    setup stays silent.
    """
    global _DB_LAST_ALERT_AT
    now = datetime.now(ET)
    if _DB_LAST_ALERT_AT is not None and (now - _DB_LAST_ALERT_AT) < DB_ALERT_REMINDER_INTERVAL:
        return
    first_alert = _DB_LAST_ALERT_AT is None
    _DB_LAST_ALERT_AT = now
    _discord(
        ('🚨 DATABASE CONNECTION FAILED on startup — ' if first_alert else
         '🚨 DATABASE STILL UNREACHABLE (reminder) — ') +
        'DATABASE_URL is set but PostgreSQL is unreachable. Falling back to '
        'ephemeral JSON storage — any open positions will not persist across '
        'the next restart. '
        "Check Railway's Postgres service and the DATABASE_URL variable on "
        'this service immediately.'
    )


def _init_db():
    """Create tables if they don't exist. Logs connection on success."""
    conn = _get_db()
    if conn is None:
        if DATABASE_URL:
            reason = 'psycopg2 not installed' if psycopg2 is None else 'connection failed (see [db] Connection failed log above)'
            print(f'[db] WARNING — DATABASE_URL set but {reason}; falling back to JSON')
            _alert_db_unreachable()
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS credit_spread_positions (
                id             SERIAL PRIMARY KEY,
                short_symbol   TEXT NOT NULL,
                long_symbol    TEXT NOT NULL,
                short_strike   DOUBLE PRECISION,
                long_strike    DOUBLE PRECISION,
                expiration     TEXT,
                credit         DOUBLE PRECISION,
                max_risk       DOUBLE PRECISION,
                breakeven      DOUBLE PRECISION,
                profit_target  DOUBLE PRECISION,
                stop_loss_cost DOUBLE PRECISION,
                open_time      TEXT,
                entry_order_id TEXT,
                short_delta    DOUBLE PRECISION,
                spy_entry_px   DOUBLE PRECISION,
                reconciled     BOOLEAN DEFAULT FALSE,
                note           TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS credit_spread_state (
                id                   INTEGER PRIMARY KEY DEFAULT 1,
                weekly_realized_loss DOUBLE PRECISION DEFAULT 0.0,
                cooldown_active      BOOLEAN DEFAULT FALSE,
                week_start_date      TEXT,
                daily_summary_sent   TEXT
            )
        """)
        conn.commit()
        print('[db] Connected to PostgreSQL — persistent storage active')

        # One-time migration: seed from JSON files if tables are empty on first boot
        cur.execute('SELECT COUNT(*) FROM credit_spread_positions')
        if cur.fetchone()[0] == 0 and os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE) as f:
                    j = json.load(f)
                if isinstance(j, dict) and j.get('positions'):
                    _db_save_positions(j)
                    print(f'[db] Migrated {len(j["positions"])} position(s) from JSON')
            except Exception as me:
                print(f'[db] JSON positions migration failed: {me}')

        cur.execute('SELECT COUNT(*) FROM credit_spread_state')
        if cur.fetchone()[0] == 0 and os.path.exists(WEEKLY_FILE):
            try:
                with open(WEEKLY_FILE) as f:
                    j = json.load(f)
                if isinstance(j, dict) and j.get('week_start_date'):
                    _db_save_weekly(j)
                    print('[db] Migrated weekly state from JSON')
            except Exception as me:
                print(f'[db] JSON weekly migration failed: {me}')

    except Exception as e:
        print(f'[db] _init_db failed: {e}')


def _db_load_positions():
    """Load positions list and daily_summary_sent from PostgreSQL. Returns None on failure."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT short_symbol, long_symbol, short_strike, long_strike,
                   expiration, credit, max_risk, breakeven, profit_target,
                   stop_loss_cost, open_time, entry_order_id, short_delta,
                   spy_entry_px, reconciled, note
            FROM credit_spread_positions ORDER BY id
        """)
        positions = []
        for row in cur.fetchall():
            pos = {
                'short_symbol':   row[0],
                'long_symbol':    row[1],
                'short_strike':   row[2],
                'long_strike':    row[3],
                'expiration':     row[4],
                'credit':         row[5],
                'max_risk':       row[6],
                'breakeven':      row[7],
                'profit_target':  row[8],
                'stop_loss_cost': row[9],
                'open_time':      row[10],
                'entry_order_id': row[11],
                'short_delta':    row[12],
                'spy_entry_px':   row[13],
            }
            if row[14]:
                pos['reconciled'] = True
            if row[15]:
                pos['note'] = row[15]
            positions.append(pos)

        cur.execute('SELECT daily_summary_sent FROM credit_spread_state WHERE id = 1')
        row = cur.fetchone()
        return {'positions': positions, 'daily_summary_sent': row[0] if row else None}
    except Exception as e:
        print(f'[db] _db_load_positions failed: {e}')
        return None


def _db_save_positions(ps):
    """Replace all rows in credit_spread_positions and update daily_summary_sent."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute('DELETE FROM credit_spread_positions')
        for pos in ps.get('positions', []):
            cur.execute("""
                INSERT INTO credit_spread_positions (
                    short_symbol, long_symbol, short_strike, long_strike,
                    expiration, credit, max_risk, breakeven, profit_target,
                    stop_loss_cost, open_time, entry_order_id, short_delta,
                    spy_entry_px, reconciled, note
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                pos.get('short_symbol'),    pos.get('long_symbol'),
                pos.get('short_strike'),    pos.get('long_strike'),
                pos.get('expiration'),      pos.get('credit'),
                pos.get('max_risk'),        pos.get('breakeven'),
                pos.get('profit_target'),   pos.get('stop_loss_cost'),
                pos.get('open_time'),       pos.get('entry_order_id'),
                pos.get('short_delta'),     pos.get('spy_entry_px'),
                bool(pos.get('reconciled', False)),
                pos.get('note'),
            ))
        cur.execute("""
            INSERT INTO credit_spread_state (id, daily_summary_sent)
            VALUES (1, %s)
            ON CONFLICT (id) DO UPDATE SET daily_summary_sent = EXCLUDED.daily_summary_sent
        """, (ps.get('daily_summary_sent'),))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_save_positions failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def _db_load_weekly():
    """Load weekly state from PostgreSQL. Returns None on failure or no row."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT weekly_realized_loss, cooldown_active, week_start_date
            FROM credit_spread_state WHERE id = 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        return {
            'weekly_realized_loss': float(row[0] or 0.0),
            'cooldown_active':      bool(row[1]),
            'week_start_date':      row[2],
        }
    except Exception as e:
        print(f'[db] _db_load_weekly failed: {e}')
        return None


def _db_save_weekly(w):
    """Upsert weekly state in PostgreSQL."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO credit_spread_state
                (id, weekly_realized_loss, cooldown_active, week_start_date)
            VALUES (1, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                weekly_realized_loss = EXCLUDED.weekly_realized_loss,
                cooldown_active      = EXCLUDED.cooldown_active,
                week_start_date      = EXCLUDED.week_start_date
        """, (
            w.get('weekly_realized_loss', 0.0),
            bool(w.get('cooldown_active', False)),
            w.get('week_start_date'),
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_save_weekly failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        return False


# ── MARKET HOURS ───────────────────────────────────────────────────────────────

def is_market_hours():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def _in_entry_window():
    now  = datetime.now(ET)
    mins = now.hour * 60 + now.minute
    return (ENTRY_HOUR_START * 60 + ENTRY_MIN_START) <= mins <= (ENTRY_HOUR_END * 60 + ENTRY_MIN_END)


def _is_summary_time():
    now = datetime.now(ET)
    return now.hour == SUMMARY_HOUR and now.minute >= SUMMARY_MIN


# ── DISCORD ────────────────────────────────────────────────────────────────────

def _discord(msg):
    """Rate-limit-aware Discord post. ACTIVE-gated."""
    if not ACTIVE:
        return
    if not DISCORD_WEBHOOK:
        print(f'[Discord] {msg}')
        return
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK, json={'content': msg}, timeout=10)
            if r.status_code == 429:
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Discord 429, retrying in {wait:.1f}s …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return
        except Exception as e:
            print(f'Discord error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                time.sleep(1)


# ── LOGGING ────────────────────────────────────────────────────────────────────

def _log(entry):
    """Rolling 500-entry scan log. Always runs regardless of ACTIVE."""
    try:
        try:
            with open(SCAN_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        if len(log) > LOG_MAX:
            log = log[-LOG_MAX:]
        with open(SCAN_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log] write failed: {e}')


def _log_trade(entry):
    """Permanent trade history — never truncated."""
    try:
        try:
            with open(TRADE_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log_trade] write failed: {e}')


# ── POSITIONS STATE ────────────────────────────────────────────────────────────
# credit_spread_positions.json
# Schema: {"positions": [...], "daily_summary_sent": null | "YYYY-MM-DD"}

def _load_positions():
    if DATABASE_URL:
        result = _db_load_positions()
        if result is not None:
            return result
        print('[db] _load_positions: DB unavailable, falling back to JSON')
    try:
        with open(POSITIONS_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'positions' in d:
            return d
    except Exception:
        pass
    return {'positions': [], 'daily_summary_sent': None}


def _save_positions(ps):
    if DATABASE_URL:
        if _db_save_positions(ps):
            return
        print('[db] _save_positions: DB write failed, falling back to JSON')
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(ps, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_save_positions] failed: {e}')


# ── WEEKLY STATE ───────────────────────────────────────────────────────────────
# credit_spread_state.json — persists across Railway restarts independently.
# Schema: {"weekly_realized_loss": float, "cooldown_active": bool, "week_start_date": "YYYY-MM-DD"}
#
# weekly_realized_loss tracks net loss this week (wins reduce it, losses grow it).
# Cooldown triggers when weekly_realized_loss >= WEEKLY_LOSS_LIMIT ($1,000).
# Resets automatically every Monday.

def _this_monday():
    today = datetime.now(ET).date()
    return (today - timedelta(days=today.weekday())).isoformat()


def _empty_weekly():
    return {
        'weekly_realized_loss': 0.0,
        'cooldown_active':      False,
        'week_start_date':      _this_monday(),
    }


def _load_weekly():
    if DATABASE_URL:
        result = _db_load_weekly()
        if result is not None:
            return result
        print('[db] _load_weekly: DB unavailable, falling back to JSON')
        _alert_db_unreachable()
    try:
        with open(WEEKLY_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'week_start_date' in d:
            return d
    except Exception:
        pass
    return _empty_weekly()


def _save_weekly(w):
    if DATABASE_URL:
        if _db_save_weekly(w):
            return
        print('[db] _save_weekly: DB write failed, falling back to JSON')
    try:
        with open(WEEKLY_FILE, 'w') as f:
            json.dump(w, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_save_weekly] failed: {e}')


def _reset_weekly_if_needed(w):
    monday = _this_monday()
    if w.get('week_start_date') != monday:
        print(f'  [weekly reset] New week {monday} — loss and cooldown cleared')
        w = _empty_weekly()
        _save_weekly(w)
    return w


# ── FILE INIT ──────────────────────────────────────────────────────────────────

def _init_files():
    defaults = [
        (POSITIONS_FILE, {'positions': [], 'daily_summary_sent': None}),
        (WEEKLY_FILE,    _empty_weekly()),
        (TRADE_LOG_FILE, []),
        (SCAN_LOG_FILE,  []),
    ]
    for path, empty in defaults:
        if DATABASE_URL and path in (POSITIONS_FILE, WEEKLY_FILE):
            continue  # DB is the primary store for these two files
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(empty, f, indent=2)
            print(f'[init] Created {os.path.basename(path)}')


# ── STARTUP RECONCILIATION ─────────────────────────────────────────────────────

def _reconcile_on_startup():
    """
    Load state files, cross-check against Alpaca open options positions, and return (pos_state, weekly).

    Alpaca sync: queries /v2/positions for open SPY option legs.
    Each tracked spread = 2 legs. Mismatch → Discord alert.
    If Alpaca has MORE legs than the file expects, placeholder entries are added to
    block new entries until the user manually resolves untracked positions.
    If Alpaca has FEWER legs, we alert only (stale state entries).
    """
    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)
    now_str   = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')

    # ── Cross-check local state against Alpaca open options positions ──────────
    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is None:
            raise RuntimeError('positions fetch returned None after retries')
        option_legs = [
            p for p in r.json()
            if p.get('asset_class') == 'us_option'
            and any(str(p.get('symbol', '')).startswith(t) for t in TICKERS)
        ]
        expected_legs = len(pos_state.get('positions', [])) * 2
        actual_legs   = len(option_legs)

        if actual_legs == expected_legs:
            print(f'[reconcile] Alpaca options OK — '
                  f'{actual_legs} leg(s) / {len(pos_state["positions"])} spread(s)')
        else:
            orphaned = actual_legs - expected_legs
            direction = f'{orphaned} untracked leg(s) — new entries blocked' if orphaned > 0 \
                        else f'{-orphaned} extra state entry(ies) — stale entries possible'
            msg = (
                f'⚠️ OPTIONS MISMATCH on startup | '
                f'Alpaca: {actual_legs} option leg(s), '
                f'state file expects {expected_legs} ({len(pos_state["positions"])} spread(s)). '
                f'{direction}. Manual review required.'
            )
            print(f'[reconcile] {msg}')
            _log({'timestamp': now_str, 'event': 'RECONCILE_OPTIONS_MISMATCH',
                  'alpaca_legs': actual_legs, 'expected_legs': expected_legs,
                  'alpaca_symbols': [p['symbol'] for p in option_legs]})
            _discord(msg)

            # Add blocker placeholders for each untracked spread (2 legs = 1 spread)
            for _ in range(max(0, orphaned // 2)):
                pos_state['positions'].append({
                    'short_symbol':   'UNKNOWN',
                    'long_symbol':    'UNKNOWN',
                    'short_strike':   0.0,
                    'long_strike':    0.0,
                    'expiration':     '2099-01-01',  # far future — never triggers expiry close
                    'credit':         0.0,
                    'max_risk':       0.0,
                    'breakeven':      0.0,
                    'profit_target':  0.0,
                    'stop_loss_cost': 999.0,         # never triggers stop
                    'open_time':      now_str,
                    'entry_order_id': None,
                    'short_delta':    None,
                    'spy_entry_px':   None,
                    'reconciled':     True,
                    'note':           'Untracked Alpaca position — manual close required',
                })
            if orphaned > 0:
                _save_positions(pos_state)

    except Exception as e:
        print(f'[reconcile] Alpaca positions check failed (startup continues): {e}')
        _log({'timestamp': now_str, 'event': 'RECONCILE_API_ERROR', 'error': str(e)})

    # ── Log position state ─────────────────────────────────────────────────────
    positions = pos_state.get('positions', [])
    tracked   = [p for p in positions if not p.get('reconciled')]
    if positions:
        labels = [
            f'{p.get("short_strike", 0):.0f}/{p.get("long_strike", 0):.0f}P '
            f'exp={p.get("expiration", "?")}  credit=${p.get("credit", 0):.2f}'
            + (' [UNTRACKED]' if p.get('reconciled') else '')
            for p in positions
        ]
        print(f'[startup] Resumed {len(tracked)} tracked + '
              f'{len(positions) - len(tracked)} untracked position(s):')
        for lbl in labels:
            print(f'  {lbl}')
        _log({'timestamp': now_str, 'event': 'STARTUP_RESUME',
              'positions': len(positions), 'labels': labels})
    else:
        print('[startup] No open positions found — starting clean.')
        _log({'timestamp': now_str, 'event': 'STARTUP_CLEAN'})

    print(f'[startup] Weekly loss: ${weekly["weekly_realized_loss"]:.2f}  '
          f'cooldown: {weekly["cooldown_active"]}  '
          f'week_start: {weekly["week_start_date"]}')

    return pos_state, weekly


# ── MARKET DATA ────────────────────────────────────────────────────────────────

def _flatten_columns(df):
    """Handle MultiIndex columns returned by yfinance >=0.2 for single-ticker downloads."""
    if hasattr(df.columns, 'levels'):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass
    return df


def _spy_price():
    """Current SPY mid-price from Alpaca IEX feed."""
    try:
        r = _alpaca_get(
            f'{DATA_URL}/v2/stocks/SPY/quotes/latest',
            headers=_data_headers(),
            params={'feed': 'iex'},
        )
        if r is None:
            return None
        q   = r.json().get('quote', {})
        bid = float(q.get('bp') or 0)
        ask = float(q.get('ap') or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)
        r2 = _alpaca_get(
            f'{DATA_URL}/v2/stocks/SPY/trades/latest',
            headers=_data_headers(),
            params={'feed': 'iex'},
        )
        if r2 is None:
            return None
        return float(r2.json()['trade']['p'])
    except Exception as e:
        print(f'  [spy_price] {e}')
        return None


def _above_sma20(ticker):
    """Return (above_sma: bool|None, close: float|None, sma: float|None)."""
    try:
        df = yf.download(ticker, period='40d', interval='1d',
                         progress=False, auto_adjust=True, timeout=10)
        if df.empty:
            return None, None, None
        df     = _flatten_columns(df)
        closes = df['Close'].dropna().values
        if len(closes) < SMA_PERIOD:
            return None, None, None
        sma     = float(closes[-SMA_PERIOD:].mean())
        current = float(closes[-1])
        return current > sma, current, sma
    except Exception as e:
        print(f'  [sma20] {e}')
        return None, None, None


def _vix_ivrank():
    """Return (ivr_pct: float|None, vix: float|None). IVR = 252-day percentile."""
    try:
        df = yf.download('^VIX', period=f'{VIX_IVR_WINDOW + 60}d',
                         interval='1d', progress=False, auto_adjust=False, timeout=10)
        if df.empty:
            return None, None
        df          = _flatten_columns(df)
        closes      = df['Close'].dropna().values
        if len(closes) < 2:
            return None, None
        window      = closes[-VIX_IVR_WINDOW:] if len(closes) >= VIX_IVR_WINDOW else closes
        current_vix = float(closes[-1])
        ivr         = float((window < current_vix).sum()) / len(window) * 100
        return round(ivr, 1), round(current_vix, 2)
    except Exception as e:
        print(f'  [vix_ivrank] {e}')
        return None, None


# ── OPTIONS CHAIN ──────────────────────────────────────────────────────────────

def _find_target_expiration(ticker):
    """Return expiry closest to TARGET_DTE for ticker. Widens to 5–10 DTE if needed."""
    try:
        expirations = yf.Ticker(ticker).options
        if not expirations:
            return None
        today     = date.today()
        best      = None
        best_diff = float('inf')
        for exp_str in expirations:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                diff = abs(dte - TARGET_DTE)
                if diff < best_diff:
                    best, best_diff = exp, diff
        if best:
            return best
        for exp_str in expirations:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if 5 <= dte <= 10:
                diff = abs(dte - TARGET_DTE)
                if diff < best_diff:
                    best, best_diff = exp, diff
        return best
    except Exception as e:
        print(f'  [find_expiry] {e}')
        return None


def _fetch_options_chain(ticker, expiry, now_str):
    """
    Fetch put chain for ticker/expiry.
    Contracts: paper-api.alpaca.markets/v2/options/contracts (trading API)
    Snapshots: data.alpaca.markets/v1beta1/options/snapshots (market data API)
    Returns dict {symbol: {strike, bid, ask, mid, delta, iv}} or None on any failure.
    """
    try:
        r = _alpaca_get(
            f'{PAPER_BASE_URL}/v2/options/contracts',
            headers=_headers(),
            params={
                'underlying_symbols': ticker,
                'type':               'put',
                'expiration_date':    expiry.isoformat(),
                'limit':              200,
            },
            timeout=15,
        )
        if r is None:
            reason = 'contracts fetch returned None after retries'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR',
                  'expiry': str(expiry), 'reason': reason})
            return None
        contracts = r.json().get('option_contracts', [])
        if not contracts:
            reason = f'no contracts returned by Alpaca for expiry {expiry}'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry),
                  'reason': reason})
            return None

        symbols    = [c['symbol'] for c in contracts]
        strike_map = {c['symbol']: float(c['strike_price']) for c in contracts}

        chain = {}
        for batch_start in range(0, len(symbols), 100):
            batch = symbols[batch_start:batch_start + 100]
            try:
                rs = _alpaca_get(
                    f'{DATA_URL}/v1beta1/options/snapshots',
                    headers=_data_headers(),
                    params={'symbols': ','.join(batch), 'feed': 'indicative'},
                    timeout=15,
                )
                if rs is None:
                    reason = f'snapshots batch failed after retries (offset {batch_start})'
                    print(f'  [chain] {reason}')
                    _log({'timestamp': now_str, 'event': 'CHAIN_BATCH_ERROR', 'reason': reason})
                    continue
                for sym, snap in rs.json().get('snapshots', {}).items():
                    q   = snap.get('latestQuote', {})
                    bid = float(q.get('bp') or 0)
                    ask = float(q.get('ap') or 0)
                    mid = round((bid + ask) / 2, 4) if (bid + ask) > 0 else 0.0
                    g   = snap.get('greeks') or {}
                    chain[sym] = {
                        'strike': strike_map.get(sym, 0.0),
                        'bid':    bid,
                        'ask':    ask,
                        'mid':    mid,
                        'delta':  float(g['delta']) if g.get('delta') is not None else None,
                        'iv':     snap.get('impliedVolatility'),
                    }
            except Exception as e:
                reason = f'snapshot batch error: {type(e).__name__}: {e}'
                print(f'  [chain] {reason}')
                _log({'timestamp': now_str, 'event': 'CHAIN_BATCH_ERROR', 'reason': reason})

        if not chain:
            reason = 'all snapshot batches returned empty'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry),
                  'reason': reason})
            return None

        return chain

    except Exception as e:
        reason = f'{type(e).__name__}: {e}'
        print(f'  [chain] fetch failed: {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR', 'expiry': str(expiry),
              'reason': reason})
        return None


def _bs_put_delta(S, K, T_years, sigma):
    """Black-Scholes European put delta in [-1, 0]. Returns None on error."""
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return -1.0 if K > S else 0.0
        d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T_years) / (
            sigma * math.sqrt(T_years)
        )
        return norm.cdf(d1) - 1.0
    except Exception:
        return None


def _find_short_strike(chain, spy_px, expiry, vix, now_str):
    """
    Find OTM put with |delta| closest to TARGET_DELTA (0.20).

    Delta priority per contract:
      1. Alpaca greeks.delta  (most accurate)
      2. Per-contract impliedVolatility via Black-Scholes
      3. VIX / 100 via Black-Scholes  (last resort)

    Rejects result if best |delta| is > DELTA_TOLERANCE (0.05) from target.
    Returns (symbol, strike, delta) or (None, None, None).
    """
    T_years        = max((expiry - date.today()).days / 365.0, 1 / 365)
    sigma_fallback = (vix or 20.0) / 100.0

    best_sym, best_strike, best_delta = None, None, None
    best_diff = float('inf')
    bs_used = bs_failed = 0

    for sym, data in chain.items():
        strike = data['strike']
        if strike <= 0 or data['mid'] <= 0 or strike >= spy_px:
            continue  # skip zero-mid and ITM puts

        delta = data['delta']
        if delta is None:
            per_iv = data.get('iv')
            sigma  = float(per_iv) if per_iv else sigma_fallback
            delta  = _bs_put_delta(spy_px, strike, T_years, sigma)
            if delta is not None:
                bs_used += 1
            else:
                bs_failed += 1
                continue

        diff = abs(abs(delta) - TARGET_DELTA)
        if diff < best_diff:
            best_diff, best_sym, best_strike, best_delta = diff, sym, strike, delta

    if bs_used or bs_failed:
        print(f'  [chain] BS fallback: {bs_used} used, {bs_failed} failed')

    if best_sym is None:
        reason = 'no OTM puts with computable delta in chain'
        print(f'  [chain] {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_NO_DELTA', 'reason': reason})
        return None, None, None

    if best_diff > DELTA_TOLERANCE:
        reason = (f'closest delta={abs(best_delta):.3f} at ${best_strike:.0f} is '
                  f'{best_diff:.3f} from target {TARGET_DELTA} '
                  f'(tolerance {DELTA_TOLERANCE})')
        print(f'  [chain] {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_DELTA_TOO_FAR',
              'reason': reason, 'best_strike': best_strike,
              'best_delta': round(best_delta, 4)})
        return None, None, None

    return best_sym, best_strike, best_delta


def _find_long_symbol(chain, short_strike):
    """Find option symbol for the long leg (short_strike − SPREAD_WIDTH)."""
    target    = short_strike - SPREAD_WIDTH
    best_sym  = best_k = None
    best_diff = float('inf')
    for sym, data in chain.items():
        diff = abs(data['strike'] - target)
        if diff < best_diff:
            best_diff, best_sym, best_k = diff, sym, data['strike']
    return best_sym, best_k


def _spread_mid(chain, short_sym, long_sym):
    """Net credit = short_mid − long_mid. Returns None if data missing or degenerate."""
    try:
        s_mid = chain[short_sym]['mid']
        l_mid = chain[long_sym]['mid']
        if s_mid <= 0 or l_mid < 0:
            return None
        return round(s_mid - l_mid, 4)
    except (KeyError, TypeError):
        return None


# ── SPREAD VALUE FOR MONITORING ────────────────────────────────────────────────

def _current_cost_to_close(short_sym, long_sym):
    """
    Current debit to close = short_mid − long_mid.
    Positive = we pay to close (normal for a short spread). Returns None on error.
    """
    try:
        rs = _alpaca_get(
            f'{DATA_URL}/v1beta1/options/snapshots',
            headers=_data_headers(),
            params={'symbols': f'{short_sym},{long_sym}', 'feed': 'indicative'},
        )
        if rs is None:
            return None
        snaps = rs.json().get('snapshots', {})

        def _mid(sym):
            q   = snaps.get(sym, {}).get('latestQuote', {})
            bid = float(q.get('bp') or 0)
            ask = float(q.get('ap') or 0)
            return (bid + ask) / 2.0 if (bid + ask) > 0 else None

        s = _mid(short_sym)
        l = _mid(long_sym)
        if s is None or l is None:
            return None
        return round(s - l, 4)
    except Exception as e:
        print(f'  [cost_to_close] {e}')
        return None


# ── ORDER MANAGEMENT ───────────────────────────────────────────────────────────
# No Alpaca SDK. Uses Alpaca v2 REST API via requests==2.32.5.
# Multi-leg spread orders: POST /v2/orders with order_class='mleg'.
# Leg fields: symbol, side, ratio_qty, position_intent.

def _place_open_order(short_sym, long_sym, credit):
    """Limit order to open the credit spread. ACTIVE-gated."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          'limit',
            'time_in_force': 'day',
            'order_class':   'mleg',
            'limit_price':   str(round(credit, 2)),
            'legs': [
                {'symbol': short_sym, 'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_open'},
                {'symbol': long_sym,  'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_open'},
            ],
        }
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get('id')
    except Exception as e:
        print(f'  [place_open] {e}')
        return None


def _place_close_order(short_sym, long_sym, order_type='market', limit_price=None):
    """Order to close the spread. ACTIVE-gated. order_type: 'market' | 'limit'."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          order_type,
            'time_in_force': 'day',
            'order_class':   'mleg',
            'legs': [
                {'symbol': short_sym, 'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_close'},
                {'symbol': long_sym,  'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_close'},
            ],
        }
        if order_type == 'limit' and limit_price is not None:
            payload['limit_price'] = str(round(limit_price, 2))
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get('id')
    except Exception as e:
        print(f'  [place_close] {e}')
        return None


def _get_order(order_id):
    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/orders/{order_id}', headers=_headers())
        if r is None:
            return None
        return r.json()
    except Exception as e:
        print(f'  [get_order] {e}')
        return None


def _cancel_order(order_id):
    if not ACTIVE:
        return
    try:
        r = requests.delete(f'{PAPER_BASE_URL}/v2/orders/{order_id}',
                            headers=_headers(), timeout=10)
        if r.status_code not in (200, 204):
            print(f'  [cancel_order] status {r.status_code}')
    except Exception as e:
        print(f'  [cancel_order] {e}')


# ── ENTRY EXECUTION ────────────────────────────────────────────────────────────

def _attempt_entry(pos_state, weekly, now_str, ticker, short_sym, long_sym,
                   short_strike, long_strike, expiry, credit, stock_px, short_delta):
    """
    Place opening order, poll for fill up to ORDER_FILL_TIMEOUT seconds.
    Updates pos_state['positions'] in place on fill.
    Returns True if filled. ACTIVE-gated.
    """
    if not ACTIVE:
        return False

    order_id = _place_open_order(short_sym, long_sym, credit)
    if not order_id:
        _log({'timestamp': now_str, 'event': 'ORDER_PLACE_FAILED',
              'short': short_sym, 'long': long_sym, 'credit': credit})
        print('  [entry] order placement failed')
        return False

    print(f'  [entry] order {order_id} placed — polling fill (max {ORDER_FILL_TIMEOUT}s)…')
    deadline = time.time() + ORDER_FILL_TIMEOUT

    while time.time() < deadline:
        time.sleep(30)
        order = _get_order(order_id)
        if order is None:
            continue
        status = order.get('status', '')

        if status == 'filled':
            raw_price   = float(order.get('filled_avg_price') or credit)
            fill_credit = abs(raw_price)   # Alpaca returns negative for net credit received
            if raw_price < 0:
                print(f'  [entry] filled_avg_price was negative ({raw_price}) — using abs()')
            max_risk    = round((SPREAD_WIDTH - fill_credit) * 100, 2)
            breakeven   = round(short_strike - fill_credit, 2)
            pos = {
                'ticker':         ticker,
                'short_symbol':   short_sym,
                'long_symbol':    long_sym,
                'short_strike':   short_strike,
                'long_strike':    long_strike,
                'expiration':     expiry.isoformat(),
                'credit':         round(fill_credit, 4),
                'max_risk':       max_risk,
                'breakeven':      breakeven,
                'profit_target':  round(fill_credit * PROFIT_TARGET_PCT, 4),
                'stop_loss_cost': round(fill_credit * STOP_LOSS_PCT, 4),
                'open_time':      now_str,
                'entry_order_id': order_id,
                'short_delta':    round(short_delta, 4) if short_delta else None,
                'spy_entry_px':   stock_px,
            }
            pos_state['positions'].append(pos)
            _save_positions(pos_state)
            _log({'timestamp': now_str, 'event': 'ENTRY_FILLED', **pos})
            _log_trade({'timestamp': now_str, 'type': 'OPEN', **pos})
            _discord(
                f'🟢 CREDIT SPREAD OPEN | '
                f'{ticker} {short_strike:.0f}/{long_strike:.0f}P  exp {expiry} | '
                f'Credit: ${fill_credit:.2f} | Max risk: ${max_risk:.2f} | '
                f'Breakeven: ${breakeven:.2f} | '
                f'Positions open: {len(pos_state["positions"])}'
            )
            print(f'  [entry] FILLED ${fill_credit:.2f}  {short_sym} / {long_sym}')
            return True

        if status == 'partially_filled':
            print(f'  [entry] partial fill detected on multi-leg order — cancelling {order_id}')
            _log({'timestamp': now_str, 'event': 'ORDER_PARTIAL_FILL', 'order_id': order_id})
            _cancel_order(order_id)
            _discord(
                f'⚠️ Partial fill on spread entry order — cancelled. '
                f'Check Alpaca for any open legs that need manual closing.'
            )
            return False

        if status in ('cancelled', 'expired', 'rejected', 'done_for_day'):
            print(f'  [entry] order {status} — no fill')
            _log({'timestamp': now_str, 'event': f'ORDER_{status.upper()}',
                  'order_id': order_id})
            return False

    print(f'  [entry] fill timeout ({ORDER_FILL_TIMEOUT}s) — cancelling {order_id}')
    _cancel_order(order_id)
    _log({'timestamp': now_str, 'event': 'ENTRY_FILL_TIMEOUT', 'order_id': order_id})
    return False


# ── POSITION MONITORING ────────────────────────────────────────────────────────

def _record_exit(pos_state, weekly, pos, reason, close_cost, now_str):
    """
    Log exit, update weekly_realized_loss, fire Discord.
    Does NOT remove pos from pos_state — caller handles that and saves both files.

    weekly_realized_loss tracks net loss this week:
      loss trade  → loss grows  (pnl negative, -pnl positive)
      win  trade  → loss shrinks (pnl positive, -pnl negative), floored at 0
    """
    credit  = pos['credit']
    pnl     = round((credit - close_cost) * 100, 2)   # 100 multiplier per contract
    tkr     = pos.get('ticker', pos.get('short_symbol', 'SPY')[:3])
    label   = f'{tkr} {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'

    old_loss = weekly.get('weekly_realized_loss', 0.0)
    new_loss = round(max(0.0, old_loss - pnl), 2)
    weekly['weekly_realized_loss'] = new_loss

    exit_entry = {
        'timestamp':            now_str,
        'type':                 'CLOSE',
        'event':                'EXIT',
        'reason':               reason,
        'label':                label,
        'short_symbol':         pos['short_symbol'],
        'long_symbol':          pos['long_symbol'],
        'expiration':           pos['expiration'],
        'credit':               credit,
        'close_cost':           close_cost,
        'pnl':                  pnl,
        'weekly_realized_loss': new_loss,
    }
    _log(exit_entry)
    _log_trade(exit_entry)

    reason_labels = {
        'PROFIT_TARGET': 'profit target',
        'STOP_LOSS':     'stop loss',
        'EXPIRATION':    'expiration close',
    }
    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
    _discord(
        f'{"✅" if pnl >= 0 else "🔴"} SPREAD CLOSED '
        f'({reason_labels.get(reason, reason)}) | '
        f'{label} | '
        f'Credit: ${credit:.2f}  Close: ${close_cost:.4f} | '
        f'P&L: {pnl_str} | '
        f'Week loss: ${new_loss:.2f} / ${WEEKLY_LOSS_LIMIT:.0f}'
    )

    if new_loss >= WEEKLY_LOSS_LIMIT and not weekly.get('cooldown_active'):
        weekly['cooldown_active'] = True
        _discord(
            f'🚨 WEEKLY LOSS LIMIT HIT — ${new_loss:.2f} in net losses this week. '
            f'No new entries until Monday.'
        )
        _log({'timestamp': now_str, 'event': 'WEEKLY_LOSS_COOLDOWN',
              'weekly_realized_loss': new_loss})
        print(f'  [weekly limit] COOLDOWN activated — week loss ${new_loss:.2f}')


def _monitor_positions(pos_state, weekly, now_str):
    """
    Check all open positions for profit target / stop loss / expiration close.
    ACTIVE-gated. Modifies pos_state['positions'] in place; saves both files on any change.

    Isolation from paper_execution.py:
      - paper_execution.py manages equity positions via position_state.json.
      - Its reconcile_positions_on_startup() filters asset_class != 'us_equity' AND
        requires symbol in TICKERS (['SPY','QQQ','AAPL','NVDA','TSLA'] — raw equity symbols).
        OCC option symbols (e.g. 'SPY250703P00575000') can never pass that filter.
      - No state file is shared between the two systems.
    """
    if not ACTIVE:
        return

    positions = pos_state.get('positions', [])
    if not positions:
        return

    today    = datetime.now(ET).date()   # ET date — Railway container runs in UTC
    now_et   = datetime.now(ET)
    to_close = []

    for pos in positions:
        # Skip placeholder entries added by startup reconciliation for untracked Alpaca positions
        if pos.get('reconciled') and pos.get('short_symbol') == 'UNKNOWN':
            print(f'  [monitor] Skipping untracked reconciled position — manual review required')
            continue

        short_sym = pos['short_symbol']
        long_sym  = pos['long_symbol']
        expiry    = date.fromisoformat(pos['expiration'])
        credit    = pos['credit']
        tkr       = pos.get('ticker', pos.get('short_symbol', 'SPY')[:3])
        label     = f'{tkr} {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {expiry}'

        try:
            # ── Expiration force-close (9:45am ET on expiry day) ─────────────
            if today == expiry and now_et.hour == 9 and now_et.minute >= 45:
                print(f'  {label}: EXPIRATION CLOSE')
                cost = _current_cost_to_close(short_sym, long_sym) or 0.0
                ok   = _place_close_order(short_sym, long_sym, order_type='market')
                if ok:
                    _record_exit(pos_state, weekly, pos, 'EXPIRATION', cost, now_str)
                    to_close.append(pos)
                    _log({'timestamp': now_str, 'event': 'MONITOR_EXPIRY_CLOSE',
                          'label': label, 'close_cost': cost})
                else:
                    print(f'  {label}: EXPIRY CLOSE ORDER FAILED')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'EXPIRATION'})
                    _discord(
                        f'🚨 CLOSE ORDER FAILED | {label} | EXPIRATION DAY | '
                        f'Manual close required in Alpaca immediately.'
                    )
                continue

            # ── Current spread value ──────────────────────────────────────────
            cost = _current_cost_to_close(short_sym, long_sym)
            if cost is None:
                print(f'  {label}: value unavailable — skipping this cycle')
                _log({'timestamp': now_str, 'event': 'MONITOR_VALUE_UNAVAILABLE',
                      'label': label})
                continue

            print(f'  {label}: cost={cost:.4f}  credit={credit:.4f}  '
                  f'tgt≤{pos["profit_target"]:.4f}  stop≥{pos["stop_loss_cost"]:.4f}')

            # ── Profit target: cost ≤ 50% of original credit ─────────────────
            if cost <= pos['profit_target']:
                print(f'  {label}: PROFIT TARGET')
                ok = _place_close_order(short_sym, long_sym,
                                        order_type='limit',
                                        limit_price=pos['profit_target'])
                if ok:
                    _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', cost, now_str)
                    to_close.append(pos)
                else:
                    print(f'  {label}: profit-target close failed — retrying next cycle')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'PROFIT_TARGET'})
                    _discord(f'⚠️ Close order failed | {label} | Profit target | Retrying next scan')
                continue

            # ── Stop loss: cost ≥ 200% of original credit ────────────────────
            if cost >= pos['stop_loss_cost']:
                print(f'  {label}: STOP LOSS')
                ok = _place_close_order(short_sym, long_sym, order_type='market')
                if ok:
                    _record_exit(pos_state, weekly, pos, 'STOP_LOSS', cost, now_str)
                    to_close.append(pos)
                else:
                    print(f'  {label}: stop-loss close failed — retrying next cycle')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'STOP_LOSS'})
                    _discord(f'⚠️ Close order failed | {label} | Stop loss | Retrying next scan')

        except Exception as e:
            print(f'  {label}: MONITOR ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'MONITOR_POSITION_ERROR',
                  'label': label, 'error': f'{type(e).__name__}: {e}'})

    if to_close:
        for pos in to_close:
            try:
                pos_state['positions'].remove(pos)
            except ValueError:
                pass
        _save_positions(pos_state)
        _save_weekly(weekly)


# ── MACRO EVENT FILTER ─────────────────────────────────────────────────────────

def _macro_event_today():
    """
    Return an event name string if today (ET) is a blocked macro event day, else None.
    Checked before all other entry conditions — no API calls required.
    """
    today = datetime.now(ET).date()
    today_str = today.isoformat()

    if today_str in FOMC_DAYS:
        return 'FOMC meeting day'
    if today_str in CPI_DAYS:
        return 'CPI release day'
    if today_str in GDP_DAYS:
        return 'GDP release day'

    # Jobs Report (NFP) — always the first Friday of each month
    first          = today.replace(day=1)
    days_to_friday = (4 - first.weekday()) % 7   # Friday = weekday 4
    if today == first + timedelta(days=days_to_friday):
        return 'Jobs Report day (NFP)'

    return None


# ── ENTRY CONDITIONS ───────────────────────────────────────────────────────────

def _check_entry_conditions(ticker, pos_state, weekly):
    """
    Evaluate all gates cheapest-first. Short-circuits on first group failure.
    Returns (all_passed: bool, conditions: dict, stock_px, ivr, vix).
    """
    conds    = {}
    stock_px = ivr = vix = None

    def _c(name, passed, detail):
        conds[name] = {'passed': bool(passed), 'detail': str(detail)}

    # Macro event gate — day-wide block, no API call
    macro = _macro_event_today()
    _c('macro_event', macro is None, macro or 'none')
    if macro:
        return False, conds, stock_px, ivr, vix

    # Cheap gates (no API calls)
    _c('active',          ACTIVE,
       'True' if ACTIVE else 'False — DORMANT')
    _c('entry_window',    _in_entry_window(),
       '9:45–3:30 ET')
    _c('position_limit',  len(pos_state.get('positions', [])) < MAX_POSITIONS,
       f'{len(pos_state.get("positions", []))}/{MAX_POSITIONS} open')
    _c('weekly_cooldown', not weekly.get('cooldown_active', False),
       f'loss=${weekly.get("weekly_realized_loss", 0):.2f} / ${WEEKLY_LOSS_LIMIT:.0f}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, stock_px, ivr, vix

    # Underwater position gate — don't add a second spread if the first is at a loss
    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    if len(real_pos) == 1:
        p    = real_pos[0]
        cost = _current_cost_to_close(p['short_symbol'], p['long_symbol'])
        if cost is not None and cost > p.get('credit', 0):
            _c('underwater_block', False,
               f'cost={cost:.4f} > credit={p.get("credit", 0):.4f}')
            return False, conds, stock_px, ivr, vix

    # SMA filter
    above_sma, stock_close, sma_val = _above_sma20(ticker)
    stock_px = stock_close
    sma_key  = f'{ticker.lower()}_above_sma'
    if above_sma is None:
        _c(sma_key, False, 'data unavailable')
    else:
        _c(sma_key, above_sma,
           f'{ticker}={stock_close:.2f}  SMA20={sma_val:.2f}  '
           f'{"above" if above_sma else "BELOW"}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, stock_px, ivr, vix

    # IV rank + VIX cap
    ivr, vix = _vix_ivrank()
    if ivr is None:
        _c('iv_rank', False, 'VIX data unavailable')
        _c('vix_cap',  False, 'VIX data unavailable')
    else:
        _c('iv_rank', ivr >= MIN_IVR,
           f'IVR={ivr:.1f}% (need ≥{MIN_IVR}%) — VIX={vix:.2f}')
        _c('vix_cap',  vix < MAX_VIX,
           f'VIX={vix:.2f} {"<" if vix < MAX_VIX else "≥"} {MAX_VIX}')

    return all(v['passed'] for v in conds.values()), conds, stock_px, ivr, vix


# ── TICKER EVALUATION ──────────────────────────────────────────────────────────

def _evaluate_ticker(ticker, pos_state, weekly, now_str):
    """
    Full entry evaluation for one ticker.
    Returns (signal_dict | None, conditions_dict).
    signal_dict has keys: ticker, expiry, short_sym, long_sym, short_strike,
                          long_strike, short_delta, credit, stock_px, ivr, vix.
    """
    passed, conds, stock_px, ivr, vix = _check_entry_conditions(ticker, pos_state, weekly)
    if not passed:
        return None, conds

    expiry = _find_target_expiration(ticker)
    if expiry is None:
        conds['expiry_found'] = {'passed': False, 'detail': 'no valid expiration found'}
        return None, conds
    dte = (expiry - date.today()).days
    conds['expiry_found'] = {'passed': True, 'detail': f'{expiry} ({dte} DTE)'}

    chain = _fetch_options_chain(ticker, expiry, now_str)
    if chain is None:
        conds['chain_fetched'] = {'passed': False,
                                   'detail': 'chain unavailable (see CHAIN_* log event)'}
        return None, conds
    conds['chain_fetched'] = {'passed': True, 'detail': f'{len(chain)} contracts'}

    short_sym, short_strike, short_delta = _find_short_strike(
        chain, stock_px, expiry, vix, now_str
    )
    if short_sym is None:
        conds['short_strike'] = {'passed': False,
                                  'detail': 'no suitable 20Δ put (see CHAIN_* log)'}
        return None, conds
    conds['short_strike'] = {'passed': True,
                              'detail': f'${short_strike:.0f}  delta={short_delta:.3f}'}

    long_sym, long_strike = _find_long_symbol(chain, short_strike)
    if long_sym is None:
        conds['long_strike'] = {'passed': False, 'detail': 'not found in chain'}
        return None, conds
    conds['long_strike'] = {'passed': True, 'detail': f'${long_strike:.0f}'}

    credit = _spread_mid(chain, short_sym, long_sym)
    if credit is None or credit < MIN_CREDIT:
        conds['min_credit'] = {
            'passed': False,
            'detail': (f'credit=${credit:.4f} < ${MIN_CREDIT}'
                       if credit is not None else 'mid unavailable'),
        }
        return None, conds
    conds['min_credit'] = {'passed': True, 'detail': f'credit=${credit:.4f}'}

    return {
        'ticker':       ticker,
        'expiry':       expiry,
        'short_sym':    short_sym,
        'long_sym':     long_sym,
        'short_strike': short_strike,
        'long_strike':  long_strike,
        'short_delta':  short_delta,
        'credit':       credit,
        'stock_px':     stock_px,
        'ivr':          ivr,
        'vix':          vix,
    }, conds


PREMARKET_HOUR, PREMARKET_MINUTE = 9, 0   # fires once, 9:00-9:04am ET


def _is_premarket_time():
    now = datetime.now(ET)
    return now.hour == PREMARKET_HOUR and PREMARKET_MINUTE <= now.minute < PREMARKET_MINUTE + 5


def _check_premarket_report(pos_state, weekly, now_str):
    """Send once-daily parameter/status report before market open."""
    today_str = datetime.now(ET).date().isoformat()
    if pos_state.get('premarket_report_sent') == today_str:
        return
    if not _is_premarket_time():
        return

    pos_state['premarket_report_sent'] = today_str
    _save_positions(pos_state)

    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    macro = _macro_event_today()

    msg = (
        f'🌅 **Pre-Market Report — {today_str}**\n'
        f'Strategy: {", ".join(TICKERS)} put credit spreads, {TARGET_DTE} DTE\n'
        f'Target delta: {TARGET_DELTA} (±{DELTA_TOLERANCE})  |  Spread width: ${SPREAD_WIDTH}  |  Min credit: ${MIN_CREDIT}\n'
        f'Profit target: {int(PROFIT_TARGET_PCT*100)}%  |  Stop loss: {int(STOP_LOSS_PCT*100)}%\n'
        f'Max positions: {MAX_POSITIONS}  |  Currently open: {len(real_pos)}\n'
        f'IV rank min: {MIN_IVR}%  |  VIX cap: {MAX_VIX}\n'
        f'Weekly loss so far: ${weekly.get("weekly_realized_loss", 0):.2f} / ${WEEKLY_LOSS_LIMIT:.0f}\n'
        f'Cooldown active: {weekly.get("cooldown_active", False)}\n'
        f"Today's macro block: {macro or 'none — clear to trade'}"
    )
    _discord(msg)
    print(f'[premarket] Report sent for {today_str}')
# ── DAILY SUMMARY ──────────────────────────────────────────────────────────────

def _check_daily_summary(pos_state, weekly, now_str):
    """Send once-daily summary at 3:35pm ET. ACTIVE-gated."""
    if not ACTIVE:
        return
    today_str = datetime.now(ET).date().isoformat()
    if pos_state.get('daily_summary_sent') == today_str:
        return
    if not _is_summary_time():
        return

    pos_state['daily_summary_sent'] = today_str
    _save_positions(pos_state)

    positions  = pos_state.get('positions', [])
    real_pos   = [p for p in positions
                  if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    week_loss  = weekly.get('weekly_realized_loss', 0.0)

    # Trade log: count entries (OPEN) and exits (CLOSE) separately.
    # Note: trade log is on Railway ephemeral filesystem; counts are best-effort
    # if a container restart wiped the file mid-day.
    try:
        with open(TRADE_LOG_FILE) as f:
            tlog = json.load(f)
        today_opens  = [t for t in tlog
                        if t.get('type') == 'OPEN'
                        and str(t.get('timestamp', '')).startswith(today_str)]
        today_closes = [t for t in tlog
                        if t.get('type') == 'CLOSE'
                        and str(t.get('timestamp', '')).startswith(today_str)]
        realized_pnl  = sum(t.get('pnl', 0) for t in today_closes)
        entries_today = len(today_opens)
        exits_today   = len(today_closes)
    except Exception:
        realized_pnl = entries_today = exits_today = 0

    # Fetch live prices for tracked open positions to compute unrealized P&L.
    unrealized_pnl = 0.0
    pos_costs: dict = {}
    for pos in real_pos:
        ss = pos.get('short_symbol', '')
        ls = pos.get('long_symbol', '')
        if ss and ls:
            cost = _current_cost_to_close(ss, ls)
            pos_costs[(ss, ls)] = cost
            if cost is not None:
                unrealized_pnl += round((pos.get('credit', 0) - cost) * 100, 2)

    # Cross-check Alpaca for open SPY option legs — survives a wiped positions file.
    alpaca_note = ''
    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is not None:
            tracked_legs = [p for p in r.json()
                            if p.get('asset_class') == 'us_option'
                            and any(str(p.get('symbol', '')).startswith(t) for t in TICKERS)]
            alpaca_spreads = len(tracked_legs) // 2
            file_spreads   = len(real_pos)
            if alpaca_spreads != file_spreads:
                alpaca_note = (f' ⚠️ Alpaca shows {len(tracked_legs)} leg(s) '
                               f'({alpaca_spreads} spread(s)) — state file has {file_spreads}')
            else:
                alpaca_note = f' ✓ Alpaca confirms {len(tracked_legs)} leg(s)'
    except Exception:
        pass

    pos_detail = ''
    for pos in real_pos:
        ss, ls  = pos.get('short_symbol', ''), pos.get('long_symbol', '')
        cost    = pos_costs.get((ss, ls))
        unr_str = (f'  unreal ${round((pos.get("credit",0)-cost)*100,2):+.2f}'
                   if cost is not None else '')
        tkr     = pos.get('ticker', pos.get('short_symbol', 'SPY')[:3])
        pos_detail += (
            f'\n  [{tkr}] {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P '
            f'exp={pos["expiration"]}  credit=${pos["credit"]:.2f}{unr_str}'
        )

    _discord(
        f'📊 **Credit Spread Daily Summary**\n'
        f'Open positions: {len(real_pos)}{alpaca_note}{pos_detail}\n'
        f'Entries today: {entries_today}  |  Exits today: {exits_today}\n'
        f'Realized P&L: ${realized_pnl:+.2f}  |  Unrealized: ${unrealized_pnl:+.2f}\n'
        f'Week-to-date loss: ${week_loss:.2f} / ${WEEKLY_LOSS_LIMIT:.0f} limit'
    )


# ── MAIN SCAN ──────────────────────────────────────────────────────────────────

def run_scan():
    if not is_market_hours():
        print(f'[{datetime.now(ET).strftime("%H:%M ET")}] '
              f'Credit spread: outside market hours, skipping.')
        return

    now_str = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    print(f'\n[{now_str}] Credit spread scan  (ACTIVE={ACTIVE})…')

    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)

    # ── 1. Monitor open positions ──────────────────────────────────────────────
    try:
        _monitor_positions(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [monitor] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'MONITOR_ERROR',
              'error': f'{type(e).__name__}: {e}'})

   # ── Pre-market report ──────────────────────────────────────────────────────
    try:
        _check_premarket_report(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [premarket] ERROR — {type(e).__name__}: {e}') 
    # ── 2. Daily summary ───────────────────────────────────────────────────────
    try:
        _check_daily_summary(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [summary] ERROR — {type(e).__name__}: {e}')

    # ── 3. Evaluate each ticker for entry ─────────────────────────────────────
    ticker_signals = {}   # ticker → signal dict
    ticker_conds   = {}   # ticker → conditions dict (for logging)
    scan_result    = 'no_signal'
    chosen         = None

    for ticker in TICKERS:
        try:
            signal, conds = _evaluate_ticker(ticker, pos_state, weekly, now_str)
            ticker_conds[ticker] = conds
            if signal:
                ticker_signals[ticker] = signal
        except Exception as e:
            print(f'  [{ticker}] conditions ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'CONDITIONS_ERROR',
                  'ticker': ticker, 'error': f'{type(e).__name__}: {e}'})

    # ── 4. Pick winner (higher credit) and execute ─────────────────────────────
    if ticker_signals:
        chosen = max(ticker_signals.values(), key=lambda s: s['credit'])

    if chosen:
        t = chosen['ticker']
        if not ACTIVE:
            scan_result = 'dormant_would_enter'
            print(
                f'  DORMANT MODE — entry skipped | '
                f'{t} {chosen["short_strike"]:.0f}/{chosen["long_strike"]:.0f}P  '
                f'exp={chosen["expiry"]}  credit=${chosen["credit"]:.4f}  '
                f'delta={chosen["short_delta"]:.3f}'
            )
        else:
            print(f'  ENTERING: {t} {chosen["short_strike"]:.0f}/{chosen["long_strike"]:.0f}P  '
                  f'exp={chosen["expiry"]}  credit=${chosen["credit"]:.4f}')
            filled = _attempt_entry(
                pos_state, weekly, now_str,
                t, chosen['short_sym'], chosen['long_sym'],
                chosen['short_strike'], chosen['long_strike'],
                chosen['expiry'], chosen['credit'],
                chosen['stock_px'], chosen['short_delta'],
            )
            scan_result = 'entry_filled' if filled else 'entry_not_filled'
    else:
        # Macro event is day-wide — print once, not per ticker
        macro_detail = next(
            (c['macro_event']['detail']
             for c in ticker_conds.values()
             if not c.get('macro_event', {}).get('passed', True)),
            None,
        )
        if macro_detail:
            print(f'  No entry — macro event day: {macro_detail}')
        else:
            for t, conds in ticker_conds.items():
                fails = [k for k, v in conds.items() if not v.get('passed')]
                if 'underwater_block' in fails:
                    print(f'  [{t}] No entry — blocked: existing position underwater')
                elif fails:
                    print(f'  [{t}] No entry — failed: {", ".join(fails)}')

    # ── 5. Log scan ────────────────────────────────────────────────────────────
    log_entry = {
        'timestamp':      now_str,
        'event':          'SCAN',
        'active':         ACTIVE,
        'scan_result':    scan_result,
        'open_positions': len(pos_state.get('positions', [])),
        'weekly_loss':    weekly.get('weekly_realized_loss', 0.0),
        'cooldown':       weekly.get('cooldown_active', False),
        'conditions':     ticker_conds,
    }
    if scan_result in ('dormant_would_enter', 'entry_filled', 'entry_not_filled') and chosen:
        log_entry['signal'] = {
            'ticker':       chosen['ticker'],
            'short_symbol': chosen['short_sym'],
            'long_symbol':  chosen['long_sym'],
            'expiry':       str(chosen['expiry']),
            'credit':       chosen['credit'],
            'short_delta':  round(chosen['short_delta'], 4) if chosen['short_delta'] else None,
            'stock_px':     chosen['stock_px'],
        }
    _log(log_entry)


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print('  SPY & QQQ PUT CREDIT SPREADS  |  7 DTE  |  20Δ / $5-wide')
    print(f'  ACTIVE = {ACTIVE}')
    if not ACTIVE:
        print('  *** DORMANT — scanning and logging, NO orders placed ***')
    print('  Alpaca v2 REST API  |  raw requests  |  no SDK')
    print('  Scan every 5 min, 9:30–16:00 ET, Mon–Fri')
    print('=' * 62)

    _init_db()
    _init_files()
    _reconcile_on_startup()

    schedule.every(5).minutes.do(run_scan)

    if ACTIVE:
        _discord(
            f'✅ Credit Spread system live | SPY & QQQ 7DTE put spreads | '
            f'20Δ short / $5-wide / ${MIN_CREDIT:.2f} min credit | '
            f'2 positions max combined'
        )
    else:
        print('  Dormant mode: conditions evaluated and logged each scan.')

    run_scan()

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f'[scheduler] ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
                  'event': 'SCHEDULER_ERROR', 'error': f'{type(e).__name__}: {e}'})
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f'[FATAL] {type(e).__name__}: {e}')
        traceback.print_exc()
        raise
