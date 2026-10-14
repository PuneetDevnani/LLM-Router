"""Deterministic pilot tasks and a bounded, read-only SQL tool."""
import math
import random
import sqlite3


def database(seed):
    rng = random.Random(seed)
    con = sqlite3.connect(':memory:')
    con.execute('CREATE TABLE orders (id INTEGER, month TEXT, category TEXT, status TEXT, revenue INTEGER)')
    rows = [(i, f'2026-{1 + i % 3:02}', ['books', 'electronics', 'clothing'][rng.randrange(3)],
             'cancelled' if i % 7 == 0 else 'completed', rng.randrange(5, 80) * 100)
            for i in range(1, 61)]
    con.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', rows)
    con.commit()
    return con, rows


def tasks():
    """24 pilot instances, four templates; never split variants independently."""
    result = []
    for seed in range(6):
        con, rows = database(seed)
        con.close()
        valid = [r for r in rows if r[3] == 'completed']
        specs = [
            ('total', 'Report total revenue from completed orders as {"total": number}.',
             {'total': sum(r[4] for r in valid)}),
            ('count', 'Report number of completed orders as {"count": number}.',
             {'count': len(valid)}),
            ('monthly', 'Report completed-order revenue for each month as {"monthly": {"2026-01": number, "2026-02": number, "2026-03": number}}.',
             {'monthly': {f'2026-{m:02}': sum(r[4] for r in valid if r[1] == f'2026-{m:02}') for m in range(1, 4)}}),
            ('category', 'Report completed-order revenue by category as {"category": {"books": number, "electronics": number, "clothing": number}}.',
             {'category': {c: sum(r[4] for r in valid if r[2] == c) for c in ['books', 'electronics', 'clothing']}}),
        ]
        for template, prompt, expected in specs:
            result.append({'id': f'{template}-{seed}', 'template': template,
                           'seed': seed, 'prompt': prompt, 'expected': expected})
    return result


def query(con, sql):
    if not isinstance(sql, str) or len(sql) > 4000:
        raise ValueError('sql must be a string of at most 4000 characters')
    # SQLite authorization protects against write/attach operations even through WITH.
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    safe_functions = {'sum', 'count', 'avg', 'min', 'max', 'round', 'abs', 'coalesce',
                      'ifnull', 'nullif', 'lower', 'upper', 'length', 'substr', 'strftime', 'date'}
    def authorize(action, arg1, arg2, *_):
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or '').lower() not in safe_functions:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    con.set_authorizer(authorize)
    steps = [0]
    def interrupt():
        steps[0] += 1
        return int(steps[0] > 1000)
    con.set_progress_handler(interrupt, 1000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(101)
        if len(rows) > 100:
            raise ValueError('result exceeds 100 rows; aggregate or limit your query')
        return {'columns': [d[0] for d in cur.description], 'rows': rows}
    finally:
        con.set_authorizer(None)
        con.set_progress_handler(None, 0)


def correct(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(correct(actual[k], v) for k, v in expected.items())
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
