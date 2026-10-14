"""Development-only multi-table tasks with independent Python evaluators."""
import random
import sqlite3

SCHEMA = """customers(id INTEGER PRIMARY KEY, region TEXT); orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, amount INTEGER, discount INTEGER NULL).
regions: north, south, west. Dates use YYYY-MM-DD. status: completed or cancelled (case-sensitive). amount and discount are rupees. A missing discount means zero where requested. customers.id joins orders.customer_id."""


def database(seed, mutation=None):
    rng = random.Random(seed)
    customers = [{'id': i, 'region': ['north', 'south', 'west'][(i - 1) % 3]} for i in range(1, 13)]
    orders = []
    for i in range(1, 73):
        orders.append({'id': i, 'customer_id': rng.randrange(1, 13),
                       'order_date': f'2026-{1 + (i % 2):02}-{1 + i % 28:02}',
                       'status': 'cancelled' if i % 5 == 0 else 'completed',
                       'amount': rng.randrange(5, 100) * 100,
                       'discount': None if i % 4 == 0 else rng.randrange(0, 5) * 100})
    if mutation == 'add_cancelled':
        for i in range(12):
            orders.append({'id': 1000 + i, 'customer_id': i + 1, 'order_date': '2026-01-15',
                           'status': 'cancelled', 'amount': 100000 + i * 1000, 'discount': 0})
    elif mutation == 'zero_missing_discount':
        orders = [{**r, 'discount': 0 if r['discount'] is None else r['discount']} for r in orders]
    elif mutation is not None:
        raise ValueError('unknown mutation')
    con = sqlite3.connect(':memory:')
    con.executescript('CREATE TABLE customers(id INTEGER PRIMARY KEY, region TEXT); CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, amount INTEGER, discount INTEGER);')
    con.executemany('INSERT INTO customers VALUES (:id,:region)', customers)
    con.executemany('INSERT INTO orders VALUES (:id,:customer_id,:order_date,:status,:amount,:discount)', orders)
    con.commit()
    return con, {'customers': customers, 'orders': orders}


def tasks(seeds=6):
    result = []
    for seed in range(seeds):
        con, data = database(seed)
        con.close()
        rows = data['orders']
        completed = [r for r in rows if r['status'] == 'completed']
        net = lambda r: r['amount'] - (r['discount'] or 0)
        regions = {c['id']: c['region'] for c in data['customers']}
        january = [r for r in completed if '2026-01-01' <= r['order_date'] < '2026-02-01']
        window = [r for r in completed if '2026-01-10' <= r['order_date'] < '2026-01-21']
        specs = [
            ('net_total', 'Sum amount minus discount for completed orders; missing discount means zero. Return {"net_total": number}.', {'net_total': sum(map(net, completed))}, True, True),
            ('january', 'Sum amount for completed orders on or after 2026-01-01 and before 2026-02-01. Return {"january": number}.', {'january': sum(r['amount'] for r in january)}, True, False),
            ('date_window', 'Count completed orders on or after 2026-01-10 and before 2026-01-21. Return {"count": number}.', {'count': len(window)}, True, False),
            ('region_total', 'Sum amount for completed orders by customer region. Return {"regions":{"north":number,"south":number,"west":number}}.', {'regions': {k: sum(r['amount'] for r in completed if regions[r['customer_id']] == k) for k in ['north','south','west']}}, True, False),
            ('region_net', 'Sum amount minus discount for completed orders by customer region; missing discount means zero. Return {"regions":{"north":number,"south":number,"west":number}}.', {'regions': {k: sum(net(r) for r in completed if regions[r['customer_id']] == k) for k in ['north','south','west']}}, True, True),
            ('distinct_customers', 'Count distinct customers with completed orders. Return {"customers":number}.', {'customers': len({r['customer_id'] for r in completed})}, True, False),
            ('average_net', 'Average amount minus discount across completed orders; missing discount means zero. Return {"average_net":number}.', {'average_net': sum(map(net,completed))/len(completed)}, True, True),
            ('missing_count', 'Count completed orders where discount is missing (SQL NULL). Return {"missing":number}.', {'missing': sum(r['discount'] is None for r in completed)}, True, False),
            ('completion_rate', 'Percentage of all orders whose status is completed, from 0 to 100. Return {"percentage":number}.', {'percentage':100*len(completed)/len(rows)}, False, False),
            ('discount_rate', 'Percentage of completed orders with a positive discount; missing discount means zero. Return {"percentage":number}.', {'percentage':100*sum((r['discount'] or 0)>0 for r in completed)/len(completed)}, True, True),
            ('large_net', 'Count completed orders with amount minus discount at least 5000; missing discount means zero. Return {"count":number}.', {'count':sum(net(r)>=5000 for r in completed)}, True, True),
            ('summary', 'Report both the count and total amount of completed orders in January 2026. Return {"count":number,"total":number}.', {'count':len(january),'total':sum(r['amount'] for r in january)}, True, False),
        ]
        for template, prompt, expected, excludes_cancelled, null_as_zero in specs:
            result.append({'id':f'rich-{template}-{seed}', 'template':template, 'seed':seed,
                           'prompt':prompt, 'expected':expected, 'schema':SCHEMA,
                           # Public structured contract: explicitly stated task semantics, shared by all policies.
                           'contract':{'excludes_cancelled':excludes_cancelled, 'null_discount_as_zero':null_as_zero}})
    return result
