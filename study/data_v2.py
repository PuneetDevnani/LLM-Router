"""Exploratory v2 corpus with record-based oracles and untouched v1 templates."""
from collections import defaultdict
from datetime import date, timedelta
import random
import sqlite3

from pilot.richer import SCHEMA as ORDERS_SCHEMA, tasks as old_tasks

MONTH_START='2026-01-10'
MONTH_END='2026-02-20'
HAVING_COUNT=24

GROUPS={
    'train':['net_total','january','region_total','region_net','average_net','missing_count','distinct_customers','large_net'],
    'validation':['date_window','completion_rate','discount_rate','summary'],
    'test':['top_regions','cancelled_ratio','customers_no_orders','discount_by_region','month_totals','having_regions','max_per_customer','weighted_net'],
    'schema_shift':['stock_value','low_stock_skus','top_stock_products','east_stock_share'],
}
INVENTORY_SCHEMA='products(id INTEGER PRIMARY KEY, sku TEXT, unit_cost INTEGER, reorder_point INTEGER); inventory(product_id INTEGER, warehouse TEXT, quantity INTEGER). warehouses: east, central, west. unit_cost is rupees per unit. quantity is nonnegative units. products.id joins inventory.product_id.'
NEW_SPECS={
    'top_regions':('Return the two customer regions with the most completed orders, breaking count ties alphabetically. Return {"regions": {region: count}}.','regions','count',True),
    'cancelled_ratio':('Return the percentage of all orders that are cancelled, from 0 to 100, as {"percentage": number}.','percentage','ratio',False),
    'customers_no_orders':('Count customers with no orders of any status, including customers absent from orders. Return {"customers": number}.','customers','count',False),
    'discount_by_region':('Sum discounts on completed orders by customer region; missing discount means zero. Return {"regions": {region: number}}.','regions','sum',True),
    'month_totals':(f'Sum amount for completed orders on or after {MONTH_START} and before {MONTH_END}, grouped by YYYY-MM month. Return {{"months": {{month: number}}}}.','months','sum',True),
    'having_regions':(f'Return regions having at least {HAVING_COUNT} completed orders, with their completed-order counts. Return {{"regions": {{region: count}}}}.','regions','count',True),
    'max_per_customer':('Return the maximum completed-order amount for each customer with a completed order. Customer IDs must be string labels. Return {"customers": {customer_id: number}}.','customers','maximum',True),
    'weighted_net':('Return the amount-weighted mean net value of completed orders. Net value is amount minus discount, with missing discount zero; weight is amount. Return {"weighted_net": number}.','weighted_net','weighted_average',False),
    'stock_value':('Sum quantity times unit_cost by warehouse across all inventory. Return {"warehouses": {warehouse: number}}.','warehouses','sum',True),
    'low_stock_skus':('Count distinct products whose quantity in at least one warehouse is below their reorder_point. Return {"products": number}.','products','count',False),
    'top_stock_products':('Return the three products with greatest total stock value across warehouses, breaking ties by sku alphabetically. Stock value is quantity times unit_cost. Return {"products": {sku: number}}.','products','sum',True),
    'east_stock_share':('Return the percentage of all inventory units held in the east warehouse, from 0 to 100. Return {"percentage": number}.','percentage','ratio',False),
}
REFERENCE_SQL={
    'net_total':"SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'",
    'january':"SELECT SUM(amount) FROM orders WHERE status='completed' AND order_date>='2026-01-01' AND order_date<'2026-02-01'",
    'date_window':"SELECT COUNT(*) FROM orders WHERE status='completed' AND order_date>='2026-01-10' AND order_date<'2026-01-21'",
    'region_total':"SELECT c.region,SUM(o.amount) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region",
    'region_net':"SELECT c.region,SUM(o.amount-COALESCE(o.discount,0)) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region",
    'distinct_customers':"SELECT COUNT(DISTINCT customer_id) FROM orders WHERE status='completed'",
    'average_net':"SELECT AVG(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'",
    'missing_count':"SELECT COUNT(*) FROM orders WHERE status='completed' AND discount IS NULL",
    'completion_rate':"SELECT 100.0*SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)/COUNT(*) FROM orders",
    'discount_rate':"SELECT 100.0*SUM(CASE WHEN COALESCE(discount,0)>0 THEN 1 ELSE 0 END)/COUNT(*) FROM orders WHERE status='completed'",
    'large_net':"SELECT COUNT(*) FROM orders WHERE status='completed' AND amount-COALESCE(discount,0)>=5000",
    'summary':"SELECT COUNT(*),SUM(amount) FROM orders WHERE status='completed' AND order_date>='2026-01-01' AND order_date<'2026-02-01'",
    'top_regions':"SELECT c.region,COUNT(*) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region ORDER BY COUNT(*) DESC,c.region LIMIT 2",
    'cancelled_ratio':"SELECT 100.0*SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END)/COUNT(*) FROM orders",
    'customers_no_orders':"SELECT COUNT(*) FROM customers c LEFT JOIN orders o ON o.customer_id=c.id WHERE o.id IS NULL",
    'discount_by_region':"SELECT c.region,SUM(COALESCE(o.discount,0)) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region",
    'month_totals':f"SELECT substr(order_date,1,7),SUM(amount) FROM orders WHERE status='completed' AND order_date>='{MONTH_START}' AND order_date<'{MONTH_END}' GROUP BY substr(order_date,1,7)",
    'having_regions':f"SELECT c.region,COUNT(*) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region HAVING COUNT(*)>={HAVING_COUNT}",
    'max_per_customer':"SELECT CAST(customer_id AS TEXT),MAX(amount) FROM orders WHERE status='completed' GROUP BY customer_id",
    'weighted_net':"SELECT 1.0*SUM(amount*(amount-COALESCE(discount,0)))/SUM(amount) FROM orders WHERE status='completed'",
    'stock_value':"SELECT i.warehouse,SUM(i.quantity*p.unit_cost) FROM inventory i JOIN products p ON p.id=i.product_id GROUP BY i.warehouse",
    'low_stock_skus':"SELECT COUNT(DISTINCT i.product_id) FROM inventory i JOIN products p ON p.id=i.product_id WHERE i.quantity<p.reorder_point",
    'top_stock_products':"SELECT p.sku,SUM(i.quantity*p.unit_cost) FROM inventory i JOIN products p ON p.id=i.product_id GROUP BY p.sku ORDER BY SUM(i.quantity*p.unit_cost) DESC,p.sku LIMIT 3",
    'east_stock_share':"SELECT 100.0*SUM(CASE WHEN warehouse='east' THEN quantity ELSE 0 END)/SUM(quantity) FROM inventory",
}


def order_records(seed):
    """Seed changes statuses, null positions, dates and customer participation."""
    rng=random.Random(seed)
    customers=[{'id':i,'region':['north','south','west'][(i-1)%3]} for i in range(1,19)]
    active=[rng.choice([c['id'] for c in customers if c['region']==r]) for r in ('north','south','west')]
    active+=rng.sample([c['id'] for c in customers if c['id'] not in active],rng.randrange(6,14))
    orders=[]
    for i in range(1,73):
        orders.append({'id':i,'customer_id':active[i-1] if i<=3 else rng.choice(active),
            'order_date':(date(2026,1,1)+timedelta(days=rng.randrange(59))).isoformat(),
            'status':'completed' if i<=3 or rng.random()<.76 else 'cancelled',
            'amount':rng.randrange(5,151)*100,'discount':None if rng.random()<.24 else rng.randrange(6)*100})
    return {'customers':customers,'orders':orders}


def inventory_records(seed):
    rng=random.Random(10000+seed)
    products=[{'id':i,'sku':f'SKU{i:02}','unit_cost':rng.randrange(5,101)*10,'reorder_point':rng.randrange(10,41)} for i in range(1,13)]
    inventory=[{'product_id':p['id'],'warehouse':w,'quantity':rng.randrange(1 if p['id']==1 else 0,121)}
               for p in products for w in ('east','central','west')]
    return {'products':products,'inventory':inventory}


def database(seed,mutation=None,schema='orders'):
    """Create a private in-memory environment for the selected public schema."""
    if schema not in ('orders','inventory'): raise ValueError('Unknown v2 schema')
    data=order_records(seed) if schema=='orders' else inventory_records(seed)
    if mutation=='add_cancelled' and schema=='orders':
        data['orders'] += [{'id':1000+i,'customer_id':i+1,'order_date':'2026-01-15','status':'cancelled','amount':100000+i*1000,'discount':0} for i in range(12)]
    elif mutation=='zero_missing_discount' and schema=='orders':
        data['orders']=[{**r,'discount':0 if r['discount'] is None else r['discount']} for r in data['orders']]
    elif mutation is not None: raise ValueError('Unsupported v2 mutation/schema')
    con=sqlite3.connect(':memory:')
    if schema=='orders':
        con.executescript('CREATE TABLE customers(id INTEGER PRIMARY KEY,region TEXT); CREATE TABLE orders(id INTEGER PRIMARY KEY,customer_id INTEGER,order_date TEXT,status TEXT,amount INTEGER,discount INTEGER);')
        con.executemany('INSERT INTO customers VALUES (:id,:region)',data['customers'])
        con.executemany('INSERT INTO orders VALUES (:id,:customer_id,:order_date,:status,:amount,:discount)',data['orders'])
    else:
        con.executescript('CREATE TABLE products(id INTEGER PRIMARY KEY,sku TEXT,unit_cost INTEGER,reorder_point INTEGER); CREATE TABLE inventory(product_id INTEGER,warehouse TEXT,quantity INTEGER);')
        con.executemany('INSERT INTO products VALUES (:id,:sku,:unit_cost,:reorder_point)',data['products'])
        con.executemany('INSERT INTO inventory VALUES (:product_id,:warehouse,:quantity)',data['inventory'])
    con.commit();return con,data


def oracle(template,data):
    """Independent Python record calculations; no SQL or live tool outputs."""
    if 'orders' not in data:
        products={p['id']:p for p in data['products']};rows=data['inventory']
        if template=='stock_value':
            values=defaultdict(int)
            for r in rows: values[r['warehouse']]+=r['quantity']*products[r['product_id']]['unit_cost']
            return {'warehouses':dict(values)}
        if template=='low_stock_skus': return {'products':len({r['product_id'] for r in rows if r['quantity']<products[r['product_id']]['reorder_point']})}
        if template=='top_stock_products':
            values=defaultdict(int)
            for r in rows: values[products[r['product_id']]['sku']]+=r['quantity']*products[r['product_id']]['unit_cost']
            return {'products':dict(sorted(values.items(),key=lambda kv:(-kv[1],kv[0]))[:3])}
        if template=='east_stock_share': return {'percentage':100*sum(r['quantity'] for r in rows if r['warehouse']=='east')/sum(r['quantity'] for r in rows)}
        raise ValueError('Unknown inventory template')
    rows=data['orders'];completed=[r for r in rows if r['status']=='completed']
    regions={c['id']:c['region'] for c in data['customers']};net=lambda r:r['amount']-(r['discount'] or 0)
    january=[r for r in completed if '2026-01-01'<=r['order_date']<'2026-02-01']
    if template=='net_total': return {'net_total':sum(net(r) for r in completed)}
    if template=='january': return {'january':sum(r['amount'] for r in january)}
    if template=='date_window': return {'count':sum('2026-01-10'<=r['order_date']<'2026-01-21' for r in completed)}
    if template=='distinct_customers': return {'customers':len({r['customer_id'] for r in completed})}
    if template=='average_net': return {'average_net':sum(net(r) for r in completed)/len(completed)}
    if template=='missing_count': return {'missing':sum(r['discount'] is None for r in completed)}
    if template=='completion_rate': return {'percentage':100*len(completed)/len(rows)}
    if template=='discount_rate': return {'percentage':100*sum((r['discount'] or 0)>0 for r in completed)/len(completed)}
    if template=='large_net': return {'count':sum(net(r)>=5000 for r in completed)}
    if template=='summary': return {'count':len(january),'total':sum(r['amount'] for r in january)}
    if template=='cancelled_ratio': return {'percentage':100*sum(r['status']=='cancelled' for r in rows)/len(rows)}
    if template=='customers_no_orders': return {'customers':len({c['id'] for c in data['customers']}-{r['customer_id'] for r in rows})}
    if template=='weighted_net': return {'weighted_net':sum(r['amount']*net(r) for r in completed)/sum(r['amount'] for r in completed)}
    values=defaultdict(int)
    if template in ('region_total','region_net','discount_by_region','top_regions','having_regions'):
        for r in completed:
            value=r['amount'] if template=='region_total' else net(r) if template=='region_net' else (r['discount'] or 0) if template=='discount_by_region' else 1
            values[regions[r['customer_id']]]+=value
        if template=='top_regions': values=dict(sorted(values.items(),key=lambda kv:(-kv[1],kv[0]))[:2])
        if template=='having_regions': values={k:v for k,v in values.items() if v>=HAVING_COUNT}
        return {'regions':dict(values)}
    if template=='month_totals':
        for r in completed:
            if MONTH_START<=r['order_date']<MONTH_END: values[r['order_date'][:7]]+=r['amount']
        return {'months':dict(values)}
    if template=='max_per_customer':
        for r in completed: values[str(r['customer_id'])]=max(values[str(r['customer_id'])],r['amount'])
        return {'customers':dict(values)}
    raise ValueError('Unknown orders template')


def corpus():
    """480 instances: 8 old train, 4 old validation, 8 fresh test, 4 shift."""
    base={t['template']:t for t in old_tasks(seeds=1)};result=[]
    for split,templates in GROUPS.items():
        for template in templates:
            if template in base:
                b=base[template];prompt=b['prompt'];contract=dict(b['contract'])
                keys=list(b['expected']);grouped=template in ('region_total','region_net')
                aggregation='average' if template=='average_net' else 'ratio' if template in ('completion_rate','discount_rate') else 'count' if template in ('date_window','distinct_customers','missing_count','large_net') else 'sum'
            else:
                prompt,key,aggregation,grouped=NEW_SPECS[template];keys=[key]
                contract={'excludes_cancelled':template in ('top_regions','discount_by_region','month_totals','having_regions','max_per_customer','weighted_net'),
                          'null_discount_as_zero':template in ('discount_by_region','weighted_net')}
            contract.update(report_keys=keys,grouped=grouped,aggregation=aggregation,distinct=template in ('distinct_customers','low_stock_skus'),
                            net_value=template in ('net_total','average_net','region_net','large_net','weighted_net'),
                            # Human-authored permission to change discount values;
                            # NULL positions are preserved. Supplied to every policy.
                            discount_irrelevant=template in ('january','date_window','region_total','distinct_customers',
                                'completion_rate','summary','top_regions','cancelled_ratio','customers_no_orders',
                                'month_totals','having_regions','max_per_customer'))
            # Human-authored sign constraint, shared by every policy. No gold
            # value or dataset-derived bound enters this optional field.
            if aggregation=='sum' or contract['net_value']:
                contract['nonnegative']=True
            if template in ('january','summary'): contract.update(date_start='2026-01-01',date_end_exclusive='2026-02-01')
            if template=='date_window': contract.update(date_start='2026-01-10',date_end_exclusive='2026-01-21')
            if template=='month_totals': contract.update(date_start=MONTH_START,date_end_exclusive=MONTH_END)
            if template=='large_net': contract['monetary_threshold']=5000
            if template=='having_regions': contract['count_threshold']=HAVING_COUNT
            if aggregation=='ratio':
                contract['valid_range']=[0,100]
                contract['denominator']='completed orders' if template=='discount_rate' else 'all inventory units' if template=='east_stock_share' else 'all orders'
            schema='inventory' if split=='schema_shift' else 'orders'
            for seed in range(10):
                records=inventory_records(seed) if schema=='inventory' else order_records(seed)
                for wording in range(2):
                    result.append({'id':f'v2-{template}-{seed}-w{wording}','template':template,'split':split,'seed':seed,
                        'corpus_version':'v2','schema_variant':schema,'schema':INVENTORY_SCHEMA if schema=='inventory' else ORDERS_SCHEMA,
                        'prompt':prompt if wording==0 else 'Analyse the supplied tables. '+prompt+' Use exactly the requested output keys.',
                        'contract':dict(contract),'expected':oracle(template,records)})
    return result


def equivalent_sql(template):
    return 'SELECT * FROM ('+REFERENCE_SQL[template]+') AS equivalent_result'


def wrong_sql(task):
    if task['contract']['grouped']: return "SELECT '__incorrect__',987654321"
    return 'SELECT '+','.join('987654321' for _ in task['contract']['report_keys'])
