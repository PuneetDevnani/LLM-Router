"""Exploratory contract probes; reference answers never enter this module."""
import re
from datetime import date, timedelta
from functools import partial
import math
from pilot.checks import equivalent, verify
from pilot.environment import query


def scale_eligible(sql,output,contract):
    """Homogeneity applies only to unthresholded monetary sums and means."""
    return (contract['aggregation'] in ('sum','average','weighted_average')
            and not contract.get('monetary_threshold') and not contract.get('count_threshold')
            and not re.search(r'\bHAVING\b|(?:>=|<=|>|<|=)\s*\d',sql,re.I)
            and (contract.get('grouped') or len(output['columns'])==1))


def transformed_probe(evidence,name,sql,seed,database_factory,transform,expected):
    con,_=database_factory(seed)
    try:
        transform(con);con.commit();observed=query(con,sql)
        evidence['evidence'].append({'check':name,'status':'consistent' if equivalent(expected,observed) else 'violation',
                                    'transformed_result':observed})
    except Exception as exc:
        evidence['evidence'].append({'check':name,'status':'inconclusive','detail':str(exc)})
    finally: con.close()


def scale_amounts(con):
    con.execute('UPDATE orders SET amount=2*amount, discount=2*discount')


def duplicate_orders(con):
    offset=con.execute('SELECT COALESCE(MAX(id),0)+1 FROM orders').fetchone()[0]
    con.execute('INSERT INTO orders SELECT id+?,customer_id,order_date,status,amount,discount FROM orders',(offset,))


def insert_probe_orders(con,records):
    """Private intervention using existing customer identity, never gold data."""
    offset=con.execute('SELECT COALESCE(MAX(id),0)+1 FROM orders').fetchone()[0]
    customer=con.execute('SELECT id FROM customers ORDER BY id LIMIT 1').fetchone()[0]
    for i,(day,status,amount,discount) in enumerate(records):
        con.execute('INSERT INTO orders VALUES (?,?,?,?,?,?)',(offset+i,customer,day,status,amount,discount))


def change_discount_values(con):
    """Preserve missingness: only the magnitude of observed discounts changes."""
    con.execute('UPDATE orders SET discount=discount+137 WHERE discount IS NOT NULL')


def nonnegative_range_probe(output,contract):
    """Optional public sign constraint on numeric values, excluding group labels.

    Empty/NULL aggregates supply no sign evidence. Passing is not a proof.
    """
    if contract.get('nonnegative') is not True:
        return None
    values=[v for row in output['rows'] for i,v in enumerate(row)
            if not (contract.get('grouped') and i==0)
            and isinstance(v,(int,float)) and not isinstance(v,bool)]
    return {'check':'public_nonnegative_range',
            'status':'violation' if any(not math.isfinite(v) or v<0 for v in values)
                     else 'consistent' if values else 'inconclusive',
            'numeric_values_checked':len(values)}


def contract_boundary_probes(evidence,sql,output,seed,contract,database_factory):
    """Probe date exclusion and the public, ungrouped net-count boundary."""
    start,end=contract.get('date_start'),contract.get('date_end_exclusive')
    if start and end:
        outside=[((date.fromisoformat(start)-timedelta(days=1)).isoformat(),'completed',100003,0),
                 (end,'completed',100019,0)]
        transformed_probe(evidence,'date_boundary_records',sql,seed,database_factory,
                          partial(insert_probe_orders,records=outside),output)
    threshold=contract.get('monetary_threshold')
    if (contract.get('net_value') is True and contract['aggregation']=='count' and not contract.get('grouped')
            and isinstance(threshold,(int,float)) and math.isfinite(threshold)
            and len(output['rows'])==1 and len(output['rows'][0])==1
            and isinstance(output['rows'][0][0],(int,float))):
        expected={'columns':output['columns'],'rows':[[output['rows'][0][0]+1]]}
        records=[(start or '2026-01-15','completed',threshold-1,0),
                 (start or '2026-01-15','completed',threshold,0)]
        transformed_probe(evidence,'net_count_boundary_records',sql,seed,database_factory,
                          partial(insert_probe_orders,records=records),expected)


def contract_verify_v3(sql,output,seed,contract,database_factory):
    """Development-designed probes; no gold or template identity is available."""
    evidence=contract_verify(sql,output,seed,contract,database_factory)
    sign=nonnegative_range_probe(output,contract)
    if sign is not None: evidence['evidence'].append(sign)
    evidence['concerns']=[c for c in evidence['concerns'] if not c.startswith('requested count')]
    contract_boundary_probes(evidence,sql,output,seed,contract,database_factory)
    if contract.get('excludes_cancelled'):
        rows=[(contract.get('date_start') or '2026-01-15','cancelled',100033,None)]
        transformed_probe(evidence,'cancelled_null_discount',sql,seed,database_factory,
                          partial(insert_probe_orders,records=rows),output)
    if contract.get('discount_irrelevant') is True:
        transformed_probe(evidence,'discount_irrelevance',sql,seed,database_factory,change_discount_values,output)
    if scale_eligible(sql,output,contract):
        expected={'columns':output['columns'],'rows':[]}
        for row in output['rows']:
            expected['rows'].append([v if contract.get('grouped') and i==0 else 2*v if isinstance(v,(int,float)) else v
                                     for i,v in enumerate(row)])
        transformed_probe(evidence,'monetary_scale',sql,seed,database_factory,scale_amounts,expected)
    if contract['aggregation']=='average' and not contract.get('monetary_threshold') and not re.search(r'\bHAVING\b',sql,re.I):
        transformed_probe(evidence,'duplicate_average_population',sql,seed,database_factory,duplicate_orders,output)
    limits=contract.get('valid_range')
    if contract['aggregation']=='ratio' and limits:
        values=[v for row in output['rows'] for v in row if isinstance(v,(int,float))]
        evidence['evidence'].append({'check':'public_ratio_range','status':'violation' if any(v<limits[0] or v>limits[1] for v in values) else 'consistent'})
    evidence['flagged']=bool(evidence['concerns']) or any(e['status']=='violation' for e in evidence['evidence'])
    evidence['version']='contract_v3'
    evidence['interpretation']='Fallible contract probes; passing and consistency do not establish correctness.'
    return evidence


def contract_verify(sql,output,seed,contract,database_factory):
    evidence=verify(sql,output,seed,contract,database_factory)
    # Explicit public contract permission; missing does not imply net value.
    # Weighted means are excluded because this transform also changes weights.
    if (contract.get('net_value') is True and contract.get('null_discount_as_zero')
            and contract['aggregation'] in ('sum','average','count')):
        con,_=database_factory(seed)
        try:
            con.execute('UPDATE orders SET amount=amount+137, discount=COALESCE(discount,0)+137')
            con.commit()
            transformed=query(con,sql)
            evidence['evidence'].append({'check':'preserve_net_value',
                'status':'consistent' if equivalent(output,transformed) else 'violation',
                'transformed_result':transformed})
        except Exception as exc:
            evidence['evidence'].append({'check':'preserve_net_value','status':'inconclusive','detail':str(exc)})
        finally:
            con.close()
    concerns=[]
    if contract['aggregation']=='count' and not re.search(r'\bCOUNT\s*\(',sql,re.I):
        concerns.append('requested count not visibly aggregated; alternative expressions may be valid')
    if contract.get('null_discount_as_zero') and re.search(r'\bdiscount\s+IS\s+NOT\s+NULL\b',sql,re.I):
        concerns.append('query excludes missing discounts despite the zero-null convention; inspect predicate context')
    evidence['concerns']=concerns
    evidence['flagged']=bool(concerns) or any(e['status']=='violation' for e in evidence['evidence'])
    evidence['version']='contract_v2'
    evidence['interpretation']='Exploratory corpus-specific checks; passing is not a correctness guarantee.'
    return evidence
