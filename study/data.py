"""Frozen grouped development/validation/test data. No paid data source."""
from pilot.richer import tasks as base_tasks, database, SCHEMA

GROUPS = {
    'train': ['net_total','date_window','region_total','average_net','missing_count','completion_rate'],
    'validation': ['distinct_customers','discount_rate','summary'],
    'test': ['january','region_net','large_net'],
}
FORMATS = {
    'net_total':['net_total'], 'january':['january'], 'date_window':['count'],
    'region_total':['regions'], 'region_net':['regions'], 'distinct_customers':['customers'],
    'average_net':['average_net'], 'missing_count':['missing'], 'completion_rate':['percentage'],
    'discount_rate':['percentage'], 'large_net':['count'], 'summary':['count','total'],
}


def corpus():
    result=[]
    for task in base_tasks(seeds=10):
        group=task['template']
        split=next(k for k,v in GROUPS.items() if group in v)
        contract={**task['contract'], 'report_keys':FORMATS[group],
                  'grouped':group in ('region_total','region_net')}
        # Constraints are human-authored structured user input, never inferred from gold answers.
        contract['aggregation'] = ('average' if group=='average_net' else 'ratio' if group in ('completion_rate','discount_rate')
                                   else 'count' if group in ('date_window','distinct_customers','missing_count','large_net') else 'sum')
        if group in ('january','summary'):
            contract['date_start']='2026-01-01'; contract['date_end_exclusive']='2026-02-01'
        if group=='date_window':
            contract['date_start']='2026-01-10'; contract['date_end_exclusive']='2026-01-21'
        contract['distinct']=group=='distinct_customers'
        for wording in range(2):
            result.append({**task,'id':f'{group}-{task["seed"]}-w{wording}', 'split':split,
                           'prompt':task['prompt'] if wording==0 else 'Please analyse the supplied database. '+task['prompt']+' Give only the specified result.',
                           'contract':contract})
    return result


def public(task):
    return {k:task[k] for k in ('prompt','schema','contract')}


def report_from_result(output, contract):
    keys=contract['report_keys']
    rows=output['rows']
    if contract['grouped']:
        if any(len(r)!=2 or not isinstance(r[0],str) for r in rows):
            raise ValueError('Grouped report needs group label and numeric value per row')
        labels=[r[0] for r in rows]
        if len(labels)!=len(set(labels)):
            raise ValueError('Duplicate group labels')
        return {keys[0]:dict(rows)}
    if len(rows)!=1 or len(rows[0])!=len(keys):
        raise ValueError('Scalar report needs exactly one row with the requested number of fields')
    return dict(zip(keys,rows[0]))
