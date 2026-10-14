"""Offline, oracle-checked SQL mutations on train/validation templates only."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

from pilot.environment import correct
from study.data import corpus
from study.diagnostics import reference_queries
from study.runtime import Session

VERIFIERS=('original','contract_v2')
FAMILY_SPLIT_SHA256='a04908f76c339b505cf48f89277d922d51d6712f5d1fbdde5fd7c40a14d34041'
KEYWORD_RULES={
    'sum':'requested sum not visibly aggregated',
    'average':'requested average not visibly aggregated',
    'ratio':'requested ratio has no visible division',
    'distinct':'distinctness not visibly expressed',
    'grouping':'grouping not visibly expressed',
    'date_endpoints':'requested date endpoints are not both explicit; equivalent date expressions may be valid',
    'count':'requested count not visibly aggregated; alternative expressions may be valid',
    'null_discount_filter':'query excludes missing discounts despite the zero-null convention; inspect predicate context',
}


def family_partition_spec():
    """Load the committed family allocation; fail rather than silently reassign."""
    spec=json.loads(Path(__file__).with_name('mutation_family_split.json').read_text())
    payload={k:spec[k] for k in ('version','design','evaluation')}
    if payload['version']!=1: raise ValueError('Unknown mutation family protocol')
    for partition in ('design','evaluation'):
        names=payload[partition]
        if (not isinstance(names,list) or not names or any(not isinstance(n,str) or not n for n in names)
                or len(set(names))!=len(names)): raise ValueError('Invalid family partition')
    if set(payload['design']) & set(payload['evaluation']): raise ValueError('Overlapping mutation family partitions')
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=spec.get('sha256') or digest!=FAMILY_SPLIT_SHA256:
        raise ValueError('Frozen mutation family split hash mismatch')
    return spec


def ablated_flag(evidence,removed_keyword_rules=(),remove_net_probe=False):
    """Recompute the detector OR with components removed, without changing checks."""
    unknown=set(removed_keyword_rules)-KEYWORD_RULES.keys()
    if unknown: raise ValueError('Unknown keyword ablation rule: '+str(sorted(unknown)))
    concerns=evidence.get('concerns',[])
    if set(concerns)-set(KEYWORD_RULES.values()): raise ValueError('Unregistered keyword concern in audit')
    removed={KEYWORD_RULES[k] for k in removed_keyword_rules}
    remaining=[c for c in concerns if c not in removed]
    violations=[e for e in evidence.get('evidence',[]) if e['status']=='violation'
                and not (remove_net_probe and e['check']=='preserve_net_value')]
    return bool(remaining or violations)


def ablation_effect(baseline,ablated):
    """Report absolute recall/false-alarm changes in percentage points."""
    return {'metrics':ablated,
            'recall_change_pp':100*(ablated['recall']-baseline['recall'])
                if baseline['recall'] is not None and ablated['recall'] is not None else None,
            'false_alarm_change_pp':100*(ablated['false_alarm_rate']-baseline['false_alarm_rate'])
                if baseline['false_alarm_rate'] is not None and ablated['false_alarm_rate'] is not None else None}


def component_ablations(records,verifiers=VERIFIERS):
    """Each keyword's effect with/without the net-value probe; no rule selection."""
    result={}
    for verifier in verifiers:
        baseline=family_counts(records,verifier)
        without_net=family_counts(records,verifier,remove_net_probe=True)
        result[verifier]={
            'baseline':baseline,'remove_net_value_probe':ablation_effect(baseline,without_net),
            'remove_each_keyword':{rule:{
                'net_probe_kept':ablation_effect(baseline,family_counts(records,verifier,[rule])),
                'net_probe_removed':ablation_effect(without_net,family_counts(records,verifier,[rule],True))}
                for rule in KEYWORD_RULES},
            'remove_all_keywords':ablation_effect(baseline,family_counts(records,verifier,KEYWORD_RULES)),
            'remove_all_keywords_and_net_probe':ablation_effect(baseline,family_counts(records,verifier,KEYWORD_RULES,True))}
    return result


def query_variants(task,sql,columns):
    """Predefined fault/equivalence families; no inspected test queries or labels."""
    if task['split'] not in ('train','validation'): raise ValueError('Train/validation variants only')
    variants=[]
    def add(family,kind,query):
        variants.append({'family':family,'kind':kind,'sql':query})
    add('reference','reference',sql)
    add('subquery_wrapper','equivalent','SELECT * FROM ('+sql+') AS equivalent_result')
    status=r"\bstatus\s*=\s*'completed'"
    if re.search(status,sql,re.I):
        add('status_predicate','equivalent',re.sub(status,"status != 'cancelled'",sql,flags=re.I))
    # Bias one numeric report field while preserving grouping and output shape.
    quoted=['"'+c.replace('"','""')+'"' for c in columns]
    numeric_index=1 if task['contract']['grouped'] else 0
    biased=list(quoted);biased[numeric_index]+='+1'
    add('numeric_bias','mutated','SELECT '+','.join(biased)+' FROM ('+sql+') AS biased_result')
    if task['contract']['excludes_cancelled']:
        predicate=r"(?:\b\w+\.)?\bstatus\s*=\s*'completed'"
        if re.search(predicate,sql,re.I):
            add('omitted_status_filter','mutated',re.sub(predicate,'1=1',sql,flags=re.I))
    if re.search(r'\bSUM\s*\(',sql,re.I):
        add('wrong_aggregation','mutated',re.sub(r'\bSUM\s*\(','MAX(',sql,count=1,flags=re.I))
    if re.search(r'\bAVG\s*\(',sql,re.I):
        add('wrong_aggregation','mutated',re.sub(r'\bAVG\s*\(','SUM(',sql,count=1,flags=re.I))
        match=re.search(r'AVG\((.*?)\) FROM',sql,re.I)
        if match:
            expression=match.group(1)
            add('average_expression','equivalent',sql[:match.start()]+'1.0*SUM('+expression+')/COUNT(*) FROM'+sql[match.end():])
    null_expression=r'COALESCE\(discount,0\)'
    if re.search(null_expression,sql,re.I):
        add('null_function','equivalent',re.sub(null_expression,'IFNULL(discount,0)',sql,flags=re.I))
        if task['contract']['aggregation'] in ('sum','average'):
            add('omitted_null_handling','mutated',re.sub(null_expression,'discount',sql,flags=re.I))
            add('omitted_discount','mutated',re.sub(r'amount-COALESCE\(discount,0\)','amount',sql,flags=re.I))
    if 'date_start' in task['contract']:
        cutoff=(date.fromisoformat(task['contract']['date_start'])+timedelta(days=3)).isoformat()
        add('truncated_date_window','mutated',sql.replace(task['contract']['date_end_exclusive'],cutoff))
    if 'DISTINCT customer_id' in sql:
        add('missing_distinct','mutated',sql.replace('DISTINCT customer_id','customer_id'))
        add('distinct_subquery','equivalent',"SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM orders WHERE status='completed') AS unique_customers")
    if 'COUNT(*)' in sql:
        add('count_expression','equivalent',sql.replace('COUNT(*)','SUM(1)'))
        if task['contract']['aggregation']!='ratio':
            add('count_nullable_column','mutated',sql.replace('COUNT(*)','COUNT(discount)'))
    if task['contract']['aggregation']=='ratio':
        add('ratio_scale','mutated',sql.replace('100.0*','1.0*'))
        pattern=r'SUM\(CASE WHEN (.*?) THEN 1 ELSE 0 END\)'
        add('conditional_count','equivalent',re.sub(pattern,r'COUNT(CASE WHEN \1 THEN 1 END)',sql))
    if 'discount IS NULL' in sql:
        add('reversed_null_filter','mutated',sql.replace('discount IS NULL','discount IS NOT NULL'))
    if task['contract']['grouped']:
        add('group_order','equivalent',sql+' ORDER BY 1 DESC')
        add('join_predicate','mutated',sql.replace('c.id=o.customer_id','1=1'))
    return variants


def family_counts(records,verifier,removed_keyword_rules=(),remove_net_probe=False):
    """Denominators come from independent-oracle labels, including inert mutants."""
    counts=dict(true_positive=0,false_negative=0,false_positive=0,true_negative=0,execution_errors=0,inert_mutations=0)
    for record in records:
        relation=record['oracle_relation']
        if relation=='execution_error': counts['execution_errors']+=1;continue
        wrong=relation=='differs';evidence=record['checks'][verifier]
        flagged=ablated_flag(evidence,removed_keyword_rules,remove_net_probe) if removed_keyword_rules or remove_net_probe else evidence['flagged']
        key=('true_positive' if flagged else 'false_negative') if wrong else ('false_positive' if flagged else 'true_negative')
        counts[key]+=1
        counts['inert_mutations']+=int(record['kind']=='mutated' and not wrong)
    tp,fn,fp,tn=(counts[k] for k in ('true_positive','false_negative','false_positive','true_negative'))
    return {**counts,'wrong_queries':tp+fn,'correct_queries':fp+tn,
            'recall':tp/(tp+fn) if tp+fn else None,'false_alarm_rate':fp/(fp+tn) if fp+tn else None}


def mutation_audit(tasks=None,verifiers=VERIFIERS,partition=None,references=None):
    """Execute one set per template/DB seed; wording copies do not add samples."""
    tasks=corpus() if tasks is None else tasks
    spec=family_partition_spec()
    partitions={f:p for p in ('design','evaluation') for f in spec[p]}
    if partition not in (None,'design','evaluation'): raise ValueError('Unknown family partition')
    if not verifiers or set(verifiers)-{'original','contract_v2','contract_v3'}: raise ValueError('Unknown audit verifier')
    references=reference_queries() if references is None else references;cases=[];seen=set()
    for task in tasks:
        if task['split'] not in ('train','validation'): continue
        key=(task['template'],task['seed'])
        if key in seen: continue
        seen.add(key)
        reference=Session(task);reference.state['sql']=references[task['template']];reference.execute()
        if reference.state['error'] or not correct(reference.state['report'],task['expected']):
            raise ValueError('Reference disagrees with independent oracle: '+task['id'])
        for variant in query_variants(task,reference.state['sql'],reference.state['output']['columns']):
            if variant['family'] not in partitions: raise ValueError('Unregistered mutation family: '+variant['family'])
            if partition is not None and partitions[variant['family']]!=partition: continue
            session=Session(task);session.state['sql']=variant['sql'];session.execute()
            if session.state['error']:
                if variant['kind']!='mutated': raise ValueError('Equivalent/reference query failed: '+variant['sql'])
                relation='execution_error';checks={}
            else:
                matches=correct(session.state['report'],task['expected']);relation='matches' if matches else 'differs'
                if variant['kind']!='mutated' and not matches:
                    raise ValueError('Declared equivalent query disagrees with independent oracle: '+variant['sql'])
                checks={}
                for verifier in verifiers:
                    checker=Session(task,verifier=verifier);checker.state=deepcopy(session.state);checker.check()
                    checks[verifier]=checker.state['checks']
                    if ablated_flag(checks[verifier])!=checks[verifier]['flagged']:
                        raise ValueError('Detector flag cannot be reconstructed from recorded components')
            cases.append({'task_id':task['id'],'template':task['template'],'split':task['split'],'seed':task['seed'],
                          **variant,'family_partition':partitions[variant['family']],
                          'oracle_relation':relation,'report':session.state['report'],
                          'execution_error':session.state['error'],'checks':checks})
    if not cases: raise ValueError('Train/validation reference tasks required')
    design_sql={c['sql'] for c in cases if c['family_partition']=='design'}
    evaluation_sql={c['sql'] for c in cases if c['family_partition']=='evaluation'}
    return {'evidence_status':'exploratory synthetic detector audit, not model or held-out task performance',
            'scope':'train/validation only; one set per template and DB seed; verifiers unchanged',
            'oracle':'independent Python-record expected reports; numeric tolerance from pilot.environment.correct',
            'interpretation':'Equivalent means oracle-matching on these database instances, not a proof on arbitrary databases. Inert mutants count as correct, executable cases; execution errors are excluded from detector denominators.',
            'task_instances':len(seen),'variants':len(cases),'oracle_relations':dict(Counter(c['oracle_relation'] for c in cases)),
            'distinct_sql_strings':len({c['sql'] for c in cases}),'family_split':spec,
            'cross_partition_sql_overlap':{'distinct_sql_strings':len(design_sql & evaluation_sql),
                'sql_strings':sorted(design_sql & evaluation_sql),
                'note':'Family partitions are disjoint; SQL text can recur under different task contracts.'},
            'summary':{v:family_counts(cases,v) for v in verifiers},
            'per_family':{f:{'partition':partitions[f],'distinct_sql_strings':len({c['sql'] for c in cases if c['family']==f}),
                            **{v:family_counts([c for c in cases if c['family']==f],v) for v in verifiers}}
                          for f in sorted({c['family'] for c in cases})},
            'ablations':component_ablations(cases,verifiers),
            'by_family_partition':{p:{'variants':sum(c['family_partition']==p for c in cases),
                'distinct_sql_strings':len({c['sql'] for c in cases if c['family_partition']==p}),
                'summary':{v:family_counts([c for c in cases if c['family_partition']==p],v) for v in verifiers},
                'ablations':component_ablations([c for c in cases if c['family_partition']==p],verifiers)}
                for p in ('design','evaluation')},
            'per_template':{t:{v:family_counts([c for c in cases if c['template']==t],v) for v in verifiers}
                            for t in sorted({c['template'] for c in cases})},'cases':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/v2/mutation_audit.json')
    parser.add_argument('--corpus',choices=['v1','v2'],default='v1')
    parser.add_argument('--partition',choices=['all','design','evaluation'],default='all')
    parser.add_argument('--verifiers',default=','.join(VERIFIERS))
    args=parser.parse_args();target=Path(args.output)
    if not target.resolve().is_relative_to(Path('results/v2').resolve()): parser.error('Audit outputs must be under results/v2.')
    tasks=None;references=None
    if args.corpus=='v2':
        from study.data_v2 import corpus as v2_corpus,REFERENCE_SQL
        tasks=v2_corpus();references=REFERENCE_SQL
    result=mutation_audit(tasks,tuple(args.verifiers.split(',')),None if args.partition=='all' else args.partition,references)
    result.update(corpus_version=args.corpus,requested_partition=args.partition)
    sources=['study/mutation_audit.py','study/mutation_family_split.json','study/runtime.py','study/verifier.py','study/data.py','study/diagnostics.py',
             'pilot/checks.py','pilot/environment.py','pilot/richer.py','pilot/test_richer.py']
    if args.corpus=='v2': sources.append('study/data_v2.py')
    result['source_hashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('task_instances','variants','distinct_sql_strings','oracle_relations','summary','per_family')},indent=2))
    print('Saved',target)


if __name__=='__main__': main()
