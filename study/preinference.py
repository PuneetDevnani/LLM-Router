"""Offline, dated v2 corpus amendments and development-only detector audit."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from pilot.environment import correct
from study.data_v2 import corpus,REFERENCE_SQL,MONTH_START,MONTH_END,GROUPS
from study.experiment import template_list_hash
from study.runtime import Session
from study.mutation_audit import mutation_audit,family_counts

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'results/v2'
DATE='2026-10-13'
VERIFIERS=('original','contract_v2','contract_v3')
PROBES=('date_boundary_records','net_count_boundary_records','cancelled_null_discount','discount_irrelevance','public_nonnegative_range')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_preinference(directory=None):
    """Refuse corpus/protocol amendments after any final freeze or held-out job."""
    for pattern in ('FREEZE.json','TEST_STARTED.json','test.jsonl','schema_shift.jsonl'):
        for path in V2.rglob(pattern):
            # Historical v1 evidence is outside this root; v2 authoring fixtures
            # are not JSONL model outcomes and never trigger this guard.
            if pattern.endswith('.json') or path.stat().st_size:
                raise ValueError('Pre-inference amendment blocked by '+str(path))
    registration=Path(directory) if directory is not None else V2/'corpus_v2'
    path=registration/'PREREGISTRATION.json'
    if not path.exists(): return
    current=json.loads(path.read_text())
    if not current.get('amendment_date'): return # Earlier development profiles are historical.
    active=current['dataset_sha256']
    for name in ('INFERENCE_STARTED.json','provenance.json','PLAN.json','configuration.json'):
        for metadata_path in V2.rglob(name):
            metadata=json.loads(metadata_path.read_text())
            if metadata.get('dataset_sha256')!=active: continue
            streams=[p for label in ('timing.jsonl','profile.jsonl','branches.jsonl','validation.jsonl')
                     for p in metadata_path.parent.rglob(label) if p.stat().st_size]
            if name=='INFERENCE_STARTED.json' or streams:
                raise ValueError('Amended development inference already started: '+str(metadata_path))


def constraint_audit(tasks=None):
    """Oracle-score omission queries, never model-generated test outcomes."""
    tasks=corpus() if tasks is None else tasks;result={}
    for name in ('month_totals','having_regions'):
        rows=[t for t in tasks if t['template']==name and t['id'].endswith('w0')]
        sql=REFERENCE_SQL[name]
        omissions={'having':sql.split(' HAVING')[0]} if name=='having_regions' else {
            'window':sql.replace(f" AND order_date>='{MONTH_START}' AND order_date<'{MONTH_END}'",''),
            'start':sql.replace(f" AND order_date>='{MONTH_START}'",''),
            'end':sql.replace(f" AND order_date<'{MONTH_END}'",'')}
        result[name]={}
        for label,wrong in omissions.items():
            cases=[]
            for task in rows:
                s=Session(task);s.state['sql']=wrong;s.execute()
                if s.state['error']: raise ValueError('Constraint omission must remain executable')
                cases.append({'seed':task['seed'],'changes_answer':not correct(s.state['report'],task['expected'])})
            changed=sum(c['changes_answer'] for c in cases)
            result[name][label]={'seeds':len(cases),'changed_answers':changed,'fraction':changed/len(cases),'sql':wrong,'cases':cases}
            if changed/len(cases)<.7: raise ValueError('Constraint is insufficiently exercised: '+name+' '+label)
    return result


def reference_audit(tasks=None,verifiers=VERIFIERS):
    """All authored reference fixtures, distinct template/seed populations."""
    tasks=corpus() if tasks is None else tasks;rows=[];seen=set()
    for task in tasks:
        key=(task['template'],task['seed'])
        if key in seen: continue
        seen.add(key)
        s=Session(task);s.state['sql']=REFERENCE_SQL[task['template']];s.execute()
        if s.state['error'] or not correct(s.state['report'],task['expected']): raise ValueError('Reference/oracle mismatch: '+task['id'])
        checks={}
        for version in verifiers:
            checker=Session(task,verifier=version);checker.state=deepcopy(s.state);checker.check();checks[version]=checker.state['checks']
        rows.append({'template':task['template'],'seed':task['seed'],'split':task['split'],'sql':s.state['sql'],
                     'checks':checks,'oracle_relation':'matches','kind':'reference'})
    return {'scope':'Authoring fixtures only; includes held-out references, no model outcomes or detector tuning.',
            'instances':len(rows),'distinct_sql_strings':len({r['sql'] for r in rows}),
            'summary':{v:family_counts(rows,v) for v in verifiers},
            'per_template':{t:{'instances':sum(r['template']==t for r in rows),
                'distinct_sql_strings':len({r['sql'] for r in rows if r['template']==t}),
                **{v:family_counts([r for r in rows if r['template']==t],v) for v in verifiers}} for t in sorted({r['template'] for r in rows})},
            'cases':rows}


def targeted_design_variants(task):
    """Additional design cases, restricted to the old development templates."""
    if task['split'] not in ('train','validation'): raise ValueError('Probe design is development-only')
    name=task['template'];sql=REFERENCE_SQL[name];variants=[]
    if name in ('january','date_window','summary'):
        end=task['contract']['date_end_exclusive']
        variants.append(('inclusive_end_boundary',sql.replace("order_date<'"+end+"'","order_date<='"+end+"'")))
    if name=='large_net': variants.append(('strict_net_boundary',sql.replace('>=5000','>5000')))
    if name=='net_total': variants.append(('cancelled_null_disjunction',sql+" OR discount IS NULL"))
    if name=='region_total': variants.append(('spurious_discount_dependence',sql.replace('SUM(o.amount)','SUM(o.amount-COALESCE(o.discount,0))')))
    if name in ('net_total','region_total') and task['contract'].get('nonnegative') is True:
        s=Session(task);s.state['sql']=sql;s.execute()
        columns=s.state['output']['columns'];quoted=['"'+c.replace('"','""')+'"' for c in columns]
        fields=[c if task['contract'].get('grouped') and i==0 else '-ABS('+c+')' for i,c in enumerate(quoted)]
        variants.append(('negative_result','SELECT '+','.join(fields)+' FROM ('+sql+') AS negative_result'))
    return variants


def detector_audit():
    tasks=corpus();dev=[t for t in tasks if t['split'] in ('train','validation')]
    # Existing allocation is immutable. Evaluate once after implementing the
    # registered amendment; no adaptation to evaluation-family results.
    result=mutation_audit(dev,VERIFIERS,references=REFERENCE_SQL)
    targeted=[]
    for task in dev:
        if not task['id'].endswith('w0'): continue
        for family,sql in targeted_design_variants(task):
            s=Session(task);s.state['sql']=sql;s.execute();checks={}
            for v in VERIFIERS:
                checker=Session(task,verifier=v);checker.state=deepcopy(s.state);checker.check();checks[v]=checker.state['checks']
            targeted.append({'template':task['template'],'seed':task['seed'],'family':family,'sql':sql,'kind':'mutated',
                'oracle_relation':'execution_error' if s.state['error'] else 'matches' if correct(s.state['report'],task['expected']) else 'differs',
                'checks':checks})
    result['additional_design_cases']={'note':'Designed alongside this amendment on development contracts only. Inert mutants count as oracle-matching cases, even if a counterfactual probe flags them.',
        'per_family':{f:{'distinct_sql_strings':len({r['sql'] for r in targeted if r['family']==f}),
            **{v:family_counts([r for r in targeted if r['family']==f],v) for v in VERIFIERS}} for f in sorted({r['family'] for r in targeted})},'cases':targeted}
    result['reference_sql']=reference_audit(tasks)
    result['constraint_coverage']=constraint_audit(tasks)
    result['probe_ablations']={}
    for name in PROBES:
        cases=deepcopy(result['cases'])
        for case in cases:
            for evidence in case['checks'].values():
                evidence['evidence']=[e for e in evidence['evidence'] if e['check']!=name]
                evidence['flagged']=bool(evidence['concerns']) or any(e['status']=='violation' for e in evidence['evidence'])
        result['probe_ablations'][name]={p:family_counts([c for c in cases if c['family_partition']==p],'contract_v3') for p in ('design','evaluation')}
    result['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'study/data_v2.py',ROOT/'study/verifier.py',ROOT/'study/runtime.py',ROOT/'study/mutation_audit.py']}
    return result


def amend(directory,revision=None):
    root=Path(directory).resolve();check_preinference(root)
    if not root.is_relative_to(V2): raise ValueError('Amendment outputs must be under results/v2')
    if revision not in (None,'nonnegative-split'): raise ValueError('Unknown amendment revision')
    amendment_root=V2/'amendments'/DATE
    if revision: amendment_root=amendment_root/revision
    backup=amendment_root/'corpus_before'
    if not backup.exists(): shutil.copytree(root,backup)
    old=json.loads((backup/'PREREGISTRATION.json').read_text())
    tasks=corpus();raw=json.dumps(tasks,indent=2)+'\n';config=json.loads((backup/'configuration.json').read_text())
    config.update(dataset_sha256=hashlib.sha256(raw.encode()).hexdigest(),corpus_source_sha256=digest(ROOT/'study/data_v2.py'),
                  amendment_date=DATE,generation_cache=False,cache_directory=None,groups=deepcopy(GROUPS))
    templates={name:{'instances':sum(t['template']==name for t in tasks),
        'sha256':value_hash({'tasks':[t for t in tasks if t['template']==name],'reference_sql':REFERENCE_SQL[name]})}
        for name in sorted({t['template'] for t in tasks})}
    config['template_definitions_sha256']=value_hash(templates)
    for split in ('test','schema_shift'):
        if sorted(GROUPS[split])!=config[split+'_templates']: raise ValueError('Holdout template list changed')
        config[split+'_templates_sha256']=template_list_hash(config[split+'_templates'])
    prereg={k:v for k,v in old.items() if k!='preregistration_sha256'}
    prereg.update(status='pre-inference amended dataset; new development inference and final controller freeze pending',
        amendment_date=DATE,previous_dataset_sha256=old['dataset_sha256'],previous_protocol_sha256=old['protocol_sha256'],
        protocol_sha256=digest(ROOT/'V2_PROTOCOL.md'),dataset_sha256=config['dataset_sha256'],
        corpus_source_sha256=config['corpus_source_sha256'],template_definitions_sha256=config['template_definitions_sha256'],
        amendments=['explicit net-value guard and all-template reference checks','non-vacuous monthly window and HAVING threshold',
            'opt-in exact-body generation cache with recorded usage','replayed-effort, live sample, trajectory counts and template-level criterion',
            'development-designed boundary, cancelled-NULL and discount-irrelevance probes'],
        probe_design_sha256=digest(amendment_root/'PROBE_DESIGN.json'),
        generation_cache={'opt_in':True,'default':False,'workflow_enabled':True,'key':'sha256(json.dumps(request_body).encode())',
            'accounting':'Recorded load, prompt and generation durations charged on hits and misses; live tools and probes measured.'},
        live_rerun_sample={'split':'development only','tasks':10,'models':2,'requests':20,'trial':0,
            'timing_task_selection':'study.headroom.timing_tasks','cache_bypassed':True,'no_adaptation_to_sample':True},
        trajectory_counts={'primary':4320,'schema_shift':2160,'oracle_jobs':720,'oracle_candidate_trajectories':1440,
                           'total_candidate_trajectories':7920},
        note='V1 primary; v2 exploratory. Older development profiles predate the amended contracts and cannot be resumed or pooled. No fresh held-out inference has started.')
    if revision:
        prereg['amendments']=old['amendments']+[
            'completion_rate moved to validation; distinct_customers moved to training; holdout lists unchanged',
            'optional human-authored nonnegative contract and development-designed sign range probe',
            'offline historical profile economics with finite-sample oracle diagnostics']
        prereg['amendment_revision']=revision
    prereg['preregistration_sha256']=value_hash(prereg)
    root.mkdir(parents=True,exist_ok=True)
    (root/'corpus.json').write_text(raw)
    for name,value in [('configuration.json',config),('template_hashes.json',templates),('PREREGISTRATION.json',prereg),('constraint_coverage.json',constraint_audit(tasks))]:
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (root/'DATASET_FREEZE.md').write_text(f'# Amended dataset freeze — {DATE}\n\nPre-inference amendment authorised by the researcher. Previous snapshot: {backup.relative_to(ROOT)}.\n\nDataset SHA256: {config["dataset_sha256"]}\n\nCorpus source SHA256: {config["corpus_source_sha256"]}\n\nTemplate definitions SHA256: {config["template_definitions_sha256"]}\n\nPreregistration payload SHA256: {prereg["preregistration_sha256"]}\n\nThis is a corpus freeze, not a fitted-controller freeze or permission to tune on held-out outcomes.\n')
    return prereg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['amend','audit'])
    p.add_argument('--directory',default='results/v2/corpus_v2');p.add_argument('--output',default='results/v2/amendments/2026-10-13/mutation_audit.json')
    p.add_argument('--revision',choices=['nonnegative-split'])
    args=p.parse_args()
    if args.mode=='amend': print(json.dumps(amend(args.directory,revision=args.revision),indent=2));return
    check_preinference();path=Path(args.output).resolve()
    if not path.is_relative_to(V2): p.error('Audit outputs must be under results/v2')
    if path.exists(): p.error('Preserve completed audit output; use a new filename for any distinct audit')
    result=detector_audit();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'partitions':result['by_family_partition'],'references':{k:v for k,v in result['reference_sql'].items() if k not in ('cases','per_template')},'output':str(path)},indent=2))


if __name__=='__main__': main()
