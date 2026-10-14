"""Paired descriptive analysis with template-cluster uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from study.experiment import read_rows, job_key


def _validation_stats(rows,policy):
    if any(r.get('split','validation')!='validation' for r in rows):
        raise ValueError('Select operating settings on validation rows only')
    candidates=sorted({(r['penalty'],r.get('threshold',0.0)) for r in rows
                       if r['policy']==policy and r['fault']=='clean'})
    large=[r for r in rows if r['policy']=='large' and r['fault']=='clean']
    if not large or not candidates: raise ValueError('Need clean validation reference and controller candidates')
    types={r.get('controller_type','logistic_v1') for r in rows if r['policy']==policy}
    if len(types)>1: raise ValueError('Select a single controller type before comparing operating points')
    population={(r['task_id'],r['trial']) for r in large}
    if len(population)!=len(large): raise ValueError('Duplicated reference validation runs')
    stats=[]
    for value,threshold in candidates:
        rs=[r for r in rows if r['policy']==policy and r['penalty']==value
            and r.get('threshold',0.0)==threshold and r['fault']=='clean']
        keys=[(r['task_id'],r['trial']) for r in rs]
        if len(keys)!=len(set(keys)) or set(keys)!=population:
            raise ValueError('Validation grid incomplete or duplicated; match every candidate to the reference tasks/trials')
        if not np.isfinite(value) or value<0 or not np.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('Invalid operating point')
        stats.append({'penalty':value,'threshold':threshold,'runs':len(rs),
                      'success':float(np.mean([r['success'] for r in rs])),
                      'effort':float(np.mean([r['effort_seconds'] for r in rs]))})
    return float(np.mean([r['success'] for r in large])),stats


def select_penalty(rows,policy='learned'):
    """Legacy v1 selection retained for exact reproduction of old choices."""
    reference,stats=_validation_stats(rows,policy)
    admissible=[s for s in stats if s['success']>=reference-0.10]
    selected=min(admissible,key=lambda s:(s['effort'],-s['threshold'])) if admissible else max(stats,key=lambda s:(s['success'],-s['effort'],s['threshold']))
    return {'policy':policy,'selected_penalty':selected['penalty'],'selected_threshold':selected['threshold'],'validation_large_success':reference,
            'selection_rule':'legacy lowest effort within tolerance; otherwise highest success then lowest effort',
            'candidates':stats,'note':'Legacy selection; not a statistical equivalence test.'}


def select_operating_point(rows,tolerance=0.10,policy='learned'):
    """V2 grid rule: explicitly flag ties and avoid favouring costly penalties in a degenerate fallback."""
    if not np.isfinite(tolerance) or not 0<=tolerance<=1: raise ValueError('Invalid success tolerance')
    reference,stats=_validation_stats(rows,policy)
    ok=[s for s in stats if s['success']>=reference-tolerance]
    all_tied=len({s['success'] for s in stats})==1
    if ok:
        selected=min(ok,key=lambda s:(s['effort'],-s['threshold']));mode='admissible'
    else:
        best=max(s['success'] for s in stats);tied=[s for s in stats if s['success']==best]
        degenerate=len(tied)==len(stats)
        selected=min(tied,key=lambda s:(s['penalty'],s['effort'])) if degenerate else min(tied,key=lambda s:s['effort'])
        mode='degenerate_tie' if degenerate else 'best_success_fallback'
    return {'policy':policy,'selected_penalty':selected['penalty'],'selected_threshold':selected['threshold'],
            'selection_mode':mode,'all_successes_tied':all_tied,'tolerance':tolerance,
            'validation_large_success':reference,'candidates':stats,
            'note':'degenerate_tie means validation could not distinguish settings; do not claim the choice is evidence-based.'}


def cluster_interval(rows,field):
    groups=sorted({r['template'] for r in rows}); rng=np.random.default_rng(20261012)
    values={g:np.asarray([float(r[field]) for r in rows if r['template']==g]) for g in groups}
    samples=[]
    for _ in range(2000):
        drawn=rng.choice(groups,len(groups),replace=True)
        samples.append(np.mean([np.mean(values[g]) for g in drawn]))
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def summarise(rows):
    result=[]
    for fault in sorted({r['fault'] for r in rows}):
        points=sorted({(r['policy'],r['penalty'],r.get('threshold',0.0),job_key(r)[-1]) for r in rows})
        for policy,penalty,threshold,controller_type in points:
            rs=[r for r in rows if r['fault']==fault and r['policy']==policy
                and r['penalty']==penalty and r.get('threshold',0.0)==threshold and job_key(r)[-1]==controller_type]
            if not rs: continue
            result.append({'policy':policy,'fault':fault,'penalty':penalty,'threshold':threshold,'controller_type':controller_type,'runs':len(rs),
                'unique_tasks':len({r['task_id'] for r in rs}),'templates':len({r['template'] for r in rs}),
                'success_rate':float(np.mean([r['success'] for r in rs])),
                'success_cluster_interval':cluster_interval(rs,'success'),
                'mean_effort_seconds':float(np.mean([r['effort_seconds'] for r in rs])),
                'effort_cluster_interval':cluster_interval(rs,'effort_seconds'),
                'median_wall_seconds':float(np.median([r['wall_seconds'] for r in rs])),
                'p95_wall_seconds':float(np.quantile([r['wall_seconds'] for r in rs],.95)),
                'mean_calls':float(np.mean([r['calls'] for r in rs])),
                'total_effort_seconds':sum(r['effort_seconds'] for r in rs),
                'failed_run_effort_seconds':sum(r['effort_seconds'] for r in rs if not r['success']),
                'verification_operations':sum(e['kind']=='verification' for r in rs for e in r['trace']),
                'switches':sum(sum(a!=b for a,b in zip(ms,ms[1:])) for r in rs
                    for ms in [[e['model'] for e in r['trace'] if e['kind']=='generation']]),
                'by_template':{t:{'successes':sum(r['success'] for r in rs if r['template']==t),
                                     'runs':sum(r['template']==t for r in rs)} for t in sorted({r['template'] for r in rs})}})
    return result


def paired(rows,left='learned',right='large',controller_type=None):
    if controller_type is not None:
        rows=[r for r in rows if job_key(r)[-1] in ('none',controller_type)]
    diffs=[]
    for fault in sorted({r['fault'] for r in rows}):
        for policy in (left,right):
            keys=[(r['task_id'],r['trial']) for r in rows if r['policy']==policy and r['fault']==fault]
            if len(keys)!=len(set(keys)):
                raise ValueError('Paired comparison requires one operating point per policy; filter the grid first')
        l={(r['task_id'],r['trial']):r for r in rows if r['policy']==left and r['fault']==fault}
        rr={(r['task_id'],r['trial']):r for r in rows if r['policy']==right and r['fault']==fault}
        common=l.keys() & rr.keys()
        if not common: continue
        ds=[{'template':l[k]['template'],'success':int(l[k]['success'])-int(rr[k]['success']),
             'effort_seconds':l[k]['effort_seconds']-rr[k]['effort_seconds']} for k in common]
        diffs.append({'left':left,'right':right,'fault':fault,'paired_runs':len(ds),
                      'success_difference':float(np.mean([d['success'] for d in ds])),
                      'success_difference_interval':cluster_interval(ds,'success'),
                      'effort_difference_seconds':float(np.mean([d['effort_seconds'] for d in ds])),
                      'effort_difference_interval':cluster_interval(ds,'effort_seconds')})
        if controller_type is not None: diffs[-1]['controller_type']=controller_type
    return diffs


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['select','summarise']); parser.add_argument('--directory',default='results/v2')
    parser.add_argument('--policy',choices=['learned','table'],default='learned')
    parser.add_argument('--controller',choices=['logistic_v1','table','logistic_v2'],default='logistic_v1')
    parser.add_argument('--selection-rule',choices=['legacy','v2'],default='legacy')
    args=parser.parse_args(); root=Path(args.directory)
    if root.resolve()==Path('results/main').resolve():
        parser.error('The frozen primary analysis is preserved. Use a separate revised directory.')
    if not root.resolve().is_relative_to(Path('results/v2').resolve()):
        parser.error('New outputs must be under results/v2.')
    if args.mode=='select':
        select=select_operating_point if args.selection_rule=='v2' else select_penalty
        rows=read_rows(root/'validation.jsonl')
        rows=[r for r in rows if r['policy']=='large' or
              (r['policy']==args.policy and job_key(r)[-1]==args.controller)]
        result=select(rows,policy=args.policy)
        result['controller_type']=args.controller
        result['analyse_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        name='selection.json' if args.policy=='learned' else 'table_selection.json'
        (root/name).write_text(json.dumps(result,indent=2)+'\n')
    else:
        test=read_rows(root/'test.jsonl'); branches=read_rows(root/'branches.jsonl')
        result={'evidence_status':'post-hoc exploratory; not pooled with the primary study',
                'test':summarise(test),'paired':paired(test,controller_type=args.controller)+paired(test,'learned','rules',args.controller)+paired(test,'learned','learned_model_only',args.controller)+paired(test,'learned','escalate_default',args.controller)+paired(test,'table','large','table'),
                'training_branches':len(branches),'training_states':len(read_rows(root/'training_states.jsonl')),
                'branch_collection_incremental_effort_seconds':sum(r['remaining_effort_seconds'] for r in branches),
                'uncertainty_method':'2000 template-cluster resamples of within-template means; only three held-out templates; intervals are descriptive and coarse',
                'selection':json.loads((root/'selection.json').read_text())}
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
