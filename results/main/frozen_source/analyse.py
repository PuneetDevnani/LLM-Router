"""Paired descriptive analysis with template-cluster uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from study.experiment import read_rows


def select_penalty(rows):
    candidates=sorted({r['penalty'] for r in rows if r['policy']=='learned'})
    large=[r for r in rows if r['policy']=='large' and r['fault']=='clean']
    reference=np.mean([r['success'] for r in large])
    stats=[]
    for value in candidates:
        rs=[r for r in rows if r['policy']=='learned' and r['penalty']==value and r['fault']=='clean']
        stats.append({'penalty':value,'success':float(np.mean([r['success'] for r in rs])),
                      'effort':float(np.mean([r['effort_seconds'] for r in rs]))})
    admissible=[s for s in stats if s['success']>=reference-0.10]
    selected=min(admissible,key=lambda s:s['effort']) if admissible else max(stats,key=lambda s:(s['success'],-s['effort']))
    return {'selected_penalty':selected['penalty'],'validation_large_success':float(reference),
            'selection_rule':'lowest validation effort within 10 percentage points of stronger-model point success; otherwise highest success, then lowest effort',
            'candidates':stats,'note':'This operating-point rule is not a statistical equivalence test.'}


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
        for policy in sorted({r['policy'] for r in rows}):
            rs=[r for r in rows if r['fault']==fault and r['policy']==policy]
            if not rs: continue
            result.append({'policy':policy,'fault':fault,'runs':len(rs),
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


def paired(rows,left='learned',right='large'):
    diffs=[]
    for fault in sorted({r['fault'] for r in rows}):
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
    return diffs


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['select','summarise']); parser.add_argument('--directory',default='results/main')
    args=parser.parse_args(); root=Path(args.directory)
    if args.mode=='select':
        result=select_penalty(read_rows(root/'validation.jsonl'))
        (root/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        test=read_rows(root/'test.jsonl'); branches=read_rows(root/'branches.jsonl')
        result={'test':summarise(test),'paired':paired(test)+paired(test,'learned','rules')+paired(test,'learned','learned_model_only'),
                'training_branches':len(branches),'training_states':len(read_rows(root/'training_states.jsonl')),
                'branch_collection_incremental_effort_seconds':sum(r['remaining_effort_seconds'] for r in branches),
                'uncertainty_method':'2000 template-cluster resamples of within-template means; only three held-out templates; intervals are descriptive and coarse',
                'selection':json.loads((root/'selection.json').read_text())}
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
