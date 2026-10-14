"""Prospective v2 workflow: development first, immutable freeze, one-shot holdouts."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from study.runtime import Session,api
from study.data import public
from study.experiment import read_rows,run_policy,job_key,CONTROLLER_FILES
from study.controller import CONTROLLER_TYPES
from study.headroom import frozen_development,timing_tasks,headroom_summary
from study.task_routing import TaskRouter
from study.analyse import select_operating_point,summarise
from study.freeze_protocol import (ROOT,confined,digest,source_files,canonical,write_freeze,
                                   validate_freeze,begin_test,claim_job,complete_job)
from study.freeze_protocol import verify_completed_result

RUNTIME={'small_model':'qwen3:1.7b','large_model':'qwen3:4b-q4_K_M','verifier':'contract_v3',
         'fresh_escalation':True,'neutral_feedback':True,'diversify_seed':True,
         'consistency_verification':True,'consistency_k':3,'initial_routing':True,'max_iterations':7}
BASELINES=('small','large','escalate_default','escalate_fresh','rules','task_routing')
POINTS={'penalties':[0,.15,.3],'thresholds':[0,.1,.2]}


def initialise(directory,corpus_directory):
    root=confined(directory);corpus=confined(corpus_directory)
    tasks,config=frozen_development(corpus)
    runtime={**RUNTIME,'generation_cache':True,'cache_directory':str(root/'generation_cache')}
    plan={'version':1,'corpus_directory':str(corpus),'dataset_sha256':config['dataset_sha256'],
          'runtime':runtime,'grid':POINTS,'initial_models':['small','large'],'controllers':['table','logistic_v2'],
          'training_seeds':5,'validation_seeds':10,'stages':['initial','checked','repaired','consistency','routing'],
          'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_files()},
          'criterion':{'tolerance':.1,'lower_mean_effort':True,'both_fault_conditions':True}}
    root.mkdir(parents=True,exist_ok=True);path=root/'PLAN.json'
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('Source or plan changed; use a new directory')
    if not path.exists(): path.write_text(json.dumps(plan,indent=2)+'\n')
    return plan,tasks


def verify_provider(plan):
    installed={m['name']:m['digest'] for m in api('/api/tags')['models']}
    models={m:installed[m] for m in (plan['runtime']['small_model'],plan['runtime']['large_model'])}
    return {'models':models,'provider_version':api('/api/version')}


def profile(directory,plan,tasks,timing=False):
    root=Path(directory)/'profile';root.mkdir(exist_ok=True)
    provenance={**verify_provider(plan),'runtime':plan['runtime'],'dataset_sha256':plan['dataset_sha256'],
                'source_hashes':plan['source_hashes'],'serving_order':'interleaved timing; grouped-model profile'}
    path=root/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance: raise ValueError('Profile provenance changed')
    if not path.exists(): path.write_text(json.dumps(provenance,indent=2)+'\n')
    if plan['runtime'].get('generation_cache'):
        from study.generation_cache import GenerationCache
        GenerationCache(plan['runtime']['cache_directory']).bind({k:provenance[k] for k in ('models','provider_version')})
    population=timing_tasks(tasks) if timing else tasks
    path=root/('timing.jsonl' if timing else 'profile.jsonl');rows=read_rows(path)
    done={(r['task_id'],r['model']) for r in rows};runtime=dict(plan['runtime'])
    runtime.pop('max_iterations')
    with path.open('a') as stream:
        if not timing:
            if not (root/'timing_summary.json').exists(): raise ValueError('Run timing and inspect its estimate before development')
            for r in read_rows(root/'timing.jsonl'):
                key=(r['task_id'],r['model'])
                if key not in done:
                    r=dict(r,reused_from='timing.jsonl');stream.write(json.dumps(r)+'\n');stream.flush();rows.append(r);done.add(key)
        jobs=([(t,m) for t in population for m in provenance['models']] if timing else
              [(t,m) for m in provenance['models'] for t in population])
        for task,model in jobs:
            if (task['id'],model) in done: continue
            marker=root/'INFERENCE_STARTED.json'
            if not marker.exists(): marker.write_text(json.dumps({'dataset_sha256':plan['dataset_sha256']})+'\n')
            s=Session(task,max_calls=1,**runtime);started=time.perf_counter();s.generate(model)
            r=s.result();r.update(model=model,wall_seconds=time.perf_counter()-started)
            stream.write(json.dumps(r)+'\n');stream.flush();rows.append(r);done.add((task['id'],model))
            print('profile',len(done),'/',len(jobs),task['id'],model,r['success'],flush=True)
    if timing:
        mean=sum(r['wall_seconds'] for r in rows)/len(rows)
        # Conservative generation cap: two initial routes, five collection stages,
        # up to ten candidate actions and six calls per branch including verification.
        cap=160*5*10*6+160*2+4*40*9*6+6*40*6+120*3*2*9*6+120*3*2*2*3
        result={'timing_tasks':10,'model_calls':20,'mean_seconds':mean,'conservative_model_call_cap':cap,
                'estimated_hours_at_cap':cap*mean/3600,
                'note':'Budget cap, not expected action frequency. Includes consistency samples; warming and action eligibility change actual duration.'}
    else:
        result=headroom_summary(rows,tasks,plan['runtime']['small_model'],plan['runtime']['large_model'])
        result.update(dataset_sha256=plan['dataset_sha256'])
    (root/('timing_summary.json' if timing else 'profile_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def experiment_args(root,plan,initial,controller):
    runtime=plan['runtime'];args=['--output',str(root),'--corpus','v2','--controller',controller,'--initial-model',initial,
        '--small-model',runtime['small_model'],'--large-model',runtime['large_model'],'--verifier',runtime['verifier'],
        '--consistency-k',str(runtime['consistency_k']),'--max-iterations',str(runtime['max_iterations'])]
    for name in ('fresh_escalation','neutral_feedback','diversify_seed','consistency_verification','initial_routing'):
        if runtime[name]: args.append('--'+name.replace('_','-'))
    if runtime.get('generation_cache'): args.extend(['--generation-cache','--cache-directory',runtime['cache_directory']])
    return args


def run_module(module,args):
    print('Stage:',module,' '.join(args),flush=True)
    subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,check=True)


def append_unique(path,rows):
    existing=read_rows(path);done={job_key(r) for r in existing}
    with Path(path).open('a') as stream:
        for row in rows:
            key=job_key(row)
            if key in done: continue
            stream.write(json.dumps(row)+'\n');done.add(key)


def develop(directory,plan,tasks):
    root=Path(directory);report=profile(root,plan,tasks)
    if not report['complete'] or not report['suitable_pair']: raise ValueError('Current runtime/model pair failed the fixed headroom gate; collection not launched')
    router_path=root/'routing'/'task_router.json';router_path.parent.mkdir(exist_ok=True)
    if not router_path.exists(): run_module('study.task_routing',['--profile-directory',str(root/'profile'),'--output',str(router_path)])
    # Finish all collection and all fitted comparisons before writing validation.
    leaves=[]
    for initial in plan['initial_models']:
        collection=root/'collection'/initial;collection.mkdir(parents=True,exist_ok=True)
        args=experiment_args(collection,plan,initial,'logistic_v2')
        if not (collection/'configuration.json').exists(): run_module('study.experiment',['freeze',*args])
        for stage in plan['stages']:
            run_module('study.experiment',['collect',*args,'--seeds','5','--faults','clean,transient','--stage',stage])
        for controller in plan['controllers']:
            leaf=root/'controllers'/initial/controller;leaf.mkdir(parents=True,exist_ok=True)
            for name in ('configuration.json','corpus.json','branches.jsonl','training_states.jsonl','provenance.json'):
                source=collection/name;target=leaf/name
                if target.exists() and digest(target)!=digest(source): raise ValueError('Preserved training snapshot differs')
                if not target.exists(): shutil.copyfile(source,target)
            if not (leaf/CONTROLLER_FILES[controller]).exists():
                run_module('study.experiment',['train',*experiment_args(leaf,plan,initial,controller),'--cross-validate'])
            leaves.append((initial,controller,leaf))
    baseline=root/'validation_baselines';baseline.mkdir(exist_ok=True)
    if not (baseline/'configuration.json').exists(): run_module('study.experiment',['freeze',*experiment_args(baseline,plan,'small','logistic_v2')])
    run_module('study.experiment',['evaluate',*experiment_args(baseline,plan,'small','logistic_v2'),'--split','validation','--seeds','10',
        '--policies',','.join(BASELINES),'--penalties','0','--thresholds','0','--router',str(router_path)])
    references=read_rows(baseline/'validation.jsonl');selections=[]
    for initial,controller,leaf in leaves:
        append_unique(leaf/'validation.jsonl',references)
        policy='table' if controller=='table' else 'learned'
        run_module('study.experiment',['evaluate',*experiment_args(leaf,plan,initial,controller),'--split','validation','--seeds','10',
            '--policies',policy,'--penalties',','.join(map(str,POINTS['penalties'])),'--thresholds',','.join(map(str,POINTS['thresholds']))])
        selected=select_operating_point(read_rows(leaf/'validation.jsonl'),policy=policy)
        selected.update(initial_model=initial,controller_type=controller,analyse_sha256=digest(ROOT/'study/analyse.py'))
        path=leaf/'selection.json'
        if path.exists() and json.loads(path.read_text())!=selected: raise ValueError('Preserved selection differs')
        if not path.exists(): path.write_text(json.dumps(selected,indent=2)+'\n')
        selections.append(dict(selected,directory=str(leaf)))
    selected={}
    for controller in plan['controllers']:
        candidates=[s for s in selections if s['controller_type']==controller]
        def point(s): return next(c for c in s['candidates'] if c['penalty']==s['selected_penalty'] and c['threshold']==s['selected_threshold'])
        eligible=[s for s in candidates if point(s)['success']>=s['validation_large_success']-.1]
        choice=min(eligible,key=lambda s:point(s)['effort']) if eligible else min(candidates,key=lambda s:(-point(s)['success'],point(s)['effort']))
        selected[controller]=choice
    artifact={'selected':selected,'all_initial_model_selections':selections,'routing_file':str(router_path),
              'validation_only':True,'note':'Initial-route comparison uses validation only; no test outcomes were read.'}
    path=root/'selection.json'
    if path.exists() and json.loads(path.read_text())!=artifact: raise ValueError('Selected configuration changed')
    if not path.exists(): path.write_text(json.dumps(artifact,indent=2)+'\n')
    return artifact


def final_freeze(directory,plan):
    root=Path(directory);selection=json.loads((root/'selection.json').read_text());settings={}
    files=[root/'PLAN.json',root/'selection.json',Path(selection['routing_file']),Path(plan['corpus_directory'])/'PREREGISTRATION.json']
    for name in BASELINES: settings[name]={'initial_model':'small','controller_type':'none','penalty':0,'threshold':0}
    for policy,controller in [('table','table'),('learned','logistic_v2'),('learned_model_only','logistic_v2')]:
        chosen=selection['selected'][controller];leaf=Path(chosen['directory']);model=leaf/CONTROLLER_FILES[controller]
        settings[policy]={'initial_model':chosen['initial_model'],'controller_type':controller,'controller_file':str(model),
                          'penalty':chosen['selected_penalty'],'threshold':chosen['selected_threshold'],'selection_mode':chosen['selection_mode']}
        settings[policy]['selection_file']=str(leaf/'selection.json')
        files.extend([model,leaf/'selection.json',leaf/('training_cv_'+controller+'.json')])
    provider=verify_provider(plan)
    if plan['runtime'].get('generation_cache'):
        from study.generation_cache import GenerationCache
        cache=GenerationCache(plan['runtime']['cache_directory']);cache.bind(provider)
        snapshot={p.name:digest(p) for p in sorted(cache.directory.glob('*.json'))}
        path=root/'GENERATION_CACHE_FREEZE.json'
        if path.exists() and json.loads(path.read_text())!=snapshot: raise ValueError('Cache freeze changed')
        if not path.exists(): path.write_text(json.dumps(snapshot,indent=2)+'\n')
        files.extend([path,cache.directory/'CONTEXT.json'])
    spec={**provider,'corpus_directory':plan['corpus_directory'],'runtime':plan['runtime'],'policy_settings':settings,
          'router_file':selection['routing_file'],'files':list(dict.fromkeys(files)),
          'validation_complete':True,'schedule':{'seeds':10,'trials':3,'faults':['clean','transient'],'wordings':1},'criterion':plan['criterion']}
    return write_freeze(root,spec)


def oracle_outcome(task,trial,fault,runtime):
    rs=[run_policy(task,p,trial,fault,**runtime) for p in ('small','large')]
    correct=[r for r in rs if r['success']];chosen=min(correct or rs,key=lambda r:r['effort_seconds'])
    result=deepcopy(chosen);result.update(policy='oracle_upper_bound',controller_type='none',penalty=0,threshold=0,
        calls=sum(r['calls'] for r in rs),operations=sum(r['operations'] for r in rs),effort_seconds=sum(r['effort_seconds'] for r in rs),
        wall_seconds=sum(r['wall_seconds'] for r in rs),oracle_candidates=rs,
        evidence_status='Hindsight upper bound for choosing between two fixed baseline outcomes; not a deployable policy or a bound on all recovery methods.')
    if runtime.get('generation_cache'):
        for name in ('logical_model_calls','live_model_calls','cache_hits','replayed_model_effort_seconds'):
            result[name]=sum(r[name] for r in rs)
    return result


def heldout(directory):
    root=Path(directory);spec=validate_freeze(root)
    if spec['runtime'].get('generation_cache'):
        sample_path=root/'LIVE_RERUN_SAMPLE.json'
        if not sample_path.exists(): raise ValueError('Run the preregistered live development sample before held-out inference')
        live_sample_summary(root,spec)
    spec=begin_test(root);provider=verify_provider({'runtime':spec['runtime']})
    if any(provider[k]!=spec[k] for k in provider): raise ValueError('Frozen provider/model digests changed')
    tasks=json.loads((Path(spec['corpus_directory'])/'corpus.json').read_text())
    router=TaskRouter.load(spec['router_file']);controllers={}
    for policy,setting in spec['policy_settings'].items():
        if 'controller_file' in setting: controllers[policy]=CONTROLLER_TYPES[setting['controller_type']].load(setting['controller_file'])
    for split in ('test','schema_shift'):
        population=[t for t in tasks if t['split']==split and t['seed']<10 and t['id'].endswith('w0')]
        paths={p:root/(split+'_oracle.jsonl' if p=='oracle_upper_bound' else split+'.jsonl') for p in [*spec['policy_settings'],'oracle_upper_bound']}
        saved=[r for path in set(paths.values()) for r in read_rows(path)]
        for r in saved: verify_completed_result(root,split,job_key(r),r)
        done={job_key(r) for r in saved}
        # Group policies to reduce load churn; every requested task/trial/fault remains in the schedule.
        for policy in paths:
            setting=spec['policy_settings'].get(policy,{'initial_model':'small','controller_type':'none','penalty':0,'threshold':0})
            for task in population:
                for fault in spec['schedule']['faults']:
                    for trial in range(3):
                        key=(task['id'],fault,trial,policy,setting['penalty'],setting['threshold'],setting['controller_type'])
                        if key in done: continue
                        claim_job(root,split,key)
                        if policy=='oracle_upper_bound': result=oracle_outcome(task,trial,fault,spec['runtime'])
                        else:
                            result=run_policy(task,policy,trial,fault,controller=controllers.get(policy),router=router,
                                initial_model=setting['initial_model'],controller_type=setting['controller_type'],
                                penalty=setting['penalty'],threshold=setting['threshold'],**spec['runtime'])
                        with paths[policy].open('a') as stream: stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
                        complete_job(root,split,key,result);done.add(key)
                        print(split,len(done),policy,task['id'],fault,trial,result['success'],flush=True)
    return analyse_results(root)


def live_rerun_sample(directory):
    """Twenty fixed development requests, bypassing cache after final freeze."""
    root=Path(directory);spec=validate_freeze(root);path=root/'LIVE_RERUN_SAMPLE.json'
    if (root/'TEST_STARTED.json').exists(): raise ValueError('Live sample must precede held-out inference')
    if path.exists():
        result=json.loads(path.read_text())
        if result['freeze_sha256']!=digest(root/'FREEZE.json'): raise ValueError('Live sample belongs to a different freeze')
        return result
    provider=verify_provider({'runtime':spec['runtime']})
    if any(provider[k]!=spec[k] for k in provider): raise ValueError('Frozen provider changed')
    tasks,_=frozen_development(spec['corpus_directory']);runtime=dict(spec['runtime']);runtime.pop('max_iterations')
    if not runtime.get('generation_cache'): raise ValueError('Live replay sample requires cache-enabled freeze')
    rows=[];cached_requests=[]
    for task in timing_tasks(tasks):
        for model in provider['models']:
            cached=Session(task,max_calls=1,**runtime)
            cached.generate(model,cache_only=True);cached_requests.append((task,model,cached))
    ledger=root/'LIVE_RERUN_STARTED.json'
    with ledger.open('x') as stream: json.dump({'freeze_sha256':digest(root/'FREEZE.json')},stream)
    for task,model,cached in cached_requests:
        live=Session(task,max_calls=1,**{**runtime,'generation_cache':False});live.generate(model)
        from pilot.checks import equivalent
        same_output=(cached.state['output'] is None and live.state['output'] is None) or (
            cached.state['output'] is not None and live.state['output'] is not None and equivalent(cached.state['output'],live.state['output']))
        row={'task_id':task['id'],'model':model,'same_sql':cached.state['sql']==live.state['sql'],
             'same_result':same_output,'same_request':cached.last_request_sha256==live.last_request_sha256,
             'request_sha256':live.last_request_sha256,'cached':cached.result(),'live':live.result()}
        rows.append(row)
        with (root/'live_rerun_sample.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    result={'freeze_sha256':digest(root/'FREEZE.json'),'requests':len(rows),
            'sql_mismatches':sum(not r['same_sql'] for r in rows),'result_mismatches':sum(not r['same_result'] for r in rows),
            'scope':'Development-only, fixed timing tasks; no adaptation. Cache-defined outcomes are conditional on recorded generations.',
            'rows':rows}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def decomposition(rows):
    from pilot.environment import correct
    result={}
    for policy in sorted({r['policy'] for r in rows}):
        rs=[r for r in rows if r['policy']==policy];tp=fn=fp=tn=flagged=recovered=0
        for r in rs:
            events=[e for e in r['trace'] if e['kind']=='verification' and 'report' in e and e.get('error') is None]
            for e in events:
                wrong=not correct(e['report'],r['expected']);flag=e.get('flagged',False)
                tp+=int(wrong and flag);fn+=int(wrong and not flag);fp+=int(not wrong and flag);tn+=int(not wrong and not flag)
            had_flag=any(e.get('flagged',False) for e in events);flagged+=had_flag;recovered+=int(had_flag and r['success'])
        result[policy]={'detector_events':tp+fn+fp+tn,'true_positive':tp,'false_negative':fn,'false_positive':fp,'true_negative':tn,
            'detector_sensitivity':tp/(tp+fn) if tp+fn else None,'false_alarm_rate':fp/(fp+tn) if fp+tn else None,
            'flagged_trajectories':flagged,'correct_after_flag':recovered,'recovery_success_after_flag':recovered/flagged if flagged else None,
            'final_successes':sum(r['success'] for r in rs),'runs':len(rs),
            'note':'Detector events can repeat within trajectories; correct-after-flag is terminal correctness, not proof of causal recovery.'}
    return result


def cache_accounting(rows):
    """Separate logical replayed effort from physical calls, by policy/fault."""
    result=[]
    for policy,fault in sorted({(r['policy'],r['fault']) for r in rows}):
        rs=[r for r in rows if (r['policy'],r['fault'])==(policy,fault)]
        result.append({'policy':policy,'fault':fault,'trajectories':len(rs),
            **{k:sum(r.get(k,0) for r in rs) for k in ('logical_model_calls','live_model_calls','cache_hits','replayed_model_effort_seconds')},
            'charged_effort_seconds':sum(r['effort_seconds'] for r in rs),
            'observed_wall_seconds':sum(r['wall_seconds'] for r in rs)})
    return result


def live_sample_summary(directory,spec):
    root=Path(directory);sample=json.loads((root/'LIVE_RERUN_SAMPLE.json').read_text())
    tasks,_=frozen_development(spec['corpus_directory']);expected={(t['id'],m) for t in timing_tasks(tasks) for m in spec['models']}
    keys=[(r['task_id'],r['model']) for r in sample['rows']]
    if (sample['freeze_sha256']!=digest(root/'FREEZE.json') or sample['requests']!=20 or len(keys)!=20
            or len(set(keys))!=20 or set(keys)!=expected or any(not r.get('same_request') for r in sample['rows'])):
        raise ValueError('Live sample does not match the fixed twenty requests')
    mismatches={k:sum(not r[field] for r in sample['rows']) for k,field in [('sql_mismatches','same_sql'),('result_mismatches','same_result')]}
    if any(sample[k]!=v for k,v in mismatches.items()): raise ValueError('Live sample mismatch counts are inconsistent')
    return {'requests':20,**mismatches,'cached_charged_effort_seconds':sum(r['cached']['effort_seconds'] for r in sample['rows']),
            'live_effort_seconds':sum(r['live']['effort_seconds'] for r in sample['rows']),
            'live_wall_seconds':sum(r['live']['wall_seconds'] for r in sample['rows']),
            'interpretation':'Development sample only; no adaptation or replacement. Cache-defined correctness/charged effort does not establish uncached deployment benefit.'}


def analyse_results(directory):
    root=Path(directory);spec=validate_freeze(root);output={}
    for split,n in [('test',80),('schema_shift',40)]:
        rows=read_rows(root/(split+'.jsonl'));oracle=read_rows(root/(split+'_oracle.jsonl'))
        expected=n*3*2*len(spec['policy_settings'])
        if len(rows)!=expected or len({job_key(r) for r in rows})!=expected or len(oracle)!=n*3*2:
            raise ValueError('Incomplete or duplicated held-out schedule; no final summary claim')
        for row in rows+oracle: verify_completed_result(root,split,job_key(row),row)
        stats=summarise(rows);criteria=[]
        for fault in spec['schedule']['faults']:
            candidates=[s for s in stats if s['fault']==fault and s['policy'] in BASELINES]
            best=min(candidates,key=lambda s:(-s['success_rate'],s['mean_effort_seconds']))
            controller=next(s for s in stats if s['fault']==fault and s['policy']=='learned')
            passed=controller['success_rate']>=best['success_rate']-.1 and controller['mean_effort_seconds']<best['mean_effort_seconds']
            criteria.append({'fault':fault,'best_simple_baseline':best['policy'],'baseline_success':best['success_rate'],
                'baseline_effort':best['mean_effort_seconds'],'controller_success':controller['success_rate'],
                'controller_effort':controller['mean_effort_seconds'],'passed':passed})
        output[split]={'summary':stats,'decomposition':decomposition(rows),'criterion':criteria,
            'cache_accounting':cache_accounting(rows),'oracle_cache_accounting_separate':cache_accounting(oracle),
            'benefit_supported':all(c['passed'] for c in criteria),'oracle_separate':summarise(oracle)}
    output.update(evidence_status='Exploratory fresh v2, separate from primary v1',freeze_sha256=digest(root/'FREEZE.json'))
    if spec['runtime'].get('generation_cache'):
        output.update(live_replay_sample=live_sample_summary(root,spec),
                      benefit_scope='Descriptive correctness and replayed charged effort of the cache-defined policy experiment; no uncached deployment benefit established.')
    path=root/'summary.json'
    if path.exists() and json.loads(path.read_text())!=output: raise ValueError('Preserved summary differs')
    if not path.exists(): path.write_text(json.dumps(output,indent=2)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','timing','development','freeze','live-sample','test','summarise'])
    p.add_argument('--directory',default='results/v2/full_v2');p.add_argument('--corpus-directory',default='results/v2/corpus_v2')
    args=p.parse_args();root=confined(args.directory)
    if args.mode in ('live-sample','test','summarise'):
        result=live_rerun_sample(root) if args.mode=='live-sample' else heldout(root) if args.mode=='test' else analyse_results(root)
    else:
        plan,tasks=initialise(root,args.corpus_directory)
        if args.mode=='init': result=plan
        elif args.mode=='timing': result=profile(root,plan,tasks,True)
        elif args.mode=='development': result=develop(root,plan,tasks)
        else: result=final_freeze(root,plan)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
