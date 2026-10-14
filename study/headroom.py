"""Controller-free, train/validation-only v2 timing and capability profile."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from study.experiment import read_rows, validate_test_configuration
from study.runtime import Session, SMALL, LARGE, api


def frozen_development(directory,seeds=10):
    root=Path(directory);raw=(root/'corpus.json').read_bytes()
    config=json.loads((root/'configuration.json').read_text())
    if config.get('corpus_version')!='v2': raise ValueError('Frozen v2 corpus required')
    if hashlib.sha256(raw).hexdigest()!=config['dataset_sha256']: raise ValueError('Frozen corpus bytes changed')
    if config.get('corpus_source_sha256')!=hashlib.sha256(Path('study/data_v2.py').read_bytes()).hexdigest():
        raise ValueError('V2 corpus source changed after freezing')
    if not 1<=seeds<=10: raise ValueError('Use 1 to 10 seeds')
    tasks=json.loads(raw)
    validate_test_configuration(config,tasks);validate_test_configuration(config,tasks,'schema_shift')
    return [t for t in tasks if t['split'] in ('train','validation') and t['seed']<seeds and t['id'].endswith('w0')],config


def timing_tasks(tasks):
    """Exactly ten distinct development templates, with one database per task."""
    first={}
    for task in sorted(tasks,key=lambda t:(t['seed'],t['template'])):
        first.setdefault(task['template'],task)
    train=[first[t] for t in sorted(first) if first[t]['split']=='train']
    val=[first[t] for t in sorted(first) if first[t]['split']=='validation']
    chosen=train[:6]+val[:4]
    if len(chosen)!=10: raise ValueError('Timing batch requires six train and four validation templates')
    return chosen


def headroom_summary(rows,tasks,small_model,large_model):
    expected={(t['id'],model,0) for t in tasks for model in (small_model,large_model)}
    keys=[(r['task_id'],r['model'],r['trial']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)-expected: raise ValueError('Duplicated or incompatible headroom rows')
    result={}
    for model in (small_model,large_model):
        rs=[r for r in rows if r['model']==model]
        templates={t['template'] for t in tasks};by_template={}
        for template in sorted(templates):
            values=[r for r in rs if r['template']==template]
            successes=sum(r['success'] for r in values);n=len(values)
            by_template[template]={'runs':n,'successes':successes,'success_rate':successes/n if n else None}
        solved=sum(v['success_rate'] is not None and v['success_rate']>=.5 for v in by_template.values())
        result[model]={'runs':len(rs),'success_rate':sum(r['success'] for r in rs)/len(rs) if rs else None,
                       'templates_at_least_half_correct':solved,'template_fraction':solved/len(templates),
                       'per_template':by_template}
    complete=set(keys)==expected
    suitable=(.3<=result[small_model]['template_fraction']<=.7 and
              result[large_model]['success_rate']>result[small_model]['success_rate']) if complete else None
    return {'scope':'fresh generation and execution only; no controller, verifier, repair or escalation',
            'complete':complete,'suitable_pair':suitable,'models':result,
            'pair_rule':'small model correct on at least half the seeds in 30–70% of development templates; large model has higher overall success'}


def timing_estimate(rows):
    if not rows: raise ValueError('Timing rows required')
    mean=sum(r['wall_seconds'] for r in rows)/len(rows)
    # Planned maximum budgets, not a prediction of controller action frequency.
    counts={'training_states':8*5*2*3,'validation_controller_trajectories':2*2*40*9,
            'validation_baseline_trajectories':2*2*40*6,
            'primary_test_trajectories':8*10*3*2*9,'schema_shift_trajectories':4*10*3*2*9}
    trajectories=sum(v for k,v in counts.items() if k.endswith('trajectories'))
    upper_calls=counts['training_states']*7*2+160+trajectories*3
    return {'observed_calls':len(rows),'mean_generation_trajectory_seconds':mean,'planned_counts':counts,
            'headroom_profile_calls':240,'estimated_headroom_profile_seconds':240*mean,
            'rough_full_workflow_model_call_cap':upper_calls,'rough_full_workflow_hours_at_cap':upper_calls*mean/3600,
            'limitations':'Upper-budget estimate; collection eligibility, action choices, serving order and verifier changes alter cost. Excludes extra consistency sampling and audit overhead. Rerun timing after runtime/model changes.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['timing','profile'])
    parser.add_argument('--corpus-freeze',default='results/v2/corpus_v2')
    parser.add_argument('--directory',default='results/v2/headroom/qwen3-06-17')
    parser.add_argument('--small-model',default=SMALL);parser.add_argument('--large-model',default=LARGE)
    parser.add_argument('--seeds',type=int,default=10)
    args=parser.parse_args();root=Path(args.directory)
    if not root.resolve().is_relative_to(Path('results/v2').resolve()): parser.error('Write under results/v2.')
    if args.small_model==args.large_model: parser.error('Use two distinct model tags.')
    tasks,config=frozen_development(args.corpus_freeze,args.seeds)
    if args.mode=='timing': tasks=timing_tasks(tasks)
    elif not (root/'timing_summary.json').exists(): parser.error('Run the ten-task timing batch and inspect its estimate first.')
    tags=api('/api/tags');installed={m['name']:m.get('digest') for m in tags['models']}
    if {args.small_model,args.large_model}-installed.keys(): parser.error('Both candidate models must already be installed.')
    sources=['study/headroom.py','study/runtime.py','study/data_v2.py','study/data.py','study/experiment.py','pilot/environment.py']
    metadata={'dataset_sha256':config['dataset_sha256'],'test_templates_sha256':config['test_templates_sha256'],
              'models':{m:installed[m] for m in (args.small_model,args.large_model)},'ollama':api('/api/version'),
              'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
              'decoding':{'temperature':.2,'seed':0,'max_calls':1}}
    root.mkdir(parents=True,exist_ok=True);provenance=root/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())!=metadata:
        raise ValueError('Source, corpus or candidate model changed; use a new directory')
    if not provenance.exists(): provenance.write_text(json.dumps(metadata,indent=2)+'\n')
    path=root/(args.mode+'.jsonl');rows=read_rows(path)
    done={(r['task_id'],r['model'],r['trial']) for r in rows}
    with path.open('a') as stream:
        if args.mode=='profile':
            expected={(t['id'],m,0) for t in tasks for m in (args.small_model,args.large_model)}
            for recorded in read_rows(root/'timing.jsonl'):
                key=(recorded['task_id'],recorded['model'],recorded['trial'])
                if key in expected and key not in done:
                    reused=dict(recorded,phase='profile',reused_from='timing.jsonl')
                    stream.write(json.dumps(reused)+'\n');stream.flush();rows.append(reused);done.add(key)
        for task in tasks:
            for model in (args.small_model,args.large_model):
                key=(task['id'],model,0)
                if key in done: continue
                session=Session(task,max_calls=1,small_model=args.small_model,large_model=args.large_model)
                started=time.perf_counter();session.generate(model)
                result=session.result();result.update(model=model,phase=args.mode,wall_seconds=time.perf_counter()-started)
                stream.write(json.dumps(result)+'\n');stream.flush();rows.append(result);done.add(key)
                print(args.mode,task['id'],model,'success',result['success'],flush=True)
    summary=timing_estimate(rows) if args.mode=='timing' else headroom_summary(rows,tasks,args.small_model,args.large_model)
    summary.update(dataset_sha256=config['dataset_sha256'],test_templates_sha256=config['test_templates_sha256'])
    (root/(args.mode+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
