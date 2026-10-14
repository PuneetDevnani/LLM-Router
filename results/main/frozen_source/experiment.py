"""Resumable local experiments. Gold labels are used only for offline learning/scoring."""
import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from study.data import corpus, GROUPS
from study.runtime import Session, api, SMALL, LARGE, fixed_continuation, lexical_checks
from study.controller import Controller, features


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()] if Path(path).exists() else []


def provenance():
    return {'ollama':api('/api/version'),'models':api('/api/tags'),
            'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('study').glob('*.py'))}}


def run_policy(task,policy,trial=0,fault='clean',controller=None,penalty=0.1):
    s=Session(task,trial,fault)
    s.generate(LARGE if policy=='large' else SMALL)
    for _ in range(5):
        eligible=s.eligible()
        if policy.startswith('learned'):
            mode={'learned':'joint','learned_model_only':'model_only','learned_no_effort':'no_effort','learned_no_history':'no_history'}[policy]
            action,values=controller.choose(s,penalty,mode)
            s.trace.append({'kind':'prediction','values':values})
        elif s.state['error']=='tool_timeout' and 'retry' in eligible:
            action='retry'
        elif s.state['error']:
            action=('escalate' if policy=='rules' else 'repair' if policy=='small' else 'escalate') if s.calls<s.max_calls else 'stop'
        elif policy=='rules' and not s.state['checks']:
            action='verify'
        elif policy=='rules' and s.state['checks'].get('flagged') and 'escalate' in eligible:
            action='escalate'
        else: action='accept'
        s.apply(action)
        if action in ('accept','stop'): break
    else:
        s.state['report']=None
        s.trace.append({'kind':'termination','reason':'controller_iteration_budget'})
    result=s.result(); result.update(policy=policy,penalty=penalty)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['freeze','collect','train','evaluate'])
    p.add_argument('--output',default='results/main')
    p.add_argument('--split',choices=['train','validation','test'],default='validation')
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--wordings',type=int,default=1)
    p.add_argument('--trials',type=int,default=1)
    p.add_argument('--policies',default='small,large,rules,learned,learned_model_only')
    p.add_argument('--faults',default='clean')
    p.add_argument('--penalties',default='0.1')
    p.add_argument('--stage',choices=['initial','checked','repaired'],default='initial')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.mode=='freeze':
        Path('data/main-corpus.json').write_text(json.dumps(corpus(),indent=2)+'\n')
        (out/'configuration.json').write_text(json.dumps({'groups':GROUPS,'max_calls':3,'max_tool_operations':6,'context':4096,
            'formatter':'deterministic declared output schema','wording_variants':2,
            'default_decoding':{'temperature':0.2,'num_predict':384},'dataset_sha256':hashlib.sha256(Path('data/main-corpus.json').read_bytes()).hexdigest()},indent=2)+'\n')
        print('Frozen 240 instances in 12 grouped templates.'); return
    if args.mode=='train':
        rows=read_rows(out/'branches.jsonl')
        if not rows: raise ValueError('Collect training branches first')
        Controller.train(rows).save(out/'controller.json')
        print('Trained controller on',len(rows),'executed branches.'); return
    metadata=provenance(); (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    ts=[t for t in corpus() if t['split']==('train' if args.mode=='collect' else args.split) and t['seed']<args.seeds and int(t['id'][-1])<args.wordings]
    rng=random.Random(20261012); rng.shuffle(ts)
    if args.mode=='collect':
        path=out/'branches.jsonl'; existing=read_rows(path)
        states_path=out/'training_states.jsonl'
        states={(r['task_id'],r['fault'],r['trial'],r.get('stage','initial')):r for r in read_rows(states_path)}
        done={(r['task_id'],r['fault'],r['trial'],r.get('stage','initial'),r['action']) for r in existing}
        with path.open('a') as f:
            for task in ts:
                for fault in args.faults.split(','):
                    for trial in range(args.trials):
                        state_key=(task['id'],fault,trial,args.stage)
                        s=Session(task,trial,fault)
                        if state_key in states:
                            saved=states[state_key]
                            s.state=deepcopy(saved['state']); s.calls=saved['calls']; s.operations=saved['operations']; s.trace=deepcopy(saved['trace'])
                        else:
                            original=states.get((task['id'],fault,trial,'initial'))
                            if original:
                                s.state=deepcopy(original['state']); s.calls=original['calls']; s.operations=original['operations']; s.trace=deepcopy(original['trace'])
                            else: s.generate(SMALL)
                            if args.stage=='checked' and 'verify' in s.eligible(): s.apply('verify')
                            if args.stage=='repaired' and 'repair' in s.eligible(): s.apply('repair')
                            saved={'task_id':task['id'],'fault':fault,'trial':trial,'stage':args.stage,'state':s.state,'calls':s.calls,'operations':s.operations,'trace':s.trace}
                            with states_path.open('a') as sf: sf.write(json.dumps(saved)+'\n')
                            states[state_key]=saved
                        if all((task['id'],fault,trial,args.stage,a) in done for a in s.eligible()): continue
                        candidates=s.eligible(); rng.shuffle(candidates)
                        for action in candidates:
                            key=(task['id'],fault,trial,args.stage,action)
                            if key in done: continue
                            branch=s.clone(); branch.apply(action)
                            if action in ('accept','stop'): result=branch.result()
                            else: result=fixed_continuation(branch)
                            row={'task_id':task['id'],'template':task['template'],'split':'train','fault':fault,'trial':trial,'stage':args.stage,
                                 'features':features(s),'action':action,'success':result['success'],
                                 'remaining_effort_seconds':result['effort_seconds'],'initial_state':s.state,
                                 'initial_trace':s.trace,'branch_result':result}
                            f.write(json.dumps(row)+'\n'); f.flush(); done.add(key)
                        print('Collected',task['id'],fault,'branches',len(candidates),flush=True)
        return
    controller=Controller.load(out/'controller.json') if 'learned' in args.policies else None
    path=out/(args.split+'.jsonl'); rows=read_rows(path)
    done={(r['task_id'],r['fault'],r['trial'],r['policy'],r['penalty']) for r in rows}
    jobs=[(t,fault,trial,policy,float(penalty)) for t in ts for fault in args.faults.split(',') for trial in range(args.trials)
          for policy in args.policies.split(',') for penalty in (args.penalties.split(',') if policy.startswith('learned') else ['0.0'])]
    rng.shuffle(jobs)
    with path.open('a') as f:
        for task,fault,trial,policy,penalty in jobs:
            key=(task['id'],fault,trial,policy,penalty)
            if key in done: continue
            result=run_policy(task,policy,trial,fault,controller,penalty)
            f.write(json.dumps(result)+'\n'); f.flush(); done.add(key)
            print(args.split,task['id'],policy,fault,'success',result['success'],flush=True)


if __name__=='__main__': main()
