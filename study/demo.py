"""Command-line project demo with actual local inference or explicit trace replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from study.data import corpus
from study.controller import Controller, TableController, CONTROLLER_TYPES
from study.experiment import run_policy, read_rows, CONTROLLER_FILES


def display(result,replay=False):
    print('SAVED TRACE REPLAY' if replay else 'LIVE LOCAL AGENT RUN')
    print('Task:',result['public']['prompt'])
    for event in result['trace']:
        if event['kind']=='generation': print('\nModel:',event['model'],'\nProposal:',event['content'])
        if event['kind']=='decision': print('Controller action:',event['action'])
        if event['kind']=='tool': print('Tool:',event.get('error') or 'executed successfully')
        if event['kind']=='verification': print('Verification:',event['flagged'],event.get('concerns',[]))
    print('\nReport:',json.dumps(result['report'],indent=2))
    print('Offline evaluator success:',result['success'])
    print('Measured deployment effort:',round(result['effort_seconds'],3),'seconds')
    print('Evaluation labels are shown only after execution; the controller does not receive them.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',default='net_total'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--policy',default='rules',choices=['small','large','rules','learned','learned_model_only','table','escalate_default','escalate_fresh'])
    p.add_argument('--directory',default='results/v2')
    p.add_argument('--controller',choices=sorted(CONTROLLER_TYPES),default='logistic_v1')
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--verifier',choices=['original','contract_v2'],default='original')
    p.add_argument('--fresh-escalation',action='store_true')
    p.add_argument('--neutral-feedback',action='store_true')
    p.add_argument('--diversify-seed',action='store_true')
    p.add_argument('--initial-model',choices=['small','large'],default='small')
    p.add_argument('--fault',default='clean',choices=['clean','transient','outage'])
    p.add_argument('--replay',action='store_true'); args=p.parse_args()
    if args.replay:
        rows=read_rows(Path(args.directory)/'test.jsonl')
        row=next((r for r in rows if r['policy']==args.policy and r['template']==args.template and r['fault']==args.fault
                  and (args.threshold is None or r.get('threshold',0.0)==args.threshold)),None)
        if row is None: p.error('No saved trace matches; choose a held-out template january, region_net, or large_net.')
        display(row,True); return
    task=next((t for t in corpus() if t['template']==args.template and t['seed']==args.seed and t['id'].endswith('w0')),None)
    if task is None: p.error('Unknown template/seed')
    root=Path(args.directory); selected=0.0; threshold=0.0; c=None
    if not root.resolve().is_relative_to(Path('results/v2').resolve()):
        p.error('Live demo outputs must be under results/v2; original evidence is replay-only.')
    if args.policy.startswith('learned') or args.policy=='table':
        selection_path=root/('table_selection.json' if args.policy=='table' else 'selection.json')
        controller_type='table' if args.policy=='table' else args.controller
        if not selection_path.exists() or not (root/CONTROLLER_FILES[controller_type]).exists():
            p.error('Train and select this v2 controller on validation first, or use --replay --directory results/main.')
        selection=json.loads(selection_path.read_text())
        selected=selection['selected_penalty']; threshold=selection.get('selected_threshold',0.0)
        controller_type='table' if args.policy=='table' else args.controller
        c=CONTROLLER_TYPES[controller_type].load(root/CONTROLLER_FILES[controller_type])
    if args.threshold is not None: threshold=args.threshold
    result=run_policy(task,args.policy,0,args.fault,c,selected,threshold,verifier=args.verifier,controller_type=args.controller,
                      fresh_escalation=args.fresh_escalation,neutral_feedback=args.neutral_feedback,diversify_seed=args.diversify_seed,
                      initial_model=args.initial_model)
    result['evidence_status']='exploratory demonstration; not part of the frozen primary study'
    demo=root/'demo'
    demo.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (demo/(stamp+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    (demo/'latest.json').write_text(json.dumps(result,indent=2)+'\n')
    display(result)


if __name__=='__main__': main()
