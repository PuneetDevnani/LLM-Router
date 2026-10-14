"""Training and validation only; a fresh v2 test requires a separate frozen phase."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',default='results/v2')
    p.add_argument('--verifier',choices=['original','contract_v2'],default='original')
    p.add_argument('--controller',choices=['table','logistic_v2','logistic_v1'],default='logistic_v1')
    p.add_argument('--selection-rule',choices=['legacy','v2'],default='legacy')
    p.add_argument('--fresh-escalation',action='store_true')
    p.add_argument('--neutral-feedback',action='store_true')
    p.add_argument('--diversify-seed',action='store_true')
    p.add_argument('--initial-model',choices=['small','large'],default='small')
    args=p.parse_args(); root=Path(args.directory)
    if not root.resolve().is_relative_to(Path('results/v2').resolve()): p.error('Use results/v2 or a subdirectory.')
    if not (root/'branches.jsonl').exists(): p.error('Collect training branches in this directory first.')
    model_name={'table':'table_controller.json','logistic_v2':'controller_logistic_v2.json','logistic_v1':'controller.json'}[args.controller]

    def run(module,*stage_args,controller=None):
        directory_flag='--output' if module=='study.experiment' else '--directory'
        command=[sys.executable,'-m',module,*stage_args,directory_flag,str(root)]
        if module=='study.experiment':
            command+=['--verifier',args.verifier,'--controller',controller or args.controller,'--initial-model',args.initial_model]
            if args.fresh_escalation: command+=['--fresh-escalation']
            if args.neutral_feedback: command+=['--neutral-feedback']
            if args.diversify_seed: command+=['--diversify-seed']
        print('Stage:', ' '.join(command[3:]),flush=True)
        subprocess.run(command,check=True)

    # Complete both comparisons before validation creates a stream that locks training.
    required={'table':'table_controller.json','logistic_v2':'controller_logistic_v2.json'}
    required[args.controller]=model_name
    if (root/'validation.jsonl').exists() and any(not (root/name).exists() for name in required.values()):
        p.error('Validation already started without all controllers trained. Use a new v2 directory.')
    for controller,name in required.items():
        if not (root/name).exists():
            run('study.experiment','train','--cross-validate',controller=controller)
    run('study.experiment','evaluate','--split','validation','--seeds','3',
        '--policies','small,large,rules,escalate_default,learned',
        '--penalties','0,0.05,0.15,0.3,0.6','--thresholds','0,0.05,0.1,0.2,0.3')
    run('study.analyse','select','--selection-rule',args.selection_rule,'--controller',args.controller)
    selection=json.loads((root/'selection.json').read_text())
    print('Validation selection:',json.dumps(selection),flush=True)
    print('Stopped after validation. No test split was run.',flush=True)


if __name__=='__main__': main()
