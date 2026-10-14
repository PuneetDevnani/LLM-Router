"""Training-only policy replay over executed action branches; no model calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from study.controller import CONTROLLER_TYPES, _training_rows, action_list, leave_one_template_out_cv

BASELINES=('always_accept','always_escalate','hindsight_best')
OPTION_ORDER=('accept','stop','repair','escalate','escalate_fresh','verify','retry')


def _group_branches(rows):
    """One logged outcome per eligible action of a saved state."""
    _training_rows(rows)
    groups=defaultdict(dict)
    for row in rows:
        key=(row['task_id'],row['fault'],row['trial'],row.get('stage','initial'))
        if row['action'] in groups[key]: raise ValueError('Duplicate state/action branch: '+str(key))
        if groups[key]:
            previous=next(iter(groups[key].values()))
            if previous['template']!=row['template'] or previous['features']!=row['features']:
                raise ValueError('Inconsistent template/features within a logged state')
        groups[key][row['action']]=row
    return groups


def _summarise_choices(choices):
    """Keep state counts and template counts visible rather than pooling rows."""
    n=len(choices); successes=sum(r['success'] for r in choices)
    effort=sum(r['remaining_effort_seconds'] for r in choices)
    return {'states':n,'successes':successes,'success_rate':successes/n if n else None,
            'total_remaining_effort_seconds':effort,'mean_remaining_effort_seconds':effort/n if n else None,
            'action_counts':dict(sorted(Counter(r['action'] for r in choices).items())),
            'fallback_states':sum(r['fallback'] for r in choices)}


def offline_policy_value(rows,controller=None,policy='controller',penalty=0.1,accept_threshold=0.0):
    """Select using features/options, then look up the executed branch label.

    Learned decisions never receive held-out labels. Hindsight deliberately
    uses them and is an upper bound on this finite set of logged outcomes.
    Missing accept/escalate falls back to accept (escalate baseline only),
    then stop. Stopping remains a policy outcome even though predictor CV
    excludes stop rows. Effort excludes generation before the saved state.
    """
    if not math.isfinite(penalty) or penalty<0 or not math.isfinite(accept_threshold) or not 0<=accept_threshold<=1:
        raise ValueError('Invalid fixed operating point')
    if policy not in ('controller',*BASELINES): raise ValueError('Unknown offline policy')
    if policy=='controller' and controller is None: raise ValueError('Controller required')
    choices=[]
    for key,branches in sorted(_group_branches(rows).items()):
        options=[a for a in OPTION_ORDER if a in branches]
        f=next(iter(branches.values()))['features'];fallback=False
        if policy=='controller':
            action,_=controller.choose_from(f,options,penalty=penalty,accept_threshold=accept_threshold)
        elif policy=='hindsight_best':
            action=min(options,key=lambda a:(-int(branches[a]['success']),branches[a]['remaining_effort_seconds'],options.index(a)))
        else:
            preferred='accept' if policy=='always_accept' else 'escalate'
            fallback=preferred not in branches
            action=preferred if not fallback else 'accept' if policy=='always_escalate' and 'accept' in branches else 'stop'
            if action not in branches: raise ValueError('Baseline fallback lacks a logged stop branch')
        observed=branches[action]
        choices.append({'task_id':key[0],'fault':key[1],'trial':key[2],'stage':key[3],
                        'template':observed['template'],'action':action,'success':int(observed['success']),
                        'remaining_effort_seconds':observed['remaining_effort_seconds'],
                        'eligible_logged_actions':options,'fallback':fallback})
    return {'policy':policy,'upper_bound':policy=='hindsight_best',
            'aggregate':_summarise_choices(choices),
            'per_template':{t:_summarise_choices([r for r in choices if r['template']==t])
                            for t in sorted({r['template'] for r in choices})},'choices':choices}


def evaluate_offline(rows,penalty=0.1,accept_threshold=0.0):
    """Hold out entire templates; fit/calibrate on the other training templates."""
    _group_branches(rows)
    templates=sorted({r['template'] for r in rows})
    if len({r['template'] for r in rows if r['action']!='stop'})<3:
        raise ValueError('Nested offline CV requires at least three non-stop training templates')
    actions=action_list(rows);policies={}
    for name,cls in CONTROLLER_TYPES.items():
        choices=[];folds=[]
        for held_out in templates:
            training=[r for r in rows if r['template']!=held_out]
            held=[r for r in rows if r['template']==held_out]
            model=cls.train(training,actions=actions)
            value=offline_policy_value(held,model,penalty=penalty,accept_threshold=accept_threshold)
            choices.extend(value['choices'])
            folds.append({'held_out_template':held_out,'training_templates':sorted({r['template'] for r in training}),
                          'fit_samples':model.model['samples'],
                          'calibration_folds':model.model.get('calibration_folds',[]),**value['aggregate']})
        policies[name]={'policy':name,'upper_bound':False,'aggregate':_summarise_choices(choices),
                        'per_template':{fold['held_out_template']:{k:v for k,v in fold.items()
                            if k not in ('held_out_template','training_templates','fit_samples','calibration_folds')}
                            for fold in folds},'folds':folds,'choices':choices}
    for name in BASELINES: policies[name]=offline_policy_value(rows,policy=name)
    return {'evidence_status':'exploratory training-template cross-validation; not a held-out deployment result',
            'scope':'one decision per logged state, followed by its recorded v1 fixed continuation',
            'limitations':['Only recorded actions/outcomes are evaluated; no new generations or multi-step controller rollouts.',
                           'Escalate labels use anchored v1 feedback, not escalate_fresh.',
                           'Effort is remaining branch effort, excluding generation/checks before the saved state.',
                           'Stages and faults can reuse tasks; state counts are not independent task/template counts.',
                           'Hindsight best sees the realized labels and is a finite logged-branch upper bound only.'],
            'fixed_settings':{'penalty':penalty,'accept_threshold':accept_threshold,'mode':'joint','selection':'fixed, no tuning'},
            'branch_rows':len(rows),'stop_rows':sum(r['action']=='stop' for r in rows),'templates':templates,
            'policies':policies,'prediction_cv':leave_one_template_out_cv(rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branches',default='results/main/branches.jsonl')
    parser.add_argument('--output',default='results/v2/offline_policy_eval.json')
    parser.add_argument('--penalty',type=float,default=0.1)
    parser.add_argument('--threshold',type=float,default=0.0)
    args=parser.parse_args();source=Path(args.branches);target=Path(args.output)
    if not target.resolve().is_relative_to(Path('results/v2').resolve()):
        parser.error('Offline outputs must be under results/v2; results/main is read-only.')
    if target.resolve()==source.resolve(): parser.error('Output must differ from input branches.')
    data=source.read_bytes();rows=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
    result=evaluate_offline(rows,args.penalty,args.threshold)
    if source.read_bytes()!=data: raise ValueError('Input branches changed during offline evaluation')
    result['source']={'path':str(source),'sha256':hashlib.sha256(data).hexdigest()}
    result['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__),Path(__file__).with_name('controller.py'))}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,value in result['policies'].items():
        upper=' (upper bound)' if value['upper_bound'] else ''
        print(name+upper,json.dumps(value['aggregate'],sort_keys=True))
        for template,stats in value['per_template'].items(): print(' ',template,json.dumps(stats,sort_keys=True))
    print('Saved',target)


if __name__=='__main__': main()
