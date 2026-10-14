"""Offline detector audit; reference answers never enter live verification."""
import json
import argparse
import ast
from copy import deepcopy
from pathlib import Path
from study.data import corpus
from study.runtime import Session, LARGE
from pilot.environment import correct
from study.experiment import read_rows


def reference_queries():
    """Offline evaluator fixtures, never passed to live agent requests."""
    source=ast.parse(Path('pilot/test_richer.py').read_text())
    return next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='SQL' for t in n.targets))


def detector_audit(tasks,states,trajectories,verifier='original'):
    """Measure each source separately; exclude all seen v1 test families."""
    tasks={t['id']:t for t in tasks if t['split'] in ('train','validation')}
    sources={name:[] for name in ('saved_initial','reference_sql','correct_large')}
    for record in states:
        if (record['task_id'] in tasks and record.get('stage','initial')=='initial'
                and record['fault']=='clean' and not record['state']['error']):
            sources['saved_initial'].append((tasks[record['task_id']],record['state']))
    references=reference_queries()
    for task in tasks.values():
        s=Session(task,verifier=verifier);s.state['sql']=references[task['template']];s.execute()
        if s.state['error']: raise ValueError('Reference SQL failed: '+task['id'])
        if not correct(s.state['report'],task['expected']): raise ValueError('Incorrect reference SQL: '+task['id'])
        sources['reference_sql'].append((task,s.state))
    seen=set()
    for record in states:
        if (record['task_id'] not in tasks or record['fault']!='clean' or record['state']['error']
                or record['state']['model']!=LARGE): continue
        task=tasks[record['task_id']]
        if not correct(record['state']['report'],task['expected']): continue
        key=(record['task_id'],record['state']['sql'])
        if key not in seen:
            seen.add(key);sources['correct_large'].append((task,record['state']))
    for row in trajectories:
        if row['task_id'] not in tasks or row['fault']!='clean' or not row['success']: continue
        generations=[e for e in row['trace'] if e['kind']=='generation']
        tools=[e for e in row['trace'] if e['kind']=='tool']
        if not generations or generations[-1]['model']!=LARGE or not tools or tools[-1]['error']: continue
        tool=tools[-1];key=(row['task_id'],tool['sql'])
        if key in seen: continue
        seen.add(key)
        state=dict(sql=tool['sql'],output=tool['output'],report=row['report'],error=None,checks=[],model=LARGE)
        sources['correct_large'].append((tasks[row['task_id']],state))
    result={}
    for source,population in sources.items():
        counts=dict(true_positive=0,false_positive=0,true_negative=0,false_negative=0);details=[]
        per_template={t:dict(counts) for t in sorted({task['template'] for task in tasks.values()})}
        for task,state in population:
            s=Session(task,verifier=verifier);s.state=deepcopy(state)
            incorrect=not correct(s.state['report'],task['expected']);s.check();flagged=s.state['checks']['flagged']
            key=('true_positive' if incorrect else 'false_positive') if flagged else ('false_negative' if incorrect else 'true_negative')
            counts[key]+=1
            per_template[task['template']][key]+=1
            details.append({'task_id':task['id'],'template':task['template'],'incorrect':incorrect,'flagged':flagged,'checks':s.state['checks']})
        tp,fp,tn,fn=(counts[k] for k in ('true_positive','false_positive','true_negative','false_negative'))
        result[source]={'counts':counts,'specificity':tn/(tn+fp) if tn+fp else None,
                        'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,
                        'per_template':per_template,'details':details}
    return result


def small_by_template(validation,branches):
    """Observed small-only trajectories and initial accept branches, without inference."""
    populations={'validation_small':[r for r in validation if r['policy']=='small' and r['fault']=='clean'
                                    and r.get('split')=='validation'],
                 'training_small_initial_accept':[r for r in branches if r['action']=='accept'
                    and r.get('stage','initial')=='initial' and r['fault']=='clean' and r.get('split')=='train'
                    and r['initial_state']['model']!=LARGE]}
    templates=sorted({t['template'] for t in corpus() if t['split'] in ('train','validation')})
    return {name:{t:{'runs':len(rs),'successes':sum(bool(r['success']) for r in rs)}
                  for t in templates for rs in [[r for r in rows if r['template']==t]]}
            for name,rows in populations.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default='results/v2')
    parser.add_argument('--output',default=None)
    parser.add_argument('--verifier',choices=['original','contract_v2'],default='original')
    args=parser.parse_args(); root=Path(args.directory); target=Path(args.output) if args.output else root/'detector_diagnostic.json'
    if not target.resolve().is_relative_to(Path('results/v2').resolve()):
        parser.error('New diagnostics must be written under results/v2.')
    branches=read_rows(root/'branches.jsonl');validation=read_rows(root/'validation.jsonl')
    trajectories=[r['branch_result'] for r in branches]+validation
    result={'scope':'train/validation only; sources reported separately; repeated references are not independent model outputs',
            'verifier':args.verifier,'source_directory':args.directory,'evidence_status':'post-hoc exploratory diagnostic; not a held-out result',
            'sources':detector_audit(corpus(),read_rows(root/'training_states.jsonl'),trajectories,args.verifier),
            'small_by_template':small_by_template(validation,branches)}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    summary=deepcopy(result)
    for source in summary['sources'].values(): source.pop('details')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
