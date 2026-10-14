"""Offline development-profile economics; never calls a model or a verifier."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def aggregate(outcomes):
    """Summarise charged effort, correctness and effort per correct outcome."""
    if not outcomes: raise ValueError('At least one outcome is required')
    n=len(outcomes);successes=sum(r['success'] for r in outcomes)
    effort=sum(r['effort_seconds'] for r in outcomes)
    return {'runs':n,'successes':successes,'success_rate':successes/n,
            'total_effort_seconds':effort,'mean_effort_seconds':effort/n,
            'effort_per_success_seconds':effort/successes if successes else None,
            'mean_model_calls':sum(r.get('model_calls',1) for r in outcomes)/n}


def profile_economics(rows,small_model,large_model):
    """Pair the same task/trial/fault; oracle decisions use recorded labels.

    Perfect detection charges small first, plus fresh large generation only
    on small failure. Detector overhead is assumed zero, so this is an
    optimistic diagnostic, not measured recovery or feedback escalation.
    """
    if small_model==large_model: raise ValueError('Distinct model roles required')
    pairs=defaultdict(dict)
    for row in rows:
        if row.get('split') not in ('train','validation'):
            raise ValueError('Development profiles only')
        if row.get('model') not in (small_model,large_model):
            raise ValueError('Unexpected model in profile')
        effort=row.get('effort_seconds')
        if (isinstance(effort,bool) or not isinstance(effort,(int,float))
                or not math.isfinite(effort) or effort<0):
            raise ValueError('Finite nonnegative charged effort required')
        if row.get('success') not in (True,False,0,1): raise ValueError('Binary correctness required')
        key=(row['task_id'],row.get('trial',0),row.get('fault','clean'))
        if row['model'] in pairs[key]: raise ValueError('Duplicate task/model profile row')
        pairs[key][row['model']]=row
    if not pairs: raise ValueError('Empty profile')
    outcomes={p:[] for p in ('always_small','always_large','oracle_pre_routing','small_first_perfect_detector')}
    per_model=defaultdict(lambda:defaultdict(list));per_policy=defaultdict(lambda:defaultdict(list))
    for pair in pairs.values():
        if set(pair)!={small_model,large_model}: raise ValueError('Unpaired task/model profile')
        a,b=pair[small_model],pair[large_model]
        if (a['template'],a['split'])!=(b['template'],b['split']): raise ValueError('Inconsistent task metadata')
        template=a['template']
        for row in (a,b): per_model[template][row['model']].append(row)
        candidates=[r for r in (a,b) if r['success']] or [a,b]
        chosen=min(candidates,key=lambda r:(r['effort_seconds'],r['model']))
        values={'always_small':a,'always_large':b,'oracle_pre_routing':chosen,
                'small_first_perfect_detector':{
                    'success':bool(a['success'] or b['success']),
                    'effort_seconds':a['effort_seconds']+(0 if a['success'] else b['effort_seconds']),
                    'model_calls':1+int(not a['success'])}}
        for policy,value in values.items():
            outcomes[policy].append(value);per_policy[template][policy].append(value)
    return {'scope':'paired development observations only; no model calls',
            'interpretation':{
                'oracle_pre_routing':'Finite-sample hindsight upper bound on correctness from these two recorded candidates; choose the cheaper correct candidate, or cheaper candidate when both fail. No deployable router or free acquisition of oracle labels.',
                'small_first_perfect_detector':'Optimistic zero-cost perfect detector: charge small on every task, then the recorded fresh-large cost on small failures. Same union correctness as the oracle, without anchoring. No measured detector/recovery benefit.',
                'effort':'Recorded charged effort, including serving/loading costs; offline recombination cannot predict different serving orders.',
                'effort_per_success':'Total charged effort divided by correct outcomes, including expenditure on failed tasks; null when none succeed.'},
            'models':{m:aggregate([r for r in rows if r['model']==m]) for m in (small_model,large_model)},
            'policies':{p:aggregate(rs) for p,rs in outcomes.items()},
            'per_template':{t:{'models':{m:aggregate(rs) for m,rs in per_model[t].items()},
                               'policies':{p:aggregate(rs) for p,rs in per_policy[t].items()}}
                            for t in sorted(per_model)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile',nargs='?',default='results/v2/headroom/qwen3-17-4q4/profile.jsonl')
    p.add_argument('--small-model',default='qwen3:1.7b')
    p.add_argument('--large-model',default='qwen3:4b-q4_K_M')
    p.add_argument('--output',default='results/v2/headroom/economics.json')
    a=p.parse_args();source=Path(a.profile);target=Path(a.output)
    if not target.resolve().is_relative_to(Path('results/v2').resolve()): p.error('Write under results/v2 only')
    if target.exists(): p.error('Preserve completed economics output; choose a new filename')
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    try: result=profile_economics(rows,a.small_model,a.large_model)
    except ValueError as exc: p.error(str(exc))
    provenance=source.parent/'provenance.json'
    metadata=json.loads(provenance.read_text()) if provenance.exists() else {}
    active=Path('results/v2/corpus_v2/PREREGISTRATION.json')
    current=json.loads(active.read_text()).get('dataset_sha256') if active.exists() else None
    result.update(profile_path=str(source.resolve()),profile_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  profile_dataset_sha256=metadata.get('dataset_sha256'),active_dataset_sha256=current,
                  historical_profile=bool(current and metadata.get('dataset_sha256')!=current),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for template,values in result['per_template'].items():
        for model,s in values['models'].items():
            print(f'{template}: {model}: {s["successes"]}/{s["runs"]} correct; mean effort {s["mean_effort_seconds"]:.3f}s')
    print(json.dumps(result['policies'],indent=2));print('Saved',target)


if __name__=='__main__': main()
