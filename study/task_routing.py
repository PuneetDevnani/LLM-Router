"""Training-only calibrated public-contract routing, before SQL generation."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from study.controller import fit_platt, apply_platt


def task_type(public):
    """A category of public contract requirements, with no evaluator fields."""
    c=public['contract']
    return json.dumps([c['aggregation'],bool(c['grouped']),bool(c['distinct']),
                       bool(c.get('date_start')),bool(c.get('net_value',False))],separators=(',',':'))


class TaskRouter:
    def __init__(self,model): self.model=model

    @classmethod
    def fit_base(cls,rows,models,prior=2.0):
        if not rows or prior<=0: raise ValueError('Training observations and a positive prior required')
        if any(r['split']!='train' or r['fault']!='clean' or r['model'] not in models for r in rows):
            raise ValueError('Only clean training model profiles may fit routing')
        cells=defaultdict(lambda:[0,0,0.0]);totals={m:[0,0,0.0] for m in models}
        keys=[(r['task_id'],r['trial'],r['model']) for r in rows]
        if len(keys)!=len(set(keys)): raise ValueError('Duplicate routing observations')
        populations=[{(r['task_id'],r['trial']) for r in rows if r['model']==m} for m in models]
        if not populations[0] or any(p!=populations[0] for p in populations):
            raise ValueError('Candidate models must cover the same training tasks')
        for r in rows:
            for cell in (cells[task_type(r['public'])+'|'+r['model']],totals[r['model']]):
                cell[0]+=1;cell[1]+=int(r['success']);cell[2]+=r['effort_seconds']
        return cls({'models':list(models),'cells':dict(cells),'totals':totals,'prior':prior,
                    'calibration':{m:{'kind':'identity'} for m in models},'tolerance':.1,
                    'method':'public-contract empirical rates, training-template OOF Platt calibration'})

    def predict(self,public,model):
        m=self.model;n,s,e=m['totals'][model];global_p=s/n;global_c=e/n
        nc,sc,ec=m['cells'].get(task_type(public)+'|'+model,[0,0,0.0]);prior=m['prior']
        p=(sc+prior*global_p)/(nc+prior);cost=(ec+prior*global_c)/(nc+prior)
        p=max(1e-7,min(1-1e-7,p));logit=math.log(p/(1-p))
        return apply_platt(logit,m['calibration'][model]),cost

    @classmethod
    def train(cls,rows,models):
        if len(models)!=2 or len(set(models))!=2: raise ValueError('Two distinct candidate models required')
        base=cls.fit_base(rows,models);templates=sorted({r['template'] for r in rows})
        if len(templates)<3: raise ValueError('At least three training templates required for OOF calibration')
        logits={m:[] for m in models};labels={m:[] for m in models};folds=[]
        for held in templates:
            fit=cls.fit_base([r for r in rows if r['template']!=held],models)
            for r in rows:
                if r['template']==held:
                    p,_=fit.predict(r['public'],r['model']);logits[r['model']].append(math.log(p/(1-p)))
                    labels[r['model']].append(r['success'])
            folds.append({'held_out_template':held,'training_templates':[t for t in templates if t!=held]})
        unconstrained={m:fit_platt(logits[m],labels[m]) for m in models}
        base.model['calibration']=dict(unconstrained)
        for model,calibration in unconstrained.items():
            if calibration['kind']=='platt' and calibration['slope']<=0:
                base.model['calibration'][model]={'kind':'constant',
                    'probability':(sum(labels[model])+1)/(len(labels[model])+2),
                    'samples':len(labels[model]),
                    'note':'Non-positive OOF slope: category discrimination unsupported; use global training rate.'}
        base.model['unconstrained_calibration']=unconstrained
        base.model['calibration_folds']=folds;base.model['training_observations']=len(rows)
        return base

    def choose(self,public):
        values=[{'model':m,'predicted_success':p,'predicted_seconds':c}
                for m in self.model['models'] for p,c in [self.predict(public,m)]]
        best=max(v['predicted_success'] for v in values)
        eligible=[v for v in values if v['predicted_success']>=best-self.model['tolerance']]
        return min(eligible,key=lambda v:v['predicted_seconds'])['model'],values

    def save(self,path):
        path=Path(path)
        if path.exists(): raise ValueError('Preserve the fitted router; use a new directory')
        path.write_text(json.dumps(self.model,indent=2)+'\n')

    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.profile_directory);out=Path(args.output)
    if not out.resolve().is_relative_to(Path('results/v2').resolve()): parser.error('Write under results/v2')
    if out.exists(): parser.error('Existing fitted router is preserved')
    summary=json.loads((root/'profile_summary.json').read_text())
    if not summary['complete'] or not summary['suitable_pair']: parser.error('A complete suitable-pair profile is required')
    raw=(root/'profile.jsonl').read_bytes();rows=[json.loads(l) for l in raw.splitlines()]
    router=TaskRouter.train([r for r in rows if r['split']=='train'],list(summary['models']))
    router.model.update(profile_sha256=hashlib.sha256(raw).hexdigest(),
                        dataset_sha256=summary['dataset_sha256'],
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True);router.save(out)
    print('Fitted routing on',router.model['training_observations'],'clean training observations; validation/test excluded.')


if __name__=='__main__': main()
