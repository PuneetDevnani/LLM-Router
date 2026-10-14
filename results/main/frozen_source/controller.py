"""Small offline success/effort predictors; no language-model fine tuning."""
import json
import math
from pathlib import Path
import numpy as np
from study.runtime import ACTIONS, lexical_checks

FEATURES=['prompt_length','sql_length','sum','average','count','ratio','grouped','distinct','date_required',
          'exclude_cancelled','zero_nulls','execution_error','shape_error','row_count','null_count','concern_count',
          'current_large','remaining_calls','completed_check','flagged_check','sql_date_literals','join_mentions']


def features(session,history=True):
    p=session.public; c=p['contract']; s=session.state; output=s['output'] or {'rows':[]}
    check=s['checks'] if isinstance(s['checks'],dict) else {}
    sql=s['sql']; a=c['aggregation']
    return [math.log1p(len(p['prompt'])),math.log1p(len(sql)),*[float(a==v) for v in ('sum','average','count','ratio')],
            float(c['grouped']),float(c['distinct']),float('date_start' in c),float(c['excludes_cancelled']),float(c['null_discount_as_zero']),
            float(bool(s['error'])),float(s['error'] is not None and 'report' in s['error'].lower()),
            math.log1p(len(output['rows'])),math.log1p(sum(v is None for row in output['rows'] for v in row)),
            float(len(lexical_checks(sql,c))),float(s['model'].endswith('1.7b')),
            float(session.max_calls-session.calls) if history else 0,
            float(bool(check)) if history else 0,float(check.get('flagged',False)) if history else 0,
            float(sql.count('2026-')+sql.count('2023-')),float(sql.upper().count('JOIN'))]


def encode(f,action):
    # Action-specific coefficients provide a transparent small multi-action predictor.
    block=np.asarray([1.0,*f],dtype=float)
    return np.concatenate([block if a==action else np.zeros_like(block) for a in ACTIONS])


class Controller:
    def __init__(self,model): self.model=model

    @classmethod
    def train(cls,rows):
        X=np.asarray([encode(r['features'],r['action']) for r in rows]); y=np.asarray([r['success'] for r in rows],float)
        cost=np.asarray([math.log1p(r['remaining_effort_seconds']) for r in rows])
        mean=X.mean(axis=0); scale=X.std(axis=0); scale[scale<0.01]=1
        Z=(X-mean)/scale
        w=np.zeros(Z.shape[1]); intercept=0.0
        for _ in range(1800):
            prediction=1/(1+np.exp(-np.clip(Z@w+intercept,-30,30)))
            error=prediction-y
            w-=0.04*(Z.T@error/len(y)+0.03*w)
            intercept-=0.04*error.mean()
        augmented=np.column_stack([np.ones(len(Z)),Z]); penalty=np.eye(augmented.shape[1])*4.0; penalty[0,0]=0
        wc=np.linalg.solve(augmented.T@augmented+penalty,augmented.T@cost)
        return cls({'feature_names':FEATURES,'actions':ACTIONS,'mean':mean.tolist(),'scale':scale.tolist(),
                    'success_weights':w.tolist(),'success_intercept':intercept,'cost_weights':wc.tolist(),
                    'samples':len(rows),'reference_seconds':max(1,float(np.median([r['remaining_effort_seconds'] for r in rows if r['remaining_effort_seconds']>0]))),
                    'method':'L2 logistic regression and ridge regression on log1p effort'})

    def predict(self,f,action):
        if action=='stop': return 0.0,0.0
        m=self.model; z=(encode(f,action)-np.asarray(m['mean']))/np.asarray(m['scale'])
        p=1/(1+math.exp(-float(np.clip(z@np.asarray(m['success_weights'])+m['success_intercept'],-30,30))))
        c=max(0.0,math.expm1(float(np.clip(np.r_[1,z]@np.asarray(m['cost_weights']),0,8))))
        if action=='accept': c=0.0
        return p,c

    def choose(self,session,penalty=0.1,mode='joint'):
        f=features(session,history=mode!='no_history'); options=session.eligible()
        if mode=='model_only': options=[a for a in options if a!='verify']
        values=[]
        for a in options:
            p,c=self.predict(f,a)
            score=p-(0 if mode=='no_effort' else penalty*c/self.model['reference_seconds'])
            values.append({'action':a,'predicted_success':p,'predicted_seconds':c,'score':score})
        action=max(values,key=lambda v:v['score'])['action']
        return action,values

    def save(self,path): Path(path).write_text(json.dumps(self.model,indent=2)+'\n')

    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))
