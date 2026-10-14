"""Small offline success/effort predictors; no language-model fine tuning."""
import json
import math
from pathlib import Path
import numpy as np
from study.runtime import ACTIONS, LARGE, lexical_checks

FEATURES=['prompt_length','sql_length','sum','average','count','ratio','grouped','distinct','date_required',
          'exclude_cancelled','zero_nulls','execution_error','shape_error','row_count','null_count','concern_count',
          'current_large','remaining_calls','completed_check','flagged_check','sql_date_literals','join_mentions']
REVISION_FEATURES=FEATURES+['agreement_fraction','distinct_results']


def features(session,history=True,feature_names=None):
    p=session.public; c=p['contract']; s=session.state; output=s['output'] or {'rows':[]}
    check=s['checks'] if isinstance(s['checks'],dict) else {}
    sql=s['sql']; a=c['aggregation']
    values=[math.log1p(len(p['prompt'])),math.log1p(len(sql)),*[float(a==v) for v in ('sum','average','count','ratio')],
            float(c['grouped']),float(c['distinct']),float('date_start' in c),float(c['excludes_cancelled']),float(c['null_discount_as_zero']),
            float(bool(s['error'])),float(s['error'] is not None and 'report' in s['error'].lower()),
            math.log1p(len(output['rows'])),math.log1p(sum(v is None for row in output['rows'] for v in row)),
            float(len(lexical_checks(sql,c,session.verifier))),float(s['model']==session.large_model),
            float(session.max_calls-session.calls) if history else 0,
            float(bool(check)) if history else 0,float(check.get('flagged',False)) if history else 0,
            float(sql.count('2026-')+sql.count('2023-')),float(sql.upper().count('JOIN'))]
    names=FEATURES if feature_names is None else feature_names
    if names==FEATURES: return values
    if names!=REVISION_FEATURES: raise ValueError('Unknown controller feature vocabulary')
    return values+[float(check.get('agreement_fraction',0)) if history else 0,
                   math.log1p(check.get('distinct_results',0)) if history else 0]


def training_feature_names(rows):
    if not rows: raise ValueError('Training observations required')
    names=rows[0].get('feature_names',FEATURES)
    if names not in (FEATURES,REVISION_FEATURES) or any(r.get('feature_names',FEATURES)!=names for r in rows):
        raise ValueError('Mixed or unsupported feature vocabularies')
    return names


def encode(f,action,actions=ACTIONS):
    # Action-specific coefficients provide a transparent small multi-action predictor.
    block=np.asarray([1.0,*f],dtype=float)
    if action not in actions: raise ValueError('Unknown controller action: '+action)
    return np.concatenate([block if a==action else np.zeros_like(block) for a in actions])


def action_list(rows,actions=None):
    """Freeze a model's vocabulary independently of the legacy global ACTIONS."""
    result=list(actions) if actions is not None else list(ACTIONS)
    additions=('escalate_fresh','verify_consistency','generate_small','route_large')
    known=set(ACTIONS)|set(additions)
    if actions is None:
        result.extend(a for a in additions if any(r['action']==a for r in rows))
    if not result or len(set(result))!=len(result) or set(result)-known:
        raise ValueError('Invalid controller action vocabulary')
    if any(r['action'] not in result for r in rows):
        raise ValueError('Training action outside controller vocabulary')
    return result


class Controller:
    def __init__(self,model):
        self.model=model
        self.actions=list(model.get('actions',ACTIONS))

    @classmethod
    def train(cls,rows,actions=None):
        actions=action_list(rows,actions)
        names=training_feature_names(rows)
        X=np.asarray([encode(r['features'],r['action'],actions) for r in rows]); y=np.asarray([r['success'] for r in rows],float)
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
        return cls({'feature_names':names,'actions':actions,'mean':mean.tolist(),'scale':scale.tolist(),
                    'success_weights':w.tolist(),'success_intercept':intercept,'cost_weights':wc.tolist(),
                    'samples':len(rows),'reference_seconds':max(1,float(np.median([r['remaining_effort_seconds'] for r in rows if r['remaining_effort_seconds']>0]))),
                    'method':'L2 logistic regression and ridge regression on log1p effort'})

    def predict(self,f,action):
        if action=='stop': return 0.0,0.0
        m=self.model; z=(encode(f,action,self.actions)-np.asarray(m['mean']))/np.asarray(m['scale'])
        p=1/(1+math.exp(-float(np.clip(z@np.asarray(m['success_weights'])+m['success_intercept'],-30,30))))
        c=max(0.0,math.expm1(float(np.clip(np.r_[1,z]@np.asarray(m['cost_weights']),0,8))))
        if action=='accept': c=0.0
        return p,c

    def choose(self,session,penalty=0.1,mode='joint',
               accept_threshold=0.0,wrong_penalty=0.0,abstain_cost=0.0):
        """Extract observable features, then apply the shared selection rule."""
        names=self.model.get('feature_names',FEATURES)
        options={} if names==FEATURES else {'feature_names':names}
        return self.choose_from(features(session,history=mode!='no_history',**options),session.eligible(),
                                penalty,mode,accept_threshold,wrong_penalty,abstain_cost)

    def choose_from(self,f,options,penalty=0.1,mode='joint',
                    accept_threshold=0.0,wrong_penalty=0.0,abstain_cost=0.0):
        """Select an action with optional acceptance gating and terminal losses.

        Below accept_threshold, exclude both accept and stop when an eligible
        repair, escalation, verification, or retry remains. Otherwise allow
        terminal choices, including acceptance below the threshold. Retry is
        never removed by this gate; it is a non-terminal tool re-execution.
        wrong_penalty charges expected loss for a wrong accepted report;
        abstain_cost charges explicit stopping. Select settings using validation
        data only. Zero defaults preserve the original selection rule.
        """
        options=[a for a in options if a in self.actions]
        if mode=='model_only': options=[a for a in options if a not in ('verify','verify_consistency')]
        if not options: raise ValueError('No eligible actions supported by this controller')
        values=[]
        for a in options:
            p,c=self.predict(f,a)
            score=p-(0 if mode=='no_effort' else penalty*c/self.model['reference_seconds'])
            if a=='accept':
                score-=wrong_penalty*(1-p)
            elif a=='stop':
                score-=abstain_cost
            values.append({'action':a,'predicted_success':p,'predicted_seconds':c,'score':score})
        accept=next((v for v in values if v['action']=='accept'),None)
        gated=accept is not None and accept['predicted_success']<accept_threshold
        recoverable=[v for v in values if v['action'] in ('repair','escalate','escalate_fresh','verify','verify_consistency','retry','generate_small','route_large')]
        if gated and recoverable:
            values_for_choice=recoverable
            for v in values:
                v['gated']=v['action'] in ('accept','stop')
        else:
            values_for_choice=values
        action=max(values_for_choice,key=lambda v:v['score'])['action']
        return action,values

    def save(self,path): Path(path).write_text(json.dumps(self.model,indent=2)+'\n')

    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))


# Explicit alias retains the original numerical fit and prediction path.
LogisticV1Controller = Controller


class TableController(Controller):
    """Smooth sparse cells toward pooled training rates for the same action."""
    KEY = ('execution_error', 'current_large', 'flagged_check')

    @classmethod
    def train(cls,rows,prior=2.0,actions=None):
        from collections import defaultdict
        if not rows: raise ValueError('Collect training branches first')
        actions=action_list(rows,actions)
        names=training_feature_names(rows)
        if not math.isfinite(prior) or prior<=0: raise ValueError('prior must be positive and finite')
        idx=[names.index(k) for k in cls.KEY]
        cell=defaultdict(lambda:[0,0.0,0.0]); act=defaultdict(lambda:[0,0.0,0.0])
        for r in rows:
            if r.get('split','train')!='train': raise ValueError('Table training uses training rows only')
            k=r['action']+'|'+''.join(str(int(r['features'][i]>.5)) for i in idx)
            for d,key in ((cell,k),(act,r['action'])):
                d[key][0]+=1;d[key][1]+=r['success'];d[key][2]+=r['remaining_effort_seconds']
        pos=[r['remaining_effort_seconds'] for r in rows if r['remaining_effort_seconds']>0]
        return cls({'cells':dict(cell),'actions_stats':dict(act),'prior':prior,'indices':idx,'actions':actions,'feature_names':names,
                    'reference_seconds':max(1.0,float(np.median(pos))) if pos else 1.0,
                    'samples':len(rows),'controller_type':'table','method':'smoothed empirical table'})

    def predict(self,f,action):
        if action=='stop': return 0.0,0.0
        m=self.model;k=action+'|'+''.join(str(int(f[i]>.5)) for i in m['indices'])
        na,sa,ea=m['actions_stats'].get(action,[0,0.0,0.0])
        pa,ca=(sa/na,ea/na) if na else (0.0,0.0)
        n,s,e=m['cells'].get(k,[0,0.0,0.0]);prior=m['prior']
        p=(s+prior*pa)/(n+prior);c=(e+prior*ca)/(n+prior)
        return p,0.0 if action=='accept' else c


def _sigmoid(value):
    return 1.0/(1.0+np.exp(-np.clip(value,-30,30)))


def encode_v2(f,action,mean,scale,actions=ACTIONS):
    """Standardize base features first; inactive interaction blocks stay zero."""
    z=(np.asarray(f,dtype=float)-np.asarray(mean))/np.asarray(scale)
    one_hot=np.asarray([float(a==action) for a in actions])
    if not one_hot.any(): raise ValueError('Unknown action: '+action)
    return np.r_[z,one_hot,*[z if a==action else np.zeros_like(z) for a in actions]]


def _training_rows(rows,actions=None):
    if not rows: raise ValueError('Training rows required')
    actions=action_list(rows,actions)
    names=training_feature_names(rows)
    for r in rows:
        if r.get('split')!='train' or not r.get('template'):
            raise ValueError('Template-grouped training rows only; validation/test labels cannot train the model')
        f=np.asarray(r['features'],dtype=float)
        if f.shape!=(len(names),) or not np.all(np.isfinite(f)):
            raise ValueError('Invalid training features')
        if r['action'] not in actions or r['success'] not in (False,True,0,1):
            raise ValueError('Invalid action/label')
        if not math.isfinite(r['remaining_effort_seconds']) or r['remaining_effort_seconds']<0:
            raise ValueError('Invalid effort label')


def fit_platt(logits,labels):
    """Fit a calibrator to cross-fitted logits, never in-sample base predictions."""
    x=np.asarray(logits,dtype=float); y=np.asarray(labels,dtype=float)
    if x.ndim!=1 or not len(x) or x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isin(y,[0,1])):
        raise ValueError('Matched finite logits and labels required')
    rate=(float(y.sum())+1)/(len(y)+2)
    if len(set(y.tolist()))<2:
        return {'kind':'constant','probability':rate,'samples':len(y),
                'note':'Single-class calibration data; discrimination/calibration are not established.'}
    mean=float(x.mean());scale=max(float(x.std()),1e-3)
    X=np.column_stack([(x-mean)/scale,np.ones(len(x))])
    w=np.array([1.0,math.log(rate/(1-rate))])
    for _ in range(1800):
        gradient=X.T@(_sigmoid(X@w)-y)/len(y)+np.array([.01*w[0],0.0])
        w-=.05*gradient
    return {'kind':'platt','slope':float(w[0]/scale),
            'intercept':float(w[1]-w[0]*mean/scale),'samples':len(y)}


def apply_platt(logit,calibration):
    if calibration['kind']=='constant': return calibration['probability']
    if calibration['kind']=='identity': return float(_sigmoid(logit))
    return float(_sigmoid(calibration['slope']*logit+calibration['intercept']))


class LogisticV2Controller(Controller):
    """Shared features and shrunk action interactions with cross-fitted Platt scaling."""
    @classmethod
    def fit_base(cls,rows,actions=None):
        actions=action_list(rows,actions); _training_rows(rows,actions)
        excluded_stop_rows=sum(r['action']=='stop' for r in rows)
        rows=[r for r in rows if r['action']!='stop']
        if not rows: raise ValueError('Non-stop training rows required')
        F=np.asarray([r['features'] for r in rows],dtype=float)
        mean=F.mean(axis=0);scale=F.std(axis=0);scale[scale<.01]=1.0
        X=np.asarray([encode_v2(r['features'],r['action'],mean,scale,actions) for r in rows])
        y=np.asarray([r['success'] for r in rows],dtype=float)
        interaction_start=F.shape[1]+len(actions)
        regularization=np.full(X.shape[1],.03);regularization[interaction_start:]=.3
        w=np.zeros(X.shape[1]);b=math.log((y.sum()+1)/(len(y)-y.sum()+1))
        for _ in range(1800):
            error=_sigmoid(X@w+b)-y
            w-=.04*(X.T@error/len(y)+regularization*w);b-=.04*error.mean()
        augmented=np.column_stack([np.ones(len(X)),X])
        ridge=np.full(augmented.shape[1],4.0);ridge[0]=0;ridge[interaction_start+1:]=40.0
        cost=np.log1p([r['remaining_effort_seconds'] for r in rows])
        wc=np.linalg.solve(augmented.T@augmented+np.diag(ridge),augmented.T@cost)
        positive=[r['remaining_effort_seconds'] for r in rows if r['remaining_effort_seconds']>0]
        return cls({'controller_type':'logistic_v2','feature_names':training_feature_names(rows),'actions':actions,
                    'mean':mean.tolist(),'scale':scale.tolist(),'success_weights':w.tolist(),
                    'success_intercept':float(b),'cost_weights':wc.tolist(),
                    'regularization':regularization.tolist(),'ridge_regularization':ridge.tolist(),
                    'calibration':{'kind':'identity'},'samples':len(rows),'excluded_stop_rows':excluded_stop_rows,
                    'reference_seconds':max(1.0,float(np.median(positive))) if positive else 1.0,
                    'method':'base-feature scaling, shared/action/interactions, stronger interaction shrinkage, OOF Platt'})

    def raw_logit(self,f,action):
        m=self.model; x=encode_v2(f,action,m['mean'],m['scale'],m['actions'])
        return float(x@np.asarray(m['success_weights'])+m['success_intercept'])

    @classmethod
    def train(cls,rows,actions=None):
        actions=action_list(rows,actions); _training_rows(rows,actions)
        excluded_stop_rows=sum(r['action']=='stop' for r in rows)
        rows=[r for r in rows if r['action']!='stop']
        if not rows: raise ValueError('Non-stop training rows required')
        templates=sorted({r['template'] for r in rows})
        if len(templates)<2: raise ValueError('OOF calibration needs at least two training templates')
        logits=np.empty(len(rows));folds=[]
        for held_out in templates:
            indices=[i for i,r in enumerate(rows) if r['template']==held_out]
            training=[r for r in rows if r['template']!=held_out]
            base=cls.fit_base(training,actions=actions)
            for i in indices: logits[i]=base.raw_logit(rows[i]['features'],rows[i]['action'])
            folds.append({'held_out_template':held_out,'training_templates':sorted({r['template'] for r in training}),
                          'held_out_rows':len(indices)})
        controller=cls.fit_base(rows,actions=actions)
        controller.model['excluded_stop_rows']=excluded_stop_rows
        controller.model['calibration']=fit_platt(logits,[r['success'] for r in rows])
        controller.model['calibration_folds']=folds
        return controller

    def predict(self,f,action):
        if action=='stop': return 0.0,0.0
        m=self.model;x=encode_v2(f,action,m['mean'],m['scale'],m['actions'])
        p=apply_platt(self.raw_logit(f,action),m['calibration'])
        c=max(0.0,math.expm1(float(np.clip(np.r_[1,x]@np.asarray(m['cost_weights']),0,8))))
        return p,0.0 if action=='accept' else c


CONTROLLER_TYPES={'logistic_v1':Controller,'table':TableController,'logistic_v2':LogisticV2Controller}


def leave_one_template_out_cv(rows,controller_types=('logistic_v1','table','logistic_v2')):
    """Outer template holdout; v2 calibration uses inner folds of training only."""
    _training_rows(rows);templates=sorted({r['template'] for r in rows if r['action']!='stop'});actions=action_list(rows)
    if len(templates)<3: raise ValueError('Nested template CV requires at least three training templates')
    result={}
    for name in controller_types:
        cls=CONTROLLER_TYPES[name];folds=[]
        for held_out in templates:
            train=[r for r in rows if r['template']!=held_out]
            test=[r for r in rows if r['template']==held_out and r['action']!='stop']
            model=cls.train(train,actions=actions); predictions=[model.predict(r['features'],r['action']) for r in test]
            y=np.asarray([r['success'] for r in test],dtype=float)
            p=np.clip([p for p,c in predictions],1e-9,1-1e-9)
            costs=np.asarray([c for p,c in predictions])
            effort=np.asarray([r['remaining_effort_seconds'] for r in test])
            folds.append({'held_out_template':held_out,'training_templates':sorted({r['template'] for r in train}),
                          'rows':len(test),'positives':int(y.sum()),'brier':float(np.mean((p-y)**2)),
                          'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
                          'effort_mae_seconds':float(np.mean(abs(costs-effort)))})
        result[name]={'folds':folds,'rows':sum(f['rows'] for f in folds),
                      'excluded_stop_rows':sum(r['action']=='stop' for r in rows),
                      'mean_template_brier':float(np.mean([f['brier'] for f in folds])),
                      'mean_template_log_loss':float(np.mean([f['log_loss'] for f in folds])),
                      'mean_template_effort_mae_seconds':float(np.mean([f['effort_mae_seconds'] for f in folds])),
                      'scope':'training-only nested leave-one-template-out; not a deployment success estimate'}
    return result
