"""Local, measured SQL-agent execution with bounded recovery."""
import json
import re
import math
import sqlite3
import time
import urllib.request
from copy import deepcopy
from pilot.environment import query, correct
from pilot.checks import verify
from study.data import database, public, report_from_result

SMALL='qwen3:0.6b'
LARGE='qwen3:1.7b'
ACTIONS=['accept','repair','escalate','verify','stop','retry']


def api(path,payload=None,timeout=90):
    req=urllib.request.Request('http://127.0.0.1:11434'+path,
        data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def lexical_checks(sql,contract,version='original'):
    """Fallible concern signals; never a proof of incorrectness."""
    upper=sql.upper()
    concerns=[]
    aggregate=contract['aggregation']
    if aggregate=='sum' and not re.search(r'\bSUM\s*\(',upper): concerns.append('requested sum not visibly aggregated')
    if version!='contract_v3' and aggregate=='average' and not re.search(r'\bAVG\s*\(',upper): concerns.append('requested average not visibly aggregated')
    if aggregate=='ratio' and '/' not in upper: concerns.append('requested ratio has no visible division')
    if contract.get('distinct') and 'DISTINCT' not in upper: concerns.append('distinctness not visibly expressed')
    if contract.get('grouped') and 'GROUP BY' not in upper: concerns.append('grouping not visibly expressed')
    start,end=contract.get('date_start'),contract.get('date_end_exclusive')
    if start and (start not in sql or end not in sql):
        concerns.append('requested date endpoints are not both explicit; equivalent date expressions may be valid')
    return concerns


class Session:
    def __init__(self,task,trial=0,fault='clean',max_calls=3,max_operations=6,verifier='original',
                 fresh_escalation=False,neutral_feedback=False,diversify_seed=False,small_model=SMALL,large_model=LARGE,
                 consistency_verification=False,consistency_k=3,initial_routing=False,
                 generation_cache=False,cache_directory='results/v2/generation_cache'):
        if verifier not in ('original','contract_v2','contract_v3'): raise ValueError('Unknown verifier version')
        if not isinstance(consistency_k,int) or not 2<=consistency_k<=5: raise ValueError('Consistency sample count must be 2 to 5')
        self.verifier=verifier
        self.fresh_escalation=fresh_escalation; self.neutral_feedback=neutral_feedback
        self.diversify_seed=diversify_seed
        self.small_model=small_model;self.large_model=large_model
        self.consistency_verification=consistency_verification;self.consistency_k=consistency_k
        self.initial_routing=initial_routing;self.verification_calls=0
        self.generation_cache=generation_cache;self.cache_directory=str(cache_directory);self.cache=None
        if generation_cache:
            from study.generation_cache import GenerationCache
            self.cache=GenerationCache(cache_directory)
        self.task=task; self.public=public(task); self.trial=trial; self.fault=fault
        self.max_calls=max_calls; self.max_operations=max_operations
        self.calls=0; self.operations=0; self.trace=[]; self.started=time.perf_counter()
        self.last_request_sha256=None
        self.state={'sql':'','output':None,'report':None,'error':None,'checks':[], 'model':small_model}

    def clone(self):
        copy=deepcopy(self)
        copy.trace=[]; copy.started=time.perf_counter()
        return copy

    def environment(self,seed,mutation=None):
        """Private dataset routing; corpus/schema identifiers never enter prompts."""
        if self.task.get('corpus_version','v1')=='v2':
            from study.data_v2 import database as v2_database
            return v2_database(seed,mutation,self.task.get('schema_variant','orders'))
        return database(seed,mutation=mutation)

    def generate(self,model,feedback='',sampling=None,cache_only=False):
        if self.calls>=self.max_calls:
            self.state['error']='model_call_budget'; return
        self.calls+=1
        system='You generate read-only SQLite SQL for a data-analysis agent. Return only a JSON object {"sql":"SELECT ..."}. No report and no commentary. Respect all task filters and null conventions. SQLite has no :: casts. Use SUM/AVG/COUNT and COALESCE as appropriate. For grouped output return label then value. For scalar output return one row in the declared report_keys order. /no_think'
        request_body={'model':model,'messages':[{'role':'system','content':system},
                       {'role':'user','content':json.dumps(self.public)+'\n'+feedback}],
                      'format':'json','stream':False,'think':False,'keep_alive':'5m',
                      'options':{'num_ctx':4096,'num_predict':384,'temperature':0.2,
                                 'seed':self.trial+self.calls if self.diversify_seed else self.trial}}
        if sampling is not None: request_body['options'].update(sampling)
        from study.generation_cache import request_sha256
        self.last_request_sha256=request_sha256(request_body)
        before=time.perf_counter()
        try:
            if self.cache is None:
                if cache_only: raise RuntimeError('Cache-only generation requires an enabled cache')
                response=api('/api/chat',request_body)
            else:
                try:
                    record,hit=self.cache.generate(request_body,lambda:api('/api/chat',request_body),cache_only)
                except ValueError as exc:
                    raise RuntimeError('Generation cache failed integrity validation') from exc
                response=record['response']
            text=response['message']['content']
            event={'kind':'generation','model':model,'content':text,'wall_seconds':time.perf_counter()-before,
                   'usage':{k:response.get(k,0) for k in ('load_duration','prompt_eval_duration','eval_duration','prompt_eval_count','eval_count')}}
            self.trace.append(event)
            if self.cache is not None:
                event.update(cache_hit=hit,request_sha256=record['request_sha256'],
                             recorded_wall_seconds=record['recorded_wall_seconds'],
                             effort_accounting='recorded model usage; database execution measured live')
            action=json.loads(text)
            if not isinstance(action,dict) or not isinstance(action.get('sql'),str):
                raise ValueError('Return an object containing a sql string')
            self.state={'sql':action['sql'],'output':None,'report':None,'error':None,'checks':[], 'model':model}
            self.execute()
        except (ValueError,KeyError,json.JSONDecodeError) as exc:
            self.state={'sql':'','output':None,'report':None,'error':'invalid_model_action: '+str(exc),'checks':[], 'model':model}
        # Connection failures propagate: the orchestrator must stop, not silently score a provider outage as model quality.

    def execute(self):
        if self.operations>=self.max_operations:
            self.state['error']='tool_operation_budget'; return
        self.operations+=1
        before=time.perf_counter()
        con,_=self.environment(self.task['seed'])
        try:
            if self.fault=='outage' or (self.fault=='transient' and self.operations==1):
                raise TimeoutError('tool_timeout')
            output=query(con,self.state['sql'])
            if self.verifier=='contract_v3' and any(isinstance(v,float) and not math.isfinite(v) for row in output['rows'] for v in row):
                self.state['output']=None
                raise ValueError('non-finite SQL result')
            self.state['output']=output
            self.state['report']=report_from_result(output,self.public['contract'])
            self.state['error']=None
        except (sqlite3.Error,ValueError,TimeoutError) as exc:
            self.state['error']=str(exc)
            self.state['report']=None
        finally:
            con.close()
            self.trace.append({'kind':'tool','sql':self.state['sql'],'output':self.state['output'],
                               'error':self.state['error'],'seconds':time.perf_counter()-before})

    def check(self):
        if self.state['error'] or self.state['output'] is None: return
        before=time.perf_counter()
        if self.verifier in ('contract_v2','contract_v3'):
            from study.verifier import contract_verify,contract_verify_v3
            checker=contract_verify_v3 if self.verifier=='contract_v3' else contract_verify
            evidence=checker(self.state['sql'],self.state['output'],self.task['seed'],self.public['contract'],self.environment)
        else:
            evidence=verify(self.state['sql'],self.state['output'],self.task['seed'],self.public['contract'],self.environment)
        concerns=lexical_checks(self.state['sql'],self.public['contract'],self.verifier)+evidence.get('concerns',[])
        evidence['concerns']=concerns
        evidence['flagged']=evidence['flagged'] or bool(concerns)
        evidence['seconds']=time.perf_counter()-before
        self.state['checks']=evidence
        self.trace.append({'kind':'verification',**evidence})
        if self.verifier=='contract_v3': self.trace[-1].update(sql=self.state['sql'],report=deepcopy(self.state['report']),error=self.state['error'])

    def check_consistency(self):
        """Sample three public-only queries with a separate bounded verification budget."""
        if 'verify_consistency' not in self.eligible(): raise ValueError('Consistency verification is ineligible')
        before=time.perf_counter();samples=[];outputs=[self.state['output']];cost=0.0
        for i in range(self.consistency_k):
            sample=Session(self.task,self.trial,fault='clean',max_calls=1,small_model=self.small_model,large_model=self.large_model,
                           verifier=self.verifier,generation_cache=self.generation_cache,cache_directory=self.cache_directory)
            seed=self.trial+self.calls+i+1
            sample.generate(self.state['model'],sampling={'temperature':.7,'seed':seed})
            self.verification_calls+=1;self.operations+=1;cost+=sample.effort()
            samples.append({'sql':sample.state['sql'],'output':sample.state['output'],'report':sample.state['report'],
                            'error':sample.state['error'],'seed':seed,'trace':sample.trace})
            outputs.append(sample.state['output'])
        from pilot.checks import equivalent
        distinct=[]
        for output in outputs:
            if not any((output is None and v is None) or (output is not None and v is not None and equivalent(output,v)) for v in distinct): distinct.append(output)
        agreement=sum(v is not None and equivalent(outputs[0],v) for v in outputs)/len(outputs)
        previous=self.state['checks'] if isinstance(self.state['checks'],dict) else {}
        evidence={**previous,'flagged':previous.get('flagged',False) or len(distinct)>1,
                  'agreement_fraction':agreement,'distinct_results':len(distinct),'consistency_done':True,'samples':samples,
                  'interpretation':'Self-consistency is a hypothesis: a model can agree on a wrong answer.'}
        self.state['checks']=evidence
        self.trace.append({'kind':'verification','method':'consistency',**evidence,'seconds':time.perf_counter()-before,
                           'effort_seconds':cost,'sql':self.state['sql'],'report':deepcopy(self.state['report']),'error':self.state['error']})

    def feedback(self):
        checks=self.state['checks'] if isinstance(self.state['checks'],dict) else {}
        if (self.neutral_feedback and self.state['error'] is None
                and not lexical_checks(self.state['sql'],self.public['contract'],self.verifier)
                and not checks.get('flagged')):
            return 'Solve the task again from the request and contract.'
        return 'Previous SQL: '+self.state['sql']+'\nObserved error: '+str(self.state['error'])+'\nConcerns: '+json.dumps(lexical_checks(self.state['sql'],self.public['contract'],self.verifier))+'\nVerification: '+json.dumps(self.state['checks'])[:1800]+'\nGenerate a corrected SQLite query.'

    def eligible(self):
        if self.initial_routing and self.calls==0: return ['generate_small','route_large']
        actions=['accept','stop'] if self.state['error'] is None else ['stop']
        if self.calls<self.max_calls: actions+=['repair','escalate']
        if (self.fresh_escalation and self.calls<self.max_calls and self.state['error'] is None
                and (self.state['model']!=self.large_model or self.diversify_seed)):
            actions.append('escalate_fresh')
        if self.state['error'] is None and self.state['output'] is not None and not self.state['checks']: actions+=['verify']
        checks=self.state['checks'] if isinstance(self.state['checks'],dict) else {}
        if (self.consistency_verification and self.state['error'] is None and self.state['output'] is not None
                and not checks.get('consistency_done') and self.verification_calls==0
                and self.operations+self.consistency_k<=self.max_operations): actions.append('verify_consistency')
        if self.state['error']=='tool_timeout' and self.fault!='outage' and self.operations<self.max_operations:
            actions+=['retry']
        return actions

    def apply(self,action):
        if action not in self.eligible(): raise ValueError('Ineligible controller action: '+action)
        self.trace.append({'kind':'decision','action':action,'calls_before':self.calls,'operations_before':self.operations})
        if action=='repair': self.generate(self.small_model,self.feedback())
        elif action=='escalate': self.generate(self.large_model,self.feedback())
        elif action=='escalate_fresh': self.generate(self.large_model,'')
        elif action=='retry': self.execute()
        elif action=='verify': self.check()
        elif action=='verify_consistency': self.check_consistency()
        elif action=='generate_small': self.generate(self.small_model,'')
        elif action=='route_large': self.generate(self.large_model,'')
        elif action=='stop': self.state['report']=None

    def effort(self):
        total=0.0
        for e in self.trace:
            if e['kind']=='generation':
                total+=sum(e['usage'].get(k,0) or 0 for k in ('load_duration','prompt_eval_duration','eval_duration'))/1e9
            elif e['kind'] in ('tool','verification'): total+=e.get('effort_seconds',e['seconds'])
        return total

    def result(self):
        result={'task_id':self.task['id'],'template':self.task['template'],'split':self.task['split'],
                'trial':self.trial,'fault':self.fault,'success':correct(self.state['report'],self.task['expected']),
                'report':self.state['report'],'expected':self.task['expected'],'trace':self.trace,
                'effort_seconds':self.effort(),'wall_seconds':time.perf_counter()-self.started,
                'calls':self.calls+self.verification_calls,'operations':self.operations,'public':self.public}
        if self.generation_cache:
            generations=[e for e in self.trace if e['kind']=='generation']
            generations += [g for e in self.trace if e.get('method')=='consistency' for s in e['samples']
                            for g in s['trace'] if g['kind']=='generation']
            result.update(generation_cache=True,cache_directory=self.cache_directory,
                          logical_model_calls=len(generations),live_model_calls=sum(not e['cache_hit'] for e in generations),
                          cache_hits=sum(e['cache_hit'] for e in generations),
                          replayed_model_effort_seconds=sum(sum(e['usage'].get(k,0) or 0 for k in
                              ('load_duration','prompt_eval_duration','eval_duration'))/1e9 for e in generations if e['cache_hit']),
                          effort_accounting='recorded generation durations charged on hits and misses; live tool/probe effort')
        return result


def fixed_continuation(session):
    """Identical continuation after training branches; no evaluator feedback."""
    if session.state['error']=='tool_timeout' and 'retry' in session.eligible(): session.apply('retry')
    if session.state['error'] and 'escalate' in session.eligible(): session.apply('escalate')
    if session.state['error'] is None and session.state['checks'] and session.state['checks']['flagged']:
        escalation='escalate_fresh' if 'escalate_fresh' in session.eligible() else 'escalate'
        if escalation in session.eligible(): session.apply(escalation)
    return session.result()
