"""Local, measured SQL-agent execution with bounded recovery."""
import json
import re
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


def lexical_checks(sql,contract):
    """Fallible concern signals; never a proof of incorrectness."""
    upper=sql.upper()
    concerns=[]
    aggregate=contract['aggregation']
    if aggregate=='sum' and not re.search(r'\bSUM\s*\(',upper): concerns.append('requested sum not visibly aggregated')
    if aggregate=='average' and not re.search(r'\bAVG\s*\(',upper): concerns.append('requested average not visibly aggregated')
    if aggregate=='ratio' and '/' not in upper: concerns.append('requested ratio has no visible division')
    if contract.get('distinct') and 'DISTINCT' not in upper: concerns.append('distinctness not visibly expressed')
    if contract.get('grouped') and 'GROUP BY' not in upper: concerns.append('grouping not visibly expressed')
    start,end=contract.get('date_start'),contract.get('date_end_exclusive')
    if start and (start not in sql or end not in sql):
        concerns.append('requested date endpoints are not both explicit; equivalent date expressions may be valid')
    return concerns


class Session:
    def __init__(self,task,trial=0,fault='clean',max_calls=3,max_operations=6):
        self.task=task; self.public=public(task); self.trial=trial; self.fault=fault
        self.max_calls=max_calls; self.max_operations=max_operations
        self.calls=0; self.operations=0; self.trace=[]; self.started=time.perf_counter()
        self.state={'sql':'','output':None,'report':None,'error':None,'checks':[], 'model':SMALL}

    def clone(self):
        copy=deepcopy(self)
        copy.trace=[]; copy.started=time.perf_counter()
        return copy

    def generate(self,model,feedback=''):
        if self.calls>=self.max_calls:
            self.state['error']='model_call_budget'; return
        self.calls+=1
        system='You generate read-only SQLite SQL for a data-analysis agent. Return only a JSON object {"sql":"SELECT ..."}. No report and no commentary. Respect all task filters and null conventions. SQLite has no :: casts. Use SUM/AVG/COUNT and COALESCE as appropriate. For grouped output return label then value. For scalar output return one row in the declared report_keys order. /no_think'
        request_body={'model':model,'messages':[{'role':'system','content':system},
                       {'role':'user','content':json.dumps(self.public)+'\n'+feedback}],
                      'format':'json','stream':False,'think':False,'keep_alive':'5m',
                      'options':{'num_ctx':4096,'num_predict':384,'temperature':0.2,'seed':self.trial}}
        before=time.perf_counter()
        try:
            response=api('/api/chat',request_body)
            text=response['message']['content']
            event={'kind':'generation','model':model,'content':text,'wall_seconds':time.perf_counter()-before,
                   'usage':{k:response.get(k,0) for k in ('load_duration','prompt_eval_duration','eval_duration','prompt_eval_count','eval_count')}}
            self.trace.append(event)
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
        con,_=database(self.task['seed'])
        try:
            if self.fault=='outage' or (self.fault=='transient' and self.operations==1):
                raise TimeoutError('tool_timeout')
            output=query(con,self.state['sql'])
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
        evidence=verify(self.state['sql'],self.state['output'],self.task['seed'],self.public['contract'],database)
        concerns=lexical_checks(self.state['sql'],self.public['contract'])
        evidence['concerns']=concerns
        evidence['flagged']=evidence['flagged'] or bool(concerns)
        evidence['seconds']=time.perf_counter()-before
        self.state['checks']=evidence
        self.trace.append({'kind':'verification',**evidence})

    def feedback(self):
        return 'Previous SQL: '+self.state['sql']+'\nObserved error: '+str(self.state['error'])+'\nConcerns: '+json.dumps(lexical_checks(self.state['sql'],self.public['contract']))+'\nVerification: '+json.dumps(self.state['checks'])[:1800]+'\nGenerate a corrected SQLite query.'

    def eligible(self):
        actions=['accept','stop'] if self.state['error'] is None else ['stop']
        if self.calls<self.max_calls: actions+=['repair','escalate']
        if self.state['error'] is None and self.state['output'] is not None and not self.state['checks']: actions+=['verify']
        if self.state['error']=='tool_timeout' and self.fault!='outage' and self.operations<self.max_operations:
            actions+=['retry']
        return actions

    def apply(self,action):
        if action not in self.eligible(): raise ValueError('Ineligible controller action: '+action)
        self.trace.append({'kind':'decision','action':action,'calls_before':self.calls,'operations_before':self.operations})
        if action=='repair': self.generate(SMALL,self.feedback())
        elif action=='escalate': self.generate(LARGE,self.feedback())
        elif action=='retry': self.execute()
        elif action=='verify': self.check()
        elif action=='stop': self.state['report']=None

    def effort(self):
        total=0.0
        for e in self.trace:
            if e['kind']=='generation':
                total+=sum(e['usage'].get(k,0) or 0 for k in ('load_duration','prompt_eval_duration','eval_duration'))/1e9
            elif e['kind'] in ('tool','verification'): total+=e['seconds']
        return total

    def result(self):
        return {'task_id':self.task['id'],'template':self.task['template'],'split':self.task['split'],
                'trial':self.trial,'fault':self.fault,'success':correct(self.state['report'],self.task['expected']),
                'report':self.state['report'],'expected':self.task['expected'],'trace':self.trace,
                'effort_seconds':self.effort(),'wall_seconds':time.perf_counter()-self.started,
                'calls':self.calls,'operations':self.operations,'public':self.public}


def fixed_continuation(session):
    """Identical continuation after training branches; no evaluator feedback."""
    if session.state['error']=='tool_timeout' and 'retry' in session.eligible(): session.apply('retry')
    if session.state['error'] and 'escalate' in session.eligible(): session.apply('escalate')
    if session.state['checks'] and session.state['checks']['flagged'] and 'escalate' in session.eligible(): session.apply('escalate')
    return session.result()
