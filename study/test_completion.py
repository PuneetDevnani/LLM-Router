"""Development-only probe, sampling and controller vocabulary checks."""
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from study.data_v2 import corpus,REFERENCE_SQL,database
from study.runtime import Session,SMALL,LARGE,fixed_continuation,lexical_checks
from study.controller import FEATURES,REVISION_FEATURES,features,training_feature_names,Controller,TableController,LogisticV2Controller
from study.verifier import scale_eligible,scale_amounts,duplicate_orders,transformed_probe


class CompletionTests(unittest.TestCase):
    def task(self,name):
        return next(t for t in corpus() if t['template']==name and t['seed']==0 and t['id'].endswith('w0'))

    def checked(self,name,sql):
        s=Session(self.task(name),verifier='contract_v3');s.state['sql']=sql;s.execute();s.check();return s

    def test_scale_and_average_probes_accept_alternatives_and_detect_bias(self):
        for name in ('net_total','average_net','region_total','region_net','january'):
            s=self.checked(name,REFERENCE_SQL[name]);self.assertFalse(s.state['checks']['flagged'],name)
        sql=REFERENCE_SQL['average_net'].replace('AVG(amount-COALESCE(discount,0))','1.0*SUM(amount-IFNULL(discount,0))/COUNT(*)')
        self.assertFalse(self.checked('average_net',sql).state['checks']['flagged'])
        wrong=REFERENCE_SQL['average_net'].replace('AVG(','SUM(')
        evidence=self.checked('average_net',wrong).state['checks']
        self.assertTrue(any(e['check']=='duplicate_average_population' and e['status']=='violation' for e in evidence['evidence']))
        biased='SELECT "SUM(amount-COALESCE(discount,0))"+1 FROM ('+REFERENCE_SQL['net_total']+')'
        evidence=self.checked('net_total',biased).state['checks']
        self.assertTrue(any(e['check']=='monetary_scale' and e['status']=='violation' for e in evidence['evidence']))
        self.assertFalse(self.checked('missing_count',REFERENCE_SQL['missing_count'].replace('COUNT(*)','SUM(1)')).state['checks']['flagged'])
        self.assertTrue(lexical_checks(sql,self.task('average_net')['contract']))
        self.assertFalse(lexical_checks(sql,self.task('average_net')['contract'],'contract_v3'))

    def test_scale_threshold_and_shape_guards_and_inconclusive_probe(self):
        for name in ('large_net','summary'):
            task=self.task(name);s=Session(task);s.state['sql']=REFERENCE_SQL[name];s.execute()
            self.assertFalse(scale_eligible(s.state['sql'],s.state['output'],task['contract']))
        s=self.checked('net_total',REFERENCE_SQL['net_total']);c=deepcopy(s.public['contract']);c['monetary_threshold']=5000
        self.assertFalse(scale_eligible(s.state['sql'],s.state['output'],c))
        self.assertFalse(scale_eligible('SELECT SUM(amount) FROM orders WHERE amount>=5000',s.state['output'],s.public['contract']))
        evidence={'evidence':[]}
        with patch('study.verifier.query',side_effect=ValueError('probe failure')):
            transformed_probe(evidence,'failure',s.state['sql'],0,database,scale_amounts,s.state['output'])
        self.assertEqual(evidence['evidence'][0]['status'],'inconclusive')
        con,_=database(0);n=con.execute('SELECT COUNT(*) FROM orders').fetchone()[0]
        duplicate_orders(con);self.assertEqual(con.execute('SELECT COUNT(*) FROM orders').fetchone()[0],2*n);con.close()
        c=deepcopy(self.task('completion_rate')['contract']);c['valid_range']=[0,100]
        task=deepcopy(self.task('completion_rate'));task['contract']=c
        s=Session(task,verifier='contract_v3');s.state['sql']='SELECT 150';s.execute();s.check()
        self.assertTrue(any(e['check']=='public_ratio_range' and e['status']=='violation' for e in s.state['checks']['evidence']))

    def test_every_development_reference_remains_unflagged(self):
        for task in corpus():
            if task['split'] not in ('train','validation') or not task['id'].endswith('w0'): continue
            s=Session(task,verifier='contract_v3');s.state['sql']=REFERENCE_SQL[task['template']];s.execute();s.check()
            self.assertFalse(s.state['checks']['flagged'],task['id'])
        s=Session(self.task('net_total'),verifier='contract_v3');s.state['sql']='SELECT 1e999';s.execute()
        self.assertEqual(s.state['error'],'non-finite SQL result');self.assertIsNone(s.state['output'])

    def test_consistency_seeds_cost_disagreement_and_bounded_budget(self):
        task=self.task('net_total');sql=REFERENCE_SQL['net_total']
        def response(query): return {'message':{'content':json.dumps({'sql':query})},'eval_duration':1_000_000_000}
        s=Session(task,consistency_verification=True,fresh_escalation=True)
        with patch('study.runtime.api',side_effect=[response(sql),response(sql),response(sql),response('SELECT 1')]) as provider:
            s.generate(SMALL);s.apply('verify_consistency')
        bodies=[c.args[1] for c in provider.call_args_list]
        self.assertEqual([b['options']['seed'] for b in bodies[1:]],[2,3,4])
        self.assertEqual({b['options']['temperature'] for b in bodies[1:]},{.7})
        self.assertTrue(s.state['checks']['flagged']);self.assertEqual(s.state['checks']['agreement_fraction'],.75)
        self.assertEqual(s.state['checks']['distinct_results'],2);self.assertEqual(s.result()['calls'],4)
        self.assertGreaterEqual(s.effort(),4);self.assertNotIn('verify_consistency',s.eligible())
        with patch('study.runtime.api',return_value=response(sql)) as provider: fixed_continuation(s)
        self.assertEqual(provider.call_args.args[1]['model'],LARGE)
        self.assertEqual(provider.call_args.args[1]['messages'][1]['content'],json.dumps(s.public)+'\n')
        with self.assertRaises(ValueError): s.check_consistency()
        with self.assertRaises(ValueError): Session(task,consistency_k=1)

    def test_consistently_wrong_is_not_correctness_evidence(self):
        s=Session(self.task('net_total'),consistency_verification=True)
        with patch('study.runtime.api',return_value={'message':{'content':'{"sql":"SELECT 1"}'}}):
            s.generate(SMALL);s.check_consistency()
        self.assertFalse(s.state['checks']['flagged']);self.assertFalse(s.result()['success'])
        self.assertEqual(s.state['checks']['agreement_fraction'],1)

    def test_initial_routing_and_feature_vocabularies_keep_legacy_dimensions(self):
        s=Session(self.task('net_total'),initial_routing=True)
        self.assertEqual(s.eligible(),['generate_small','route_large'])
        self.assertEqual(len(features(s)),22);self.assertEqual(len(features(s,feature_names=REVISION_FEATURES)),24)
        with patch.object(s,'generate') as generate: s.apply('route_large')
        self.assertEqual(generate.call_args.args,(LARGE,''))
        s.state['checks']={'agreement_fraction':.75,'distinct_results':2}
        self.assertEqual(features(s,feature_names=REVISION_FEATURES)[-2:],[.75,math.log1p(2)])
        self.assertEqual(features(s,False,REVISION_FEATURES)[-2:],[0,0])
        rows=[dict(template=str(i),split='train',features=[0.0]*24,feature_names=REVISION_FEATURES,
                   action='route_large',success=bool(i%2),remaining_effort_seconds=1.0) for i in range(4)]
        for cls in (Controller,TableController,LogisticV2Controller):
            model=cls.train(rows);self.assertEqual(model.model['feature_names'],REVISION_FEATURES)
            self.assertEqual(model.choose(s)[0],'route_large')
        with self.assertRaises(ValueError): training_feature_names(rows+[dict(rows[0],feature_names=FEATURES)])
        with self.assertRaises(ValueError): features(s,feature_names=['gold'])
