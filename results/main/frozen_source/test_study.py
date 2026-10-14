import json
import ast
from pathlib import Path
import unittest
from copy import deepcopy
from unittest.mock import patch
from study.data import corpus, public, report_from_result, GROUPS
from study.runtime import Session, lexical_checks
from study.controller import Controller, features
from pilot.environment import query, correct
from study.data import database


class StudyTests(unittest.TestCase):
    def test_main_corpus_reference_sql_against_record_oracles(self):
        source=ast.parse(Path('pilot/test_richer.py').read_text())
        references=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='SQL' for t in n.targets))
        for task in corpus():
            con,_=database(task['seed'])
            report=report_from_result(query(con,references[task['template']]),task['contract'])
            self.assertTrue(correct(report,task['expected']),task['id'])
            con.close()
    def test_grouped_splits_and_public_boundary(self):
        tasks=corpus()
        self.assertEqual(len(tasks),240)
        for template in set(t['template'] for t in tasks):
            self.assertEqual(len({t['split'] for t in tasks if t['template']==template}),1)
        self.assertEqual(set(GROUPS['train']) & set(GROUPS['test']),set())
        for task in tasks:
            self.assertEqual(set(public(task)),{'prompt','schema','contract'})

    def test_gold_and_metadata_do_not_change_features_or_request(self):
        task=corpus()[0]; modified=deepcopy(task)
        modified.update(expected={'SECRET':'GOLD_CANARY'},id='SECRET',template='SECRET',split='SECRET')
        self.assertEqual(features(Session(task)),features(Session(modified)))
        with patch('study.runtime.api',return_value={'message':{'content':'{"sql":"SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status=\'completed\'"}'}}) as api:
            Session(modified).generate('small')
        self.assertNotIn('GOLD_CANARY',json.dumps(api.call_args.args[1]))
        self.assertNotIn('SECRET',json.dumps(api.call_args.args[1]))

    def test_formatter_equivalence_and_duplicate_rejection(self):
        self.assertEqual(report_from_result({'rows':[['north',2],['south',1]]},{'report_keys':['regions'],'grouped':True}),{'regions':{'north':2,'south':1}})
        with self.assertRaises(ValueError):
            report_from_result({'rows':[['north',2],['north',1]]},{'report_keys':['regions'],'grouped':True})
        with self.assertRaises(ValueError):
            report_from_result({'rows':[[1],[2]]},{'report_keys':['count'],'grouped':False})

    def test_retry_without_generation_and_budget_enforcement(self):
        s=Session(corpus()[0],fault='transient',max_calls=1)
        s.state['sql']="SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'"
        s.execute(); self.assertEqual(s.state['error'],'tool_timeout')
        s.apply('retry'); self.assertIsNone(s.state['error']); self.assertEqual(s.calls,0)
        s.calls=1
        self.assertNotIn('repair',s.eligible())
        with self.assertRaises(ValueError): s.apply('repair')

    def test_imperfect_checks_not_gold_solver(self):
        s=Session(corpus()[0])
        s.state['sql']="SELECT SUM(amount) FROM orders WHERE status='completed'"
        s.execute(); s.check()
        # Wrong net semantics can pass invariance when the SQL never refers to discount.
        self.assertFalse(s.state['checks']['flagged'])
        self.assertFalse(s.result()['success'])

    def test_controller_finite_predictions_and_eligible_actions(self):
        s=Session(corpus()[0]); f=features(s)
        rows=[{'features':f,'action':a,'success':a=='escalate','remaining_effort_seconds':0 if a in ('accept','stop') else 2}
              for _ in range(4) for a in ['accept','stop','escalate','repair','verify']]
        c=Controller.train(rows)
        action,values=c.choose(s)
        self.assertIn(action,s.eligible())
        self.assertTrue(all(0<=v['predicted_success']<=1 and v['predicted_seconds']>=0 for v in values))
        self.assertEqual(c.predict(f,'stop'),(0.0,0.0))


if __name__=='__main__': unittest.main()
