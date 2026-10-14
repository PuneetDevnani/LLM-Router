import json
import ast
from pathlib import Path
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
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


class ControllerSelectionTests(unittest.TestCase):
    def choose(self, predictions, **settings):
        session=Mock()
        session.eligible.return_value=list(predictions)
        controller=Controller({'reference_seconds':10.0})
        with patch('study.controller.features',return_value=[0.0]), \
             patch.object(controller,'predict',side_effect=lambda f,a: predictions[a]):
            return controller.choose(session,**settings)

    def test_zero_defaults_preserve_original_scores_and_choice(self):
        predictions={'accept':(0.012,0),'stop':(0,0),'repair':(0.06,10),
                     'escalate':(0.2,30),'verify':(0.02,2)}
        action,values=self.choose(predictions,penalty=0.3)
        expected={a:p-0.3*c/10 for a,(p,c) in predictions.items()}
        self.assertEqual(action,max(expected,key=expected.get))
        self.assertEqual({v['action']:v['score'] for v in values},expected)
        self.assertTrue(all('gated' not in v for v in values))

    def test_gate_forces_recovery_even_when_stop_scores_higher(self):
        action,values=self.choose({'accept':(0.01,0),'stop':(0,0),
                                  'repair':(0.1,30),'escalate':(0.2,40)},
                                 accept_threshold=0.5)
        self.assertEqual(action,'repair')
        self.assertEqual({v['action'] for v in values if v['gated']},{'accept','stop'})

    def test_threshold_equality_allows_acceptance(self):
        action,values=self.choose({'accept':(0.5,0),'stop':(0,0),'repair':(0.1,20)},
                                 accept_threshold=0.5)
        self.assertEqual(action,'accept')

    def test_exhausted_recovery_can_accept_below_threshold(self):
        action,_=self.choose({'accept':(0.01,0),'stop':(0,0)},accept_threshold=0.5)
        self.assertEqual(action,'accept')
        action,_=self.choose({'accept':(0.01,0),'stop':(0,0)},
                            accept_threshold=0.5,wrong_penalty=1.0)
        self.assertEqual(action,'stop')

    def test_terminal_losses_allow_abstention_or_change_its_cost(self):
        predictions={'accept':(0.4,0),'stop':(0,0)}
        action,values=self.choose(predictions,wrong_penalty=1.0)
        self.assertEqual(action,'stop')
        self.assertAlmostEqual(values[0]['score'],-0.2)
        action,values=self.choose(predictions,wrong_penalty=1.0,abstain_cost=0.3)
        self.assertEqual(action,'accept')
        self.assertAlmostEqual(values[1]['score'],-0.3)

    def test_model_only_removes_verification_before_gating(self):
        action,values=self.choose({'accept':(0.01,0),'stop':(0,0),'verify':(0.9,1)},
                                 mode='model_only',accept_threshold=0.5)
        self.assertEqual(action,'accept')
        self.assertNotIn('verify',[v['action'] for v in values])

    def test_no_effort_keeps_terminal_loss_and_retry_eligible(self):
        action,_=self.choose({'accept':(0.01,0),'stop':(0,0),'retry':(0.9,100)},
                            mode='no_effort',accept_threshold=0.5)
        self.assertEqual(action,'retry')
        action,_=self.choose({'accept':(0.4,0),'stop':(0,0)},
                            mode='no_effort',wrong_penalty=1.0)
        self.assertEqual(action,'stop')


# Include the phase-two checks in the required study test command.
from study.test_revision import PhaseTwoTests, RevisionTests
from study.test_runtime_revision import RecoveryRevisionTests
from study.test_offline_eval import OfflineEvalTests
from study.test_mutation_audit import MutationAuditTests
from study.test_guard_revision import GuardRevisionTests
from study.test_data_v2 import CorpusV2Tests
from study.test_headroom import HeadroomTests
from study.test_task_routing import TaskRoutingTests
from study.test_completion import CompletionTests
from study.test_workflow import WorkflowTests
from study.test_amendments import AmendmentTests
from study.test_economics import EconomicsTests

if __name__=='__main__': unittest.main()
