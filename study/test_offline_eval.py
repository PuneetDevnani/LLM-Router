"""Branch-replay regression tests, without inference or primary evidence writes."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from study.controller import Controller, LogisticV2Controller, FEATURES, leave_one_template_out_cv
from study.offline_eval import _group_branches, _summarise_choices, offline_policy_value, evaluate_offline, main


class OfflineEvalTests(unittest.TestCase):
    def setUp(self):
        guard=patch('study.runtime.api',side_effect=AssertionError('Offline evaluation must never call a model'))
        guard.start();self.addCleanup(guard.stop)

    @staticmethod
    def rows():
        rows=[]
        for t in range(3):
            for state in range(2):
                f=[0.0]*len(FEATURES);f[0]=float(t);f[11]=float(state);f[17]=2
                actions=['accept','stop','repair','escalate','verify'] if not state else ['stop','repair','escalate']
                for action in actions:
                    success=(action=='accept' and t==1 or action=='repair' and t==1 or action=='escalate' and t!=1)
                    cost={'accept':0,'stop':0,'repair':3,'escalate':2,'verify':1}[action]
                    rows.append(dict(task_id=f't{t}-s{state}',template=f't{t}',trial=0,fault='clean',split='train',
                                     features=f,action=action,success=bool(success),remaining_effort_seconds=cost))
        return rows

    def test_choose_from_is_same_rule_and_choose_is_thin(self):
        model=Controller({'reference_seconds':10,'actions':['accept','stop','repair']})
        session=Mock();session.eligible.return_value=['accept','stop','repair'];f=[0.0]*len(FEATURES)
        predictions={'accept':(.01,0),'stop':(0,0),'repair':(.2,2)}
        settings=dict(penalty=.15,mode='no_history',accept_threshold=.1,wrong_penalty=.3,abstain_cost=.2)
        with patch('study.controller.features',return_value=f) as extract, \
             patch.object(model,'predict',side_effect=lambda f,a:predictions[a]):
            from_session=model.choose(session,**settings)
            extract.assert_called_once_with(session,history=False)
            self.assertEqual(from_session,model.choose_from(f,session.eligible(),**settings))
        with patch.object(model,'choose_from',return_value=('stop',[])) as choose, \
             patch('study.controller.features',return_value=f):
            self.assertEqual(model.choose(session),('stop',[]))
            choose.assert_called_once_with(f,session.eligible(),.1,'joint',0.0,0.0,0.0)
        with self.assertRaises(ValueError): model.choose_from(f,['unknown'])
        with self.assertRaises(ValueError): Controller({'actions':['verify']}).choose_from(f,['verify'],mode='model_only')

    def test_grouping_preserves_stage_and_rejects_duplicate_or_inconsistent_rows(self):
        rows=self.rows();self.assertEqual(len(_group_branches(rows)),6)
        staged=[dict(r,stage='checked') for r in rows]
        self.assertEqual(len(_group_branches(rows+staged)),12)
        with self.assertRaises(ValueError): _group_branches(rows+[rows[0]])
        bad=deepcopy(rows);bad[1]['features']=list(bad[1]['features']);bad[1]['features'][0]=999
        with self.assertRaises(ValueError): _group_branches(bad)
        bad=deepcopy(rows);bad[1]['template']='different'
        with self.assertRaises(ValueError): _group_branches(bad)
        with self.assertRaises(ValueError): _group_branches([dict(rows[0],split='test')])
        empty=_summarise_choices([]);self.assertEqual(empty['states'],0);self.assertIsNone(empty['success_rate'])

    def test_baselines_value_observed_outcomes_and_label_hindsight(self):
        rows=self.rows();before=deepcopy(rows)
        accept=offline_policy_value(rows,policy='always_accept')
        escalate=offline_policy_value(rows,policy='always_escalate')
        hindsight=offline_policy_value(rows,policy='hindsight_best')
        self.assertEqual(accept['aggregate']['action_counts'],{'accept':3,'stop':3})
        self.assertEqual(accept['aggregate']['successes'],1);self.assertEqual(accept['aggregate']['fallback_states'],3)
        self.assertEqual(escalate['aggregate']['successes'],4)
        self.assertEqual(escalate['aggregate']['mean_remaining_effort_seconds'],2)
        self.assertEqual(hindsight['aggregate']['successes'],6);self.assertTrue(hindsight['upper_bound'])
        self.assertEqual(hindsight['per_template']['t1']['action_counts'],{'accept':1,'repair':1})
        self.assertFalse(escalate['upper_bound']);self.assertEqual(rows,before)
        subset=[r for r in rows if r['task_id']=='t1-s0' and r['action']!='escalate']
        self.assertEqual(offline_policy_value(subset,policy='always_escalate')['aggregate']['action_counts'],{'accept':1})
        subset=[r for r in rows if r['task_id']=='t1-s1' and r['action']=='stop']
        self.assertEqual(offline_policy_value(subset,policy='always_escalate')['aggregate']['action_counts'],{'stop':1})
        with self.assertRaises(ValueError): offline_policy_value([rows[2]],policy='always_accept')
        for settings in ({'policy':'bad'},{'policy':'controller'},{'policy':'always_accept','penalty':-1},
                         {'policy':'always_accept','accept_threshold':2}):
            with self.assertRaises(ValueError): offline_policy_value(rows,**settings)

    def test_controller_decision_uses_only_features_and_logged_options(self):
        model=Mock();model.choose_from.return_value=('escalate',[])
        rows=self.rows();result=offline_policy_value(rows,model,penalty=.3,accept_threshold=.2)
        self.assertEqual(model.choose_from.call_count,6)
        for call in model.choose_from.call_args_list:
            self.assertEqual(len(call.args),2);self.assertEqual(len(call.args[0]),len(FEATURES))
            self.assertEqual(call.kwargs,{'penalty':.3,'accept_threshold':.2})
        self.assertEqual(result['aggregate']['successes'],4)

    def test_stop_does_not_affect_v2_fit_or_oof_calibration(self):
        rows=self.rows();nonstop=[r for r in rows if r['action']!='stop']
        expected=LogisticV2Controller.train(nonstop)
        with patch.object(LogisticV2Controller,'fit_base',wraps=LogisticV2Controller.fit_base) as fit:
            actual=LogisticV2Controller.train(rows)
        self.assertTrue(all(r['action']!='stop' for call in fit.call_args_list for r in call.args[0]))
        self.assertEqual(actual.model['samples'],len(nonstop));self.assertEqual(actual.model['excluded_stop_rows'],6)
        self.assertEqual(actual.model['calibration'],expected.model['calibration'])
        self.assertEqual(actual.model['calibration']['samples'],len(nonstop))
        base=LogisticV2Controller.fit_base(rows)
        self.assertEqual(base.model['samples'],len(nonstop))
        np.testing.assert_allclose(base.model['mean'],LogisticV2Controller.fit_base(nonstop).model['mean'])
        self.assertEqual(Controller.train(rows).model['samples'],len(rows))
        for train in (LogisticV2Controller.fit_base,LogisticV2Controller.train):
            with self.assertRaises(ValueError): train([r for r in rows if r['action']=='stop'])

    def test_cv_metrics_exclude_stop_and_all_folds_exclude_held_template(self):
        rows=self.rows();result=evaluate_offline(rows)
        self.assertEqual(set(result['policies']),{'logistic_v1','table','logistic_v2','always_accept','always_escalate','hindsight_best'})
        for name in ('logistic_v1','table','logistic_v2'):
            policy=result['policies'][name];self.assertEqual(policy['aggregate']['states'],6)
            self.assertEqual(set(policy['per_template']),{'t0','t1','t2'})
            for fold in policy['folds']:
                self.assertNotIn(fold['held_out_template'],fold['training_templates'])
                for inner in fold['calibration_folds']:
                    self.assertNotEqual(inner['held_out_template'],fold['held_out_template'])
                    self.assertNotIn(fold['held_out_template'],inner['training_templates'])
                    self.assertNotIn(inner['held_out_template'],inner['training_templates'])
            cv=result['prediction_cv'][name]
            self.assertEqual(cv['rows'],len(rows)-6);self.assertEqual(cv['excluded_stop_rows'],6)
            self.assertTrue(all(fold['rows']==6 for fold in cv['folds']))
        with self.assertRaises(ValueError): evaluate_offline([r for r in rows if r['template']=='t0'])

    def test_cli_writes_only_v2_and_preserves_input_bytes(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp);source=root/'branches.jsonl';target=root/'offline.json'
            source.write_text(''.join(json.dumps(r)+'\n' for r in self.rows()));before=source.read_bytes()
            with patch('sys.argv',['offline_eval','--branches',str(source),'--output',str(target)]), \
                 contextlib.redirect_stdout(io.StringIO()): main()
            result=json.loads(target.read_text());self.assertEqual(source.read_bytes(),before)
            self.assertIn('sha256',result['source']);self.assertTrue(result['policies']['hindsight_best']['upper_bound'])
            for output in ('results/main/never_write.json',str(source)):
                with patch('sys.argv',['offline_eval','--branches',str(source),'--output',output]), \
                     patch('study.offline_eval.evaluate_offline') as evaluate,contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit): main()
                evaluate.assert_not_called()


if __name__=='__main__': unittest.main()
