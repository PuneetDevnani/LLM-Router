"""Routing fixtures are synthetic; no model provider is used."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study.task_routing import task_type, TaskRouter, main
from study.experiment import run_policy
from study.runtime import SMALL,LARGE
from study.data_v2 import corpus,REFERENCE_SQL


class TaskRoutingTests(unittest.TestCase):
    def rows(self):
        tasks=[t for t in corpus() if t['template'] in ('net_total','average_net','completion_rate') and t['seed']<6 and t['id'].endswith('w0')]
        return [dict(task_id=t['id'],template=t['template'],split='train',trial=0,fault='clean',
                     model=m,public={'contract':t['contract']},success=m==LARGE or t['template']!='average_net',
                     effort_seconds=1 if m==SMALL else 5) for t in tasks for m in (SMALL,LARGE)]

    def test_public_category_ignores_private_fields(self):
        row=self.rows()[0];changed=deepcopy(row['public'])
        changed.update(template='GOLD',seed=99,split='test',expected={'GOLD':42},id='GOLD')
        self.assertEqual(task_type(row['public']),task_type(changed))
        self.assertNotIn('GOLD',task_type(changed))

    def test_training_only_paired_unique_population_and_oof_folds(self):
        rows=self.rows();router=TaskRouter.train(rows,[SMALL,LARGE])
        for fold in router.model['calibration_folds']:
            self.assertNotIn(fold['held_out_template'],fold['training_templates'])
        self.assertEqual(router.model['training_observations'],36)
        for bad in (rows+[rows[0]],rows[:-1],[dict(rows[0],split='validation')]+rows[1:],
                    [dict(rows[0],fault='transient')]+rows[1:]):
            with self.assertRaises(ValueError): TaskRouter.train(bad,[SMALL,LARGE])
        with self.assertRaises(ValueError): TaskRouter.train(rows[:24],[SMALL,LARGE])
        with self.assertRaises(ValueError): TaskRouter.train(rows,[SMALL,SMALL])
        with self.assertRaises(ValueError): TaskRouter.fit_base(rows,[SMALL,LARGE],prior=0)

    def test_known_and_unknown_category_backoff_and_cost_choice(self):
        rows=self.rows();router=TaskRouter.fit_base(rows,[SMALL,LARGE])
        model,values=router.choose(rows[0]['public']);self.assertEqual(model,SMALL)
        self.assertEqual(len(values),2)
        unknown=deepcopy(rows[0]['public']);unknown['contract']['aggregation']='unknown'
        p,c=router.predict(unknown,SMALL);self.assertAlmostEqual(p,2/3);self.assertEqual(c,1)
        self.assertEqual(router.choose(unknown)[0],LARGE)

    def test_non_positive_calibration_slope_backs_off_without_reversing_rates(self):
        rows=self.rows()
        with patch('study.task_routing.fit_platt',return_value={'kind':'platt','slope':-2,'intercept':0,'samples':18}):
            router=TaskRouter.train(rows,[SMALL,LARGE])
        self.assertEqual(router.model['unconstrained_calibration'][SMALL]['slope'],-2)
        self.assertEqual(router.model['calibration'][SMALL]['kind'],'constant')
        public=rows[0]['public'];unknown=deepcopy(public);unknown['contract']['aggregation']='unknown'
        self.assertEqual(router.predict(public,SMALL)[0],router.predict(unknown,SMALL)[0])
        with patch('study.task_routing.fit_platt',return_value={'kind':'platt','slope':2,'intercept':0,'samples':18}):
            monotone=TaskRouter.train(rows,[SMALL,LARGE])
        self.assertEqual(monotone.model['calibration'][SMALL]['slope'],2)

    def test_save_load_and_cli_excludes_validation_and_preserves_model(self):
        rows=self.rows()
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp);path=root/'router.json'
            (root/'profile_summary.json').write_text(json.dumps({'complete':True,'suitable_pair':True,
                'dataset_sha256':'dataset','models':{SMALL:{},LARGE:{}}}))
            (root/'profile.jsonl').write_text('\n'.join(json.dumps(r) for r in rows+[dict(rows[0],split='validation')])+'\n')
            args=['routing','--profile-directory',tmp,'--output',str(path)]
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()): main()
            fitted=TaskRouter.load(path);self.assertEqual(fitted.model['training_observations'],36)
            self.assertEqual(fitted.model['dataset_sha256'],'dataset')
            raw=path.read_bytes()
            with self.assertRaises(ValueError): fitted.save(path)
            with patch('sys.argv',args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main()
            self.assertEqual(path.read_bytes(),raw)

    def test_policy_routes_before_generation_and_retries_same_model_after_error(self):
        task=corpus()[0];router=TaskRouter.fit_base(self.rows(),[SMALL,LARGE])
        good={'message':{'content':json.dumps({'sql':REFERENCE_SQL[task['template']]})}}
        with patch('study.runtime.api',side_effect=[{'message':{'content':'{"sql":"SELECT missing FROM orders"}'}},good]) as provider:
            result=run_policy(task,'task_routing',router=router)
        self.assertTrue(result['success'])
        self.assertEqual([c.args[1]['model'] for c in provider.call_args_list],[SMALL,SMALL])
        self.assertEqual(result['trace'][0]['kind'],'routing')
        self.assertNotIn(task['id'],json.dumps(result['trace'][0]))
        with patch.object(router,'choose',return_value=(LARGE,[])),patch('study.runtime.api',return_value=good) as provider:
            result=run_policy(task,'task_routing',router=router,fault='transient')
        self.assertTrue(result['success']);self.assertEqual(result['calls'],1)
        self.assertEqual(result['operations'],2);self.assertEqual(provider.call_args.args[1]['model'],LARGE)
        with patch.object(router,'choose',return_value=('foreign-model',[])),patch('study.runtime.api') as provider:
            with self.assertRaises(ValueError): run_policy(task,'task_routing',router=router)
        provider.assert_not_called()
        with self.assertRaises(ValueError): run_policy(task,'task_routing')
