"""No-provider regression tests for exploratory routing changes."""
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
from unittest.mock import patch
from study.controller import TableController, features
from study.data import corpus
from study.runtime import Session, SMALL, LARGE
from study.experiment import evaluation_jobs, job_key, run_policy, main
from study.analyse import select_penalty, summarise, paired


class RevisionTests(unittest.TestCase):
    def row(self,policy='learned',penalty=0.0,threshold=0.0,success=True,effort=1.0):
        return dict(task_id='t',template='family',split='validation',trial=0,fault='clean',
                    policy=policy,penalty=penalty,threshold=threshold,success=success,
                    effort_seconds=effort,wall_seconds=effort,calls=1,trace=[])

    def test_legacy_resume_key_is_zero_threshold_only(self):
        old=self.row(); del old['threshold']
        self.assertEqual(job_key(old),job_key(self.row()))
        self.assertNotEqual(job_key(old),job_key(self.row(threshold=0.1)))

    def test_cli_resumes_legacy_rows_and_only_runs_new_threshold(self):
        tasks=[t for t in corpus() if t['split']=='validation' and t['seed']==0 and t['id'].endswith('w0')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata={'source_hashes':{'mock':'v1'},'verifier':'original'}
            (root/'provenance.json').write_text(json.dumps(metadata))
            old=[dict(self.row(),task_id=t['id'],template=t['template']) for t in tasks]
            for r in old: del r['threshold']
            path=root/'validation.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in old))
            def result(task,policy,trial,fault,controller,penalty,threshold,verifier,controller_type):
                return dict(self.row(policy,penalty,threshold),task_id=task['id'],template=task['template'])
            argv=['experiment','evaluate','--output',tmp,'--split','validation','--seeds','1',
                  '--policies','learned','--penalties','0','--thresholds','0,0.1']
            with patch('study.experiment.V2_ROOT',root),patch('sys.argv',argv),patch('study.experiment.provenance',return_value=metadata), \
                 patch('study.experiment.Controller.load'),patch('study.experiment.run_policy',side_effect=result) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                main();self.assertEqual(run.call_count,3)
                main();self.assertEqual(run.call_count,3)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertEqual(sum(r.get('threshold',0)==.1 for r in rows),3)
            self.assertEqual(json.loads((root/'provenance.json').read_text()),metadata)

    def test_grid_has_twenty_points_and_one_copy_of_each_baseline(self):
        jobs=evaluation_jobs([{'id':'t'}],['clean'],1,['learned','table','large','escalate_default'],
                             [0,.05,.15,.3],[0,.05,.1,.2,.3])
        self.assertEqual(len(jobs),42)
        self.assertEqual(len({(t['id'],*rest) for t,*rest in jobs}),42)

    def test_selection_admissibility_and_threshold_tie_break(self):
        rows=[self.row('large'),self.row(threshold=0.0,effort=2),
              self.row(threshold=.2,effort=2),self.row(threshold=.3,success=False,effort=.1)]
        selection=select_penalty(rows)
        self.assertEqual(selection['selected_threshold'],.2)

    def test_fallback_preserves_success_then_effort_then_threshold(self):
        rows=[self.row('large'),self.row(threshold=0,success=False,effort=2),
              self.row(threshold=.2,success=False,effort=2),self.row(threshold=.3,success=False,effort=3)]
        self.assertEqual(select_penalty(rows)['selected_threshold'],.2)

    def test_old_validation_selects_the_same_penalty(self):
        rows=[self.row('large'),self.row(penalty=.3),self.row(penalty=0,success=False)]
        for r in rows: del r['threshold']
        result=select_penalty(rows)
        self.assertEqual((result['selected_penalty'],result['selected_threshold']),(.3,0.0))

    def test_incomplete_and_test_selection_rejected(self):
        rows=[self.row('large'),dict(self.row(),task_id='other')]
        with self.assertRaises(ValueError): select_penalty(rows)
        rows=[self.row('large'),dict(self.row(),split='test')]
        with self.assertRaises(ValueError): select_penalty(rows)

    def test_summaries_separate_grid_and_pairs_reject_duplicate_settings(self):
        rows=[self.row(),self.row(threshold=.2),self.row('large')]
        self.assertEqual(len(summarise(rows)),3)
        with self.assertRaises(ValueError): paired(rows)

    def test_run_policy_passes_and_records_threshold(self):
        class Stub:
            def choose(self,session,penalty,mode,accept_threshold):
                self.received=(penalty,mode,accept_threshold)
                return 'accept',[]
        c=Stub()
        with patch('study.runtime.api',return_value={'message':{'content':'{"sql":"SELECT SUM(amount) FROM orders"}'}}):
            row=run_policy(corpus()[0],'learned',controller=c,penalty=.15,threshold=.2)
        self.assertEqual(c.received,(.15,'joint',.2))
        self.assertEqual(row['threshold'],.2)

    def test_escalate_default_uses_large_even_after_executable_small_output(self):
        responses=[{'message':{'content':'{"sql":"SELECT SUM(amount) FROM orders"}'}},
                   {'message':{'content':'{"sql":"SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status=\'completed\'"}'}}]
        with patch('study.runtime.api',side_effect=responses):
            row=run_policy(corpus()[0],'escalate_default')
        self.assertEqual([e['model'] for e in row['trace'] if e['kind']=='generation'],[SMALL,LARGE])
        self.assertTrue(row['success'])

    def test_escalation_baseline_retries_timeout_without_extra_generation(self):
        responses=[{'message':{'content':'{"sql":"SELECT SUM(amount) FROM orders"}'}},
                   {'message':{'content':'{"sql":"SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status=\'completed\'"}'}}]
        with patch('study.runtime.api',side_effect=responses):
            row=run_policy(corpus()[0],'escalate_default',fault='transient')
        actions=[e['action'] for e in row['trace'] if e['kind']=='decision']
        self.assertEqual(actions,['retry','escalate','accept'])
        self.assertEqual(row['calls'],2)

    def test_empirical_cells_and_unseen_cell_fallback(self):
        f=features(Session(corpus()[0])); f[11]=0.0
        rows=[dict(features=f,action='repair',success=s,remaining_effort_seconds=c)
              for s,c in [(True,2),(False,4)]]
        table=TableController.train(rows)
        self.assertEqual(table.predict(f,'repair'),(.5,3.0))
        unseen=list(f);unseen[11]=1.0
        self.assertEqual(table.predict(unseen,'repair'),(.5,3.0))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'table.json';table.save(path)
            self.assertEqual(TableController.load(path).predict(f,'repair'),(.5,3.0))

    def session(self,sql,template='net_total'):
        s=Session(next(t for t in corpus() if t['template']==template),verifier='contract_v2')
        # These legacy fixtures predate the explicit public net-value field.
        s.public['contract']['net_value']=True
        s.state['sql']=sql;s.execute();s.check();return s

    def test_net_probe_detects_omitted_discount_without_gold(self):
        s=self.session("SELECT SUM(amount) FROM orders WHERE status='completed'")
        self.assertTrue(s.state['checks']['flagged'])
        self.assertTrue(any(e['check']=='preserve_net_value' and e['status']=='violation'
                            for e in s.state['checks']['evidence']))
        self.assertNotIn('expected',s.state['checks'])

    def test_correct_net_queries_preserve_probe(self):
        for sql,template in [("SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'",'net_total'),
                             ("SELECT AVG(amount-IFNULL(discount,0)) FROM orders WHERE status='completed'",'average_net')]:
            with self.subTest(template=template):
                s=self.session(sql,template)
                self.assertFalse(s.state['checks']['flagged'])
                self.assertTrue(s.result()['success'])

    def test_frozen_main_output_rejected_before_provider_calls(self):
        with patch('sys.argv',['experiment','evaluate','--output','results/main']), \
             patch('study.experiment.provenance') as provider,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): main()
        provider.assert_not_called()




class PhaseTwoTests(unittest.TestCase):
    @staticmethod
    def training_rows():
        from study.controller import FEATURES
        rows=[]
        for t in range(3):
            for i,action in enumerate(['accept','repair','escalate','verify']):
                f=[0.0]*len(FEATURES);f[0]=t+i/4;f[11]=float(i==1);f[16]=float(i==2)
                rows.append(dict(template=f't{t}',split='train',features=f,action=action,
                                 success=bool(i==2 or (t==1 and i==0)),remaining_effort_seconds=0.0 if action=='accept' else float(i+1)))
        return rows

    def test_saved_branch_default_choices_exactly_match_v1(self):
        from study.controller import Controller, ACTIONS
        from unittest.mock import Mock
        frozen=Path('results/main/frozen_source/controller.py')
        namespace={'__name__':'frozen_controller'}
        exec(compile(frozen.read_text(),str(frozen),'exec'),namespace)
        model=json.loads(Path('results/main/controller.json').read_text())
        old=namespace['Controller'](model);current=Controller(model)
        rows=[json.loads(line) for line in Path('results/main/branches.jsonl').read_text().splitlines()]
        session=Mock();session.eligible.return_value=list(ACTIONS)
        for row in rows:
            f=row['features']
            with patch.dict(namespace,{'features':lambda s,history=True:f}),patch('study.controller.features',return_value=f):
                self.assertEqual(current.choose(session,penalty=.3),old.choose(session,penalty=.3))

    def test_base_scaling_keeps_other_interaction_blocks_zero(self):
        import numpy as np
        from study.controller import encode_v2, FEATURES, ACTIONS, LogisticV2Controller
        f=np.arange(len(FEATURES),dtype=float)
        encoded=encode_v2(f,'repair',np.ones(len(f)),np.ones(len(f))*2)
        self.assertTrue(np.allclose(encoded[:len(f)],(f-1)/2))
        blocks=encoded[len(f)+len(ACTIONS):].reshape(len(ACTIONS),len(f))
        for i,action in enumerate(ACTIONS):
            self.assertTrue(np.allclose(blocks[i],(f-1)/2 if action=='repair' else 0))
        with self.assertRaises(ValueError): encode_v2(f,'unknown',np.zeros(len(f)),np.ones(len(f)))
        fit=LogisticV2Controller.fit_base(self.training_rows()).model
        self.assertGreater(fit['regularization'][-1],fit['regularization'][0])
        self.assertGreater(fit['ridge_regularization'][-1],fit['ridge_regularization'][1])

    def test_smoothed_table_shrinks_sparse_cell_to_action_mean(self):
        from study.controller import TableController, FEATURES
        f=[0.0]*len(FEATURES);other=list(f);other[16]=1
        rows=[dict(features=features,action='repair',success=success,remaining_effort_seconds=effort)
              for features,success,effort in [(f,True,2),(other,False,4),(other,False,6)]]
        table=TableController.train(rows)
        p,c=table.predict(f,'repair')
        self.assertAlmostEqual(p,5/9);self.assertAlmostEqual(c,10/3)
        unseen=list(f);unseen[19]=1
        self.assertEqual(table.predict(unseen,'repair'),(1/3,4))
        self.assertEqual(table.predict(f,'stop'),(0,0))
        self.assertEqual(table.predict(f,'retry'),(0,0))
        with self.assertRaises(ValueError): TableController.train(rows,prior=0)
        with self.assertRaises(ValueError): TableController.train([])
        with self.assertRaises(ValueError): TableController.train([dict(rows[0],split='test')])

    def test_calibration_uses_disjoint_template_folds_and_round_trips(self):
        import numpy as np
        from study.controller import LogisticV2Controller, FEATURES, fit_platt, apply_platt
        rows=self.training_rows()
        with patch.object(LogisticV2Controller,'fit_base',wraps=LogisticV2Controller.fit_base) as fit:
            model=LogisticV2Controller.train(rows)
        for call,held_out in zip(fit.call_args_list[:-1],['t0','t1','t2']):
            self.assertNotIn(held_out,{r['template'] for r in call.args[0]})
        for fold in model.model['calibration_folds']:
            self.assertNotIn(fold['held_out_template'],fold['training_templates'])
        self.assertEqual(model.model['calibration']['samples'],len(rows))
        for r in rows:
            p,c=model.predict(r['features'],r['action'])
            self.assertTrue(0<=p<=1 and np.isfinite(p) and c>=0 and np.isfinite(c))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';model.save(path)
            loaded=LogisticV2Controller.load(path)
            self.assertEqual(loaded.predict(rows[0]['features'],'accept'),model.predict(rows[0]['features'],'accept'))
        constant=fit_platt([0,1],[0,0])
        self.assertEqual(constant['kind'],'constant')
        self.assertTrue(0<apply_platt(5,constant)<1)
        identity={'kind':'identity'};self.assertAlmostEqual(apply_platt(0,identity),.5)
        with self.assertRaises(ValueError): fit_platt([],[])
        with self.assertRaises(ValueError): LogisticV2Controller.train([rows[0]])
        for key,value in [('split','test'),('features',[0]),('remaining_effort_seconds',-1),('action','bad')]:
            bad=[dict(rows[0],**{key:value})]
            with self.assertRaises(ValueError): LogisticV2Controller.fit_base(bad)

    def test_nested_training_cv_has_all_three_comparators(self):
        from study.controller import leave_one_template_out_cv
        result=leave_one_template_out_cv(self.training_rows())
        self.assertEqual(set(result),{'logistic_v1','logistic_v2','table'})
        for comparison in result.values():
            self.assertEqual(len(comparison['folds']),3)
            for fold in comparison['folds']:
                self.assertNotIn(fold['held_out_template'],fold['training_templates'])
        with self.assertRaises(ValueError): leave_one_template_out_cv(self.training_rows()[:4])

    def test_grid_modes_and_no_reward_for_degenerate_inaction(self):
        from study.analyse import select_operating_point
        row=RevisionTests().row
        result=select_operating_point([row('large'),row(penalty=0,success=False,effort=5),
                                       row(penalty=.3,success=False,effort=1)])
        self.assertEqual(result['selection_mode'],'degenerate_tie')
        self.assertEqual(result['selected_penalty'],0)
        result=select_operating_point([row('large'),row(penalty=0,success=False),
                                       row(penalty=.3,success=False,effort=2),
                                       dict(row(penalty=.3,success=True),trial=1),
                                       dict(row('large'),trial=1),dict(row(penalty=0,success=False),trial=1)])
        self.assertEqual(result['selection_mode'],'best_success_fallback')
        result=select_operating_point([row('large'),row(threshold=0,effort=2),row(threshold=.3,effort=2)])
        self.assertEqual(result['selection_mode'],'admissible')
        self.assertEqual(result['selected_threshold'],.3)
        with self.assertRaises(ValueError): select_operating_point([],tolerance=-1)
        with self.assertRaises(ValueError): select_operating_point([row('large'),dict(row(),split='test')])

    def test_training_cli_selects_v2_and_writes_training_cv(self):
        from study.experiment import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=self.training_rows()
            (root/'branches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            argv=['experiment','train','--output',tmp,'--controller','logistic_v2']
            with patch('study.experiment.V2_ROOT',root),patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                main()
            model=json.loads((root/'controller_logistic_v2.json').read_text())
            cv=json.loads((root/'training_cv_logistic_v2.json').read_text())
            self.assertEqual(model['controller_type'],'logistic_v2')
            self.assertEqual(cv['training_branches_sha256'],model['training_branches_sha256'])
            self.assertEqual(set(cv['comparisons']),{'table','logistic_v1','logistic_v2'})
            self.assertFalse((root/'controller.json').exists())
            self.assertFalse((root/'test.jsonl').exists())

    def test_seen_test_split_is_blocked_before_provider_calls(self):
        with patch('sys.argv',['experiment','evaluate','--split','test']), \
             patch('study.experiment.provenance') as provider,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): main()
        provider.assert_not_called()

    def test_controller_type_is_part_of_resume_identity(self):
        row=RevisionTests().row
        self.assertNotEqual(job_key(row()),job_key(dict(row(),controller_type='logistic_v2')))
        jobs=evaluation_jobs([{'id':'t'}],['clean'],1,['learned','large'],[0],[0],controller_type='table')
        self.assertEqual(jobs[0][-1],'table');self.assertEqual(jobs[1][-1],'none')


if __name__=='__main__': unittest.main()
