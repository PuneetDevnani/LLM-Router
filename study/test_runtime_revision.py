"""Offline regressions for opt-in recovery and evidence plumbing."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from study.controller import ACTIONS, Controller, TableController, LogisticV2Controller, action_list, features, encode, leave_one_template_out_cv
from study.data import corpus
from study.diagnostics import reference_queries, detector_audit, small_by_template
from study.experiment import main, provenance, run_policy
from study.runtime import Session, SMALL, LARGE, fixed_continuation
from study.analyse import paired
from study.test_revision import PhaseTwoTests, RevisionTests


class RecoveryRevisionTests(unittest.TestCase):
    def test_fresh_action_is_opt_in_bounded_and_unanchored(self):
        task=corpus()[0]; s=Session(task)
        self.assertNotIn('escalate_fresh',s.eligible())
        with self.assertRaises(ValueError): s.apply('escalate_fresh')
        s=Session(task,fresh_escalation=True);s.state['sql']='PRIVATE_PREVIOUS_SQL'
        with patch.object(s,'generate') as generate:
            s.apply('escalate_fresh')
        generate.assert_called_once_with(LARGE,'')
        s.calls=s.max_calls;self.assertNotIn('escalate_fresh',s.eligible())
        self.assertEqual(ACTIONS,['accept','repair','escalate','verify','stop','retry'])

    def test_neutral_feedback_is_opt_in_and_respects_errors_and_flags(self):
        s=Session(corpus()[0]);s.state['sql']=reference_queries()['net_total']
        legacy=s.feedback();self.assertIn('Observed error: None',legacy)
        s.neutral_feedback=True
        self.assertEqual(s.feedback(),'Solve the task again from the request and contract.')
        for change in ({'error':'syntax error'},{'checks':{'flagged':True}}, {'sql':'SELECT amount FROM orders'}):
            old=dict(s.state);s.state.update(change)
            self.assertIn('Generate a corrected SQLite query.',s.feedback());s.state=old
        s.neutral_feedback=False;self.assertEqual(s.feedback(),legacy)

    def test_fresh_policy_matches_large_prompt_and_default_stays_anchored(self):
        response={'message':{'content':json.dumps({'sql':reference_queries()['net_total']})}}
        with patch('study.runtime.api',return_value=response) as api:
            run_policy(corpus()[0],'large');large_request=api.call_args.args[1]
        with patch('study.runtime.api',return_value=response) as api:
            fresh=run_policy(corpus()[0],'escalate_fresh');fresh_request=api.call_args.args[1]
        self.assertEqual(fresh_request,large_request)
        self.assertEqual([e['action'] for e in fresh['trace'] if e['kind']=='decision'],['escalate_fresh','accept'])
        with patch('study.runtime.api',return_value=response) as api:
            run_policy(corpus()[0],'escalate_default',neutral_feedback=True)
        self.assertIn('Previous SQL:',api.call_args.args[1]['messages'][1]['content'])
        with patch('study.runtime.api',return_value=response):
            routed=run_policy(corpus()[0],'large',fault='transient')
        self.assertEqual([e['model'] for e in routed['trace'] if e['kind']=='generation'],[LARGE])
        self.assertEqual(routed['calls'],1)
        self.assertTrue(routed['success'])

    def test_fixed_continuation_uses_fresh_only_when_enabled(self):
        for fresh in (False,True):
            s=Session(corpus()[0],fresh_escalation=fresh);s.state['checks']={'flagged':True}
            with patch.object(s,'generate') as generate: fixed_continuation(s)
            self.assertEqual(generate.call_args.args[0],LARGE)
            self.assertEqual(generate.call_args.args[1],'' if fresh else s.feedback())

    def test_v1_file_loads_and_filters_new_session_action(self):
        controller=Controller.load('results/main/controller.json')
        s=Session(corpus()[0],fresh_escalation=True)
        action,values=controller.choose(s)
        self.assertNotEqual(action,'escalate_fresh')
        self.assertNotIn('escalate_fresh',[v['action'] for v in values])
        self.assertEqual(len(encode(features(s),'accept',controller.actions)),len(controller.model['mean']))
        with self.assertRaises(ValueError): controller.predict(features(s),'escalate_fresh')

    def test_controllers_save_own_vocab_and_cv_retains_rare_actions(self):
        rows=PhaseTwoTests.training_rows()
        rows[0]=dict(rows[0],action='escalate_fresh',remaining_effort_seconds=2)
        for cls in (Controller,TableController,LogisticV2Controller):
            model=cls.train(rows);self.assertIn('escalate_fresh',model.actions)
            p,c=model.predict(rows[0]['features'],'escalate_fresh');self.assertTrue(0<=p<=1 and c>=0)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'controller.json';model.save(path)
                self.assertEqual(cls.load(path).actions,model.actions)
        cv=leave_one_template_out_cv(rows)
        self.assertEqual(set(cv),{'logistic_v1','table','logistic_v2'})
        for actions in (['accept','accept'],['unknown'],['accept']):
            with self.assertRaises(ValueError): action_list(rows,actions)
        controller=Controller({'reference_seconds':1,'actions':['accept','stop','escalate_fresh']})
        session=Session(corpus()[0],fresh_escalation=True)
        with patch.object(controller,'predict',return_value=(.01,0)):
            action,values=controller.choose(session,accept_threshold=.2)
        self.assertEqual(action,'escalate_fresh')

    def test_provenance_hashes_execution_sources_including_pilot(self):
        with patch('study.experiment.api',return_value={}): metadata=provenance()
        hashes=metadata['source_hashes']
        for path in ('pilot/environment.py','pilot/checks.py','pilot/richer.py','study/runtime.py','study/controller.py',
                     'study/data.py','study/verifier.py','study/experiment.py'):
            self.assertEqual(hashes[path],hashlib.sha256(Path(path).read_bytes()).hexdigest())
        self.assertFalse(any('test' in p or p.endswith(('dashboard.py','figures.py','analyse.py')) for p in hashes))

    def test_paired_filters_controller_type_without_hiding_grid_duplicates(self):
        row=RevisionTests().row
        rows=[row('large'),dict(row(success=False),controller_type='logistic_v1'),dict(row(),controller_type='logistic_v2')]
        self.assertEqual(paired(rows,controller_type='logistic_v1')[0]['success_difference'],-1)
        self.assertEqual(paired(rows,controller_type='logistic_v2')[0]['success_difference'],0)
        with self.assertRaises(ValueError): paired(rows)
        rows.append(dict(rows[-1],threshold=.2))
        with self.assertRaises(ValueError): paired(rows,controller_type='logistic_v2')

    def test_ineligible_collection_stages_do_not_save_copies(self):
        task=corpus()[0]
        for stage in ('checked','repaired'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);s=Session(task);s.state['error']='model_call_budget';s.calls=s.max_calls
                saved=dict(task_id=task['id'],fault='clean',trial=0,stage='initial',state=s.state,calls=s.calls,operations=0,trace=[])
                (root/'training_states.jsonl').write_text(json.dumps(saved)+'\n')
                (root/'provenance.json').write_text(json.dumps({'source_hashes':{}}))
                before=(root/'training_states.jsonl').read_bytes()
                argv=['experiment','collect','--output',tmp,'--stage',stage]
                with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[task]), \
                     patch('study.experiment.provenance',return_value={'source_hashes':{}}),patch('study.runtime.api') as api, \
                     contextlib.redirect_stdout(io.StringIO()): main()
                api.assert_not_called()
                self.assertEqual((root/'training_states.jsonl').read_bytes(),before)
                self.assertEqual((root/'branches.jsonl').read_text(),'')

    def test_reference_sql_has_no_false_alarms_in_train_validation(self):
        tasks=[t for t in corpus() if t['split'] in ('train','validation')]
        with patch('study.runtime.api') as api:
            audit=detector_audit(tasks,[],[],verifier='contract_v2')
        api.assert_not_called()
        ref=audit['reference_sql'];self.assertEqual(ref['counts']['true_negative'],180)
        self.assertEqual(ref['counts']['false_positive'],0);self.assertEqual(ref['specificity'],1)
        self.assertEqual(audit['correct_large']['specificity'],None)
        self.assertTrue(all(not r['flagged'] for r in ref['details']))
        self.assertEqual(set(reference_queries()),{t['template'] for t in corpus()})

    def test_detector_audit_uses_correct_large_outputs_and_excludes_seen_tests(self):
        task=corpus()[0];s=Session(task);s.state['sql']=reference_queries()['net_total'];s.execute()
        row=s.result();row['trace'].insert(0,{'kind':'generation','model':LARGE})
        test=next(t for t in corpus() if t['split']=='test')
        with patch('study.runtime.api') as api:
            result=detector_audit([task,test],[],[row,row],verifier='contract_v2')
        api.assert_not_called()
        self.assertEqual(result['correct_large']['counts']['true_negative'],1)
        self.assertEqual(result['correct_large']['specificity'],1)
        self.assertEqual(result['reference_sql']['counts']['true_negative'],1)
        counts=small_by_template([dict(row,policy='small',split='validation')],
                 [dict(row,action='accept',initial_state=dict(s.state,model=SMALL),stage='initial')])
        self.assertEqual(counts['validation_small']['net_total'],{'runs':1,'successes':1})
        self.assertEqual(counts['training_small_initial_accept']['net_total'],{'runs':1,'successes':1})

    def test_completion_trains_comparators_before_any_validation(self):
        from scripts.complete_experiments import main as complete
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp);(root/'branches.jsonl').write_text('{}\n');(root/'selection.json').write_text('{}')
            with patch('sys.argv',['complete','--directory',tmp,'--controller','table']), \
                 patch('scripts.complete_experiments.subprocess.run') as run,contextlib.redirect_stdout(io.StringIO()): complete()
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual([command[3] for command in commands],['train','train','evaluate','select'])
        self.assertTrue(all('--cross-validate' in command for command in commands[:2]))

    def test_large_first_collection_records_large_states_without_inference(self):
        task=corpus()[0]
        def generate(session,model,feedback=''):
            self.assertEqual(model,LARGE)
            session.calls=session.max_calls;session.state['model']=model
            session.state['sql']=reference_queries()['net_total'];session.execute()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sys.argv',['experiment','collect','--output',tmp,'--initial-model','large']), \
                 patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[task]), \
                 patch('study.experiment.provenance',return_value={'source_hashes':{}}),patch.object(Session,'generate',generate), \
                 patch('study.runtime.api') as api,contextlib.redirect_stdout(io.StringIO()): main()
            api.assert_not_called()
            state=json.loads((root/'training_states.jsonl').read_text())
            self.assertEqual(state['state']['model'],LARGE)
            branches=[json.loads(line) for line in (root/'branches.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['features'][16]==1 for r in branches))
            self.assertEqual(json.loads((root/'provenance.json').read_text())['initial_model'],'large')
            before=(root/'branches.jsonl').read_bytes()
            with patch('sys.argv',['experiment','collect','--output',tmp]),patch('study.experiment.V2_ROOT',root), \
                 patch('study.experiment.provenance',return_value={'source_hashes':{}}):
                with self.assertRaisesRegex(ValueError,'Runtime options changed'): main()
            self.assertEqual((root/'branches.jsonl').read_bytes(),before)

    def test_missing_live_controller_and_original_replay_are_explicit(self):
        from study.demo import main as demo
        with patch('sys.argv',['demo','--policy','learned','--directory','results/v2/no_such_controller']), \
             patch('study.demo.run_policy') as run,contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit): demo()
        run.assert_not_called();self.assertIn('Train and select',error.getvalue())
        with patch('sys.argv',['demo','--template','january','--policy','learned','--replay','--directory','results/main']), \
             patch('study.demo.run_policy') as run,contextlib.redirect_stdout(io.StringIO()) as output: demo()
        run.assert_not_called();self.assertIn('SAVED TRACE REPLAY',output.getvalue())


if __name__=='__main__': unittest.main()
