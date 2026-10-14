"""Offline tests for recovery, routing, frozen split checks and audit components."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from study.data import corpus
from study.diagnostics import reference_queries
from study.experiment import run_policy, main as experiment, template_list_hash, validate_test_configuration
from study.runtime import Session, SMALL, LARGE, fixed_continuation
from study.analyse import main as analyse
from study.mutation_audit import (KEYWORD_RULES, FAMILY_SPLIT_SHA256, family_partition_spec,
                                 ablated_flag, ablation_effect, component_ablations, mutation_audit)


class GuardRevisionTests(unittest.TestCase):
    def setUp(self):
        guard=patch('study.runtime.api',side_effect=AssertionError('No model calls allowed'))
        guard.start();self.addCleanup(guard.stop)

    @staticmethod
    def frozen_config(templates):
        return {'test_templates':templates,'test_templates_sha256':template_list_hash(templates),
                'groups':{'train':['development'],'validation':[],'test':templates}}

    def test_fresh_eligibility_requires_clean_state_and_diverse_large_regeneration(self):
        for large in (False,True):
            for diverse in (False,True):
                for error in (None,'syntax error'):
                    s=Session(corpus()[0],fresh_escalation=True,diversify_seed=diverse)
                    s.state.update(model=LARGE if large else SMALL,error=error)
                    expected=error is None and (not large or diverse)
                    self.assertEqual('escalate_fresh' in s.eligible(),expected)
                    if not expected:
                        with self.assertRaises(ValueError): s.apply('escalate_fresh')
                    s.calls=s.max_calls;self.assertNotIn('escalate_fresh',s.eligible())

    def test_execution_error_escalation_keeps_feedback_with_fresh_enabled(self):
        response={'message':{'content':json.dumps({'sql':reference_queries()['net_total']})}}
        invalid={'message':{'content':'{"sql":"SELECT nonexistent FROM orders"}'}}
        for policy in ('rules','escalate_fresh','escalate_default'):
            with patch('study.runtime.api',side_effect=[invalid,response]) as api:
                result=run_policy(corpus()[0],policy,fresh_escalation=True,neutral_feedback=True)
            decisions=[e['action'] for e in result['trace'] if e['kind']=='decision']
            self.assertEqual(decisions[0],'escalate');self.assertNotIn('escalate_fresh',decisions)
            self.assertIn('no such column',api.call_args_list[1].args[1]['messages'][1]['content'])
        controller=Mock()
        def choose(session,*args,**kwargs):
            if session.state['error']:
                self.assertNotIn('escalate_fresh',session.eligible())
                return 'escalate',[]
            return 'accept',[]
        controller.choose.side_effect=choose
        with patch('study.runtime.api',side_effect=[invalid,response]) as api:
            run_policy(corpus()[0],'learned',controller=controller,fresh_escalation=True)
        self.assertIn('no such column',api.call_args_list[1].args[1]['messages'][1]['content'])
        s=Session(corpus()[0],fresh_escalation=True);s.state['error']='observed execution error'
        with patch.object(s,'generate') as generate: fixed_continuation(s)
        self.assertEqual(generate.call_args.args,(LARGE,s.feedback()))
        self.assertIn('observed execution error',generate.call_args.args[1])
        for diverse in (False,True):
            s=Session(corpus()[0],fresh_escalation=True,diversify_seed=diverse)
            s.state.update(model=LARGE,checks={'flagged':True})
            with patch.object(s,'generate') as generate: fixed_continuation(s)
            self.assertEqual(generate.call_args.args,(LARGE,'' if diverse else s.feedback()))

    def test_learned_and_table_routes_use_requested_initial_model_and_baselines_keep_theirs(self):
        response={'message':{'content':json.dumps({'sql':reference_queries()['net_total']})}}
        controller=Mock();controller.choose.return_value=('accept',[])
        for policy in ('learned','learned_model_only','table'):
            for initial in ('small','large'):
                with patch('study.runtime.api',return_value=response) as api:
                    result=run_policy(corpus()[0],policy,controller=controller,initial_model=initial)
                self.assertEqual(api.call_args.args[1]['model'],LARGE if initial=='large' else SMALL)
                self.assertEqual(result['initial_model'],initial)
        with patch('study.runtime.api',return_value=response) as api:
            result=run_policy(corpus()[0],'small',initial_model='large')
        self.assertEqual(api.call_args.args[1]['model'],SMALL);self.assertEqual(result['initial_model'],'small')
        with self.assertRaises(ValueError): run_policy(corpus()[0],'large_first')
        with self.assertRaises(ValueError): run_policy(corpus()[0],'learned',initial_model='invalid')

    def test_cli_forwards_initial_model_and_rejects_removed_policy(self):
        task=next(t for t in corpus() if t['split']=='validation')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            argv=['experiment','evaluate','--output',tmp,'--policies','learned','--initial-model','large']
            row=dict(task_id=task['id'],template=task['template'],split='validation',policy='learned',
                     trial=0,fault='clean',penalty=.1,threshold=0,controller_type='logistic_v1',success=False)
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[task]), \
                 patch('study.experiment.provenance',return_value={'source_hashes':{}}),patch('study.experiment.Controller.load'), \
                 patch('study.experiment.run_policy',return_value=row) as run,contextlib.redirect_stdout(io.StringIO()): experiment()
            self.assertEqual(run.call_args.kwargs['initial_model'],'large')
        with patch('sys.argv',['experiment','evaluate','--policies','large_first']), \
             patch('study.experiment.provenance') as provider,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): experiment()
        provider.assert_not_called()

    def test_hashed_test_list_validates_corpus_separation_and_rejects_seen_families(self):
        templates=['new_a','new_b'];config=self.frozen_config(templates)
        tasks=[{'template':t,'split':'test'} for t in templates]+[{'template':'development','split':'train'}]
        self.assertEqual(validate_test_configuration(config,tasks),templates)
        for value in (None,[],['a','a'],[''],[1],('a',)):
            with self.assertRaises(ValueError): template_list_hash(value)
        bad=deepcopy(config);bad['test_templates_sha256']='tampered'
        with self.assertRaisesRegex(ValueError,'hash mismatch'): validate_test_configuration(bad,tasks)
        with self.assertRaisesRegex(ValueError,'differ'): validate_test_configuration(config,tasks[1:])
        with self.assertRaisesRegex(ValueError,'overlap'): validate_test_configuration(config,tasks+[{'template':'new_a','split':'train'}])
        bad=deepcopy(config);bad['groups']['test']=['other']
        with self.assertRaisesRegex(ValueError,'disagree'): validate_test_configuration(bad,tasks)
        bad=deepcopy(config);bad['groups']['validation']=['new_a']
        with self.assertRaisesRegex(ValueError,'overlap'): validate_test_configuration(bad,tasks)
        seen=json.loads(Path('results/main/configuration.json').read_text())['groups']['test']
        with self.assertRaisesRegex(ValueError,'Seen v1'): validate_test_configuration(self.frozen_config(seen),
                    [{'template':t,'split':'test'} for t in seen])

    def test_test_guard_fails_before_provider_but_accepts_valid_frozen_synthetic_split(self):
        task=dict(corpus()[0],id='new_a-0-w0',template='new_a',split='test')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);argv=['experiment','evaluate','--split','test','--output',tmp,'--policies','small']
            config=self.frozen_config(['new_a']);bad=deepcopy(config);bad['test_templates_sha256']='wrong'
            (root/'configuration.json').write_text(json.dumps(bad))
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[task]), \
                 patch('study.experiment.provenance') as provider,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): experiment()
            provider.assert_not_called()
            (root/'configuration.json').write_text(json.dumps(config))
            row=dict(task_id=task['id'],template=task['template'],split='test',policy='small',trial=0,fault='clean',
                     penalty=0,threshold=0,controller_type='none',success=False)
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[task]), \
                 patch('study.experiment.provenance',return_value={'source_hashes':{}}),patch('study.experiment.run_policy',return_value=row) as run, \
                 contextlib.redirect_stdout(io.StringIO()): experiment()
            run.assert_called_once();self.assertEqual(json.loads((root/'test.jsonl').read_text()),row)
            self.assertEqual(json.loads((root/'provenance.json').read_text())['test_templates_sha256'],config['test_templates_sha256'])
            # Recomputing a hash after altering a frozen list cannot replace the
            # list already recorded in this stream's provenance.
            changed=dict(task,id='new_b-0-w0',template='new_b')
            (root/'configuration.json').write_text(json.dumps(self.frozen_config(['new_b'])))
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),patch('study.experiment.corpus',return_value=[changed]), \
                 patch('study.experiment.provenance',return_value={'source_hashes':{}}),patch('study.experiment.run_policy') as run:
                with self.assertRaisesRegex(ValueError,'Frozen test templates changed'): experiment()
            run.assert_not_called()

    def test_freeze_records_test_list_hash_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);argv=['experiment','freeze','--output',tmp]
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),contextlib.redirect_stdout(io.StringIO()): experiment()
            raw=(root/'configuration.json').read_bytes();config=json.loads(raw)
            self.assertEqual(config['test_templates_sha256'],template_list_hash(config['test_templates']))
            with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): experiment()
            self.assertEqual((root/'configuration.json').read_bytes(),raw)

    def test_selection_records_analysis_source_hash_for_both_controller_policies(self):
        for policy,controller in (('learned','logistic_v2'),('table','table')):
            with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
                root=Path(tmp)
                base=dict(task_id='t',template='t',trial=0,split='validation',fault='clean',penalty=0,threshold=0,success=True,effort_seconds=1)
                rows=[dict(base,policy='large'),dict(base,policy=policy,controller_type=controller)]
                (root/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with patch('sys.argv',['analyse','select','--directory',tmp,'--policy',policy,'--controller',controller,'--selection-rule','v2']), \
                     contextlib.redirect_stdout(io.StringIO()): analyse()
                name='selection.json' if policy=='learned' else 'table_selection.json'
                result=json.loads((root/name).read_text())
                self.assertEqual(result['analyse_sha256'],hashlib.sha256(Path('study/analyse.py').read_bytes()).hexdigest())

    def test_family_protocol_unique_sql_counts_and_component_ablations(self):
        spec=family_partition_spec();self.assertEqual(spec['sha256'],FAMILY_SPLIT_SHA256)
        self.assertEqual(set(spec['design']) & set(spec['evaluation']),set())
        # Net-probe ablations now require the explicit v2 public contract field.
        from study.data_v2 import corpus as v2_corpus,REFERENCE_SQL
        result=mutation_audit(v2_corpus(),references=REFERENCE_SQL)
        self.assertEqual(result['distinct_sql_strings'],len({c['sql'] for c in result['cases']}))
        self.assertEqual(sum(p['variants'] for p in result['by_family_partition'].values()),result['variants'])
        design_sql={c['sql'] for c in result['cases'] if c['family_partition']=='design'}
        evaluation_sql={c['sql'] for c in result['cases'] if c['family_partition']=='evaluation'}
        self.assertEqual(result['cross_partition_sql_overlap']['sql_strings'],sorted(design_sql & evaluation_sql))
        for family,stats in result['per_family'].items():
            self.assertEqual(stats['distinct_sql_strings'],len({c['sql'] for c in result['cases'] if c['family']==family}))
            self.assertIn(family,spec[stats['partition']])
        for verifier in ('original','contract_v2'):
            report=result['ablations'][verifier];self.assertEqual(set(report['remove_each_keyword']),set(KEYWORD_RULES))
            self.assertEqual(report['baseline'],result['summary'][verifier])
        self.assertEqual(result['ablations']['original']['remove_net_value_probe']['recall_change_pp'],0)
        self.assertLess(result['ablations']['contract_v2']['remove_net_value_probe']['recall_change_pp'],0)
        self.assertLess(result['ablations']['contract_v2']['remove_each_keyword']['count']['net_probe_kept']['false_alarm_change_pp'],0)
        bad=deepcopy(spec);bad['design']=bad['design'][1:]
        with patch.object(Path,'read_text',return_value=json.dumps(bad)):
            with self.assertRaisesRegex(ValueError,'hash mismatch'): family_partition_spec()
        bad=deepcopy(spec);bad['evaluation'].append(bad['design'][0])
        with patch.object(Path,'read_text',return_value=json.dumps(bad)):
            with self.assertRaisesRegex(ValueError,'Overlapping'): family_partition_spec()

    def test_ablation_removes_only_named_rule_or_net_probe(self):
        evidence={'flagged':True,'concerns':[KEYWORD_RULES['average']],
                  'evidence':[{'check':'preserve_net_value','status':'violation'}]}
        self.assertTrue(ablated_flag(evidence,['average']))
        self.assertTrue(ablated_flag(evidence,remove_net_probe=True))
        self.assertFalse(ablated_flag(evidence,['average'],True))
        evidence['evidence'].append({'check':'add_cancelled','status':'violation'})
        self.assertTrue(ablated_flag(evidence,['average'],True))
        with self.assertRaises(ValueError): ablated_flag(evidence,['unknown'])
        with self.assertRaises(ValueError): ablated_flag({'concerns':['new unregistered rule']})
        baseline={'recall':None,'false_alarm_rate':None}
        self.assertIsNone(ablation_effect(baseline,baseline)['recall_change_pp'])
        self.assertEqual(component_ablations([])['original']['baseline']['wrong_queries'],0)


if __name__=='__main__': unittest.main()
