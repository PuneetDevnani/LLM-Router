"""No-provider tests for runtime reproducibility and oracle-checked audits."""
import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study.data import corpus
from study.diagnostics import detector_audit, small_by_template, reference_queries, main as diagnose
from study.experiment import configuration, main as experiment
from study.mutation_audit import query_variants, family_counts, mutation_audit, main as audit_cli
from study.runtime import Session, SMALL, LARGE, fixed_continuation


class MutationAuditTests(unittest.TestCase):
    def setUp(self):
        guard=patch('study.runtime.api',side_effect=AssertionError('No model calls allowed'))
        guard.start();self.addCleanup(guard.stop)

    def test_default_request_bytes_match_pre_revision_fixture(self):
        response={'message':{'content':'{"sql":"SELECT 1"}'}}
        with patch('study.runtime.api',return_value=response) as api:
            Session(corpus()[0],trial=7).generate(SMALL)
        expected=Path('study/fixtures/session_request_v1.json').read_bytes().rstrip(b'\n')
        self.assertEqual(json.dumps(api.call_args.args[1]).encode(),expected)

    def test_diversification_only_changes_seed_and_starts_after_increment(self):
        bodies={};response={'message':{'content':'{"sql":"SELECT 1"}'}}
        for enabled in (False,True):
            with patch('study.runtime.api',return_value=response) as api:
                s=Session(corpus()[0],trial=7,diversify_seed=enabled)
                for _ in range(3): s.generate(SMALL)
            bodies[enabled]=[call.args[1] for call in api.call_args_list]
        self.assertEqual([b['options']['seed'] for b in bodies[False]],[7,7,7])
        self.assertEqual([b['options']['seed'] for b in bodies[True]],[8,9,10])
        for old,new in zip(bodies[False],bodies[True]):
            new=deepcopy(new);new['options']['seed']=7;self.assertEqual(new,old)

    def test_continuation_keeps_error_feedback_and_uses_fresh_for_flags(self):
        for enabled in (False,True):
            for error in (True,False):
                s=Session(corpus()[0],fresh_escalation=enabled)
                s.state['error']='SQL syntax error' if error else None
                s.state['checks']=[] if error else {'flagged':True}
                with patch.object(s,'generate') as generate: fixed_continuation(s)
                self.assertEqual(generate.call_args.args,(LARGE,'' if enabled and not error else s.feedback()))

    def test_configuration_provenance_and_each_collected_row_record_flags(self):
        task=corpus()[0]
        def generate(s,model,feedback=''):
            s.calls=s.max_calls;s.state['model']=model;s.state['sql']=reference_queries()['net_total'];s.execute()
        for enabled in (False,True):
            args=argparse.Namespace(verifier='original',initial_model='small',fresh_escalation=enabled,
                                    neutral_feedback=enabled,diversify_seed=enabled)
            self.assertEqual(configuration(args,'hash')['dataset_sha256'],'hash')
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);argv=['experiment','collect','--output',tmp]
                if enabled: argv+=['--fresh-escalation','--neutral-feedback','--diversify-seed']
                with patch('sys.argv',argv),patch('study.experiment.V2_ROOT',root), \
                     patch('study.experiment.corpus',return_value=[task]),patch('study.experiment.provenance',return_value={'source_hashes':{}}), \
                     patch.object(Session,'generate',generate),contextlib.redirect_stdout(io.StringIO()): experiment()
                documents=[json.loads((root/name).read_text()) for name in ('configuration.json','provenance.json')]
                for name in ('branches.jsonl','training_states.jsonl'):
                    documents += [json.loads(line) for line in (root/name).read_text().splitlines()]
                self.assertGreater(len(documents),2)
                for d in documents:
                    for k in ('fresh_escalation','neutral_feedback','diversify_seed'): self.assertIs(d[k],enabled)
                frozen=(root/'provenance.json').read_bytes()
                changed=['experiment','collect','--output',tmp]+([] if enabled else ['--diversify-seed'])
                with patch('sys.argv',changed),patch('study.experiment.V2_ROOT',root),patch('study.experiment.provenance') as provenance:
                    with self.assertRaisesRegex(ValueError,'Runtime options changed'): experiment()
                provenance.assert_not_called();self.assertEqual((root/'provenance.json').read_bytes(),frozen)
            with tempfile.TemporaryDirectory() as tmp:
                with patch('sys.argv',['experiment','freeze','--output',tmp]+argv[4:]), \
                     patch('study.experiment.V2_ROOT',Path(tmp)),contextlib.redirect_stdout(io.StringIO()): experiment()
                config=json.loads((Path(tmp)/'configuration.json').read_text())
                for k in ('fresh_escalation','neutral_feedback','diversify_seed'): self.assertIs(config[k],enabled)

    def test_diagnostic_deduplicates_trials_and_lists_empty_templates(self):
        task=corpus()[0];s=Session(task);s.state['sql']=reference_queries()['net_total'];s.execute()
        row=s.result();row['trace'].insert(0,{'kind':'generation','model':LARGE})
        result=detector_audit([task],[],[row,dict(row,trial=99)],verifier='contract_v2')
        self.assertEqual(result['correct_large']['counts']['true_negative'],1)
        self.assertEqual(result['reference_sql']['per_template']['net_total']['true_negative'],1)
        all_tasks=[t for t in corpus() if t['split'] in ('train','validation')]
        result=detector_audit(all_tasks,[],[],verifier='original')
        self.assertEqual(len(result['reference_sql']['per_template']),9)
        self.assertTrue(all(c['true_negative']==20 for c in result['reference_sql']['per_template'].values()))
        counts=small_by_template([],[]);templates={t['template'] for t in all_tasks}
        for population in counts.values():
            self.assertEqual(set(population),templates)
            self.assertTrue(all(c=={'runs':0,'successes':0} for c in population.values()))

    def test_frozen_primary_hashes_including_original_detector(self):
        hashes=json.loads(Path('study/frozen_main_hashes.json').read_text())
        self.assertIn('results/main/detector_diagnostic.json',hashes)
        self.assertGreaterEqual(len(hashes),11)
        for filename,expected in hashes.items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(),expected,filename)

    def test_active_no_argument_diagnostic_writes_v2_only(self):
        with patch('sys.argv',['diagnostics']),patch('study.diagnostics.read_rows',return_value=[]), \
             patch('study.diagnostics.corpus',return_value=[corpus()[0]]),patch.object(Path,'write_text',autospec=True,return_value=0) as write, \
             patch.object(Path,'mkdir'),contextlib.redirect_stdout(io.StringIO()): diagnose()
        self.assertEqual(write.call_args.args[0],Path('results/v2/detector_diagnostic.json'))
        with patch('sys.argv',['diagnostics','--directory','results/main']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): diagnose()

    def test_all_train_validation_references_have_verified_mutants_and_equivalents(self):
        result=mutation_audit()
        self.assertEqual(result['task_instances'],90)
        self.assertEqual(set(result['per_template']),{t['template'] for t in corpus() if t['split'] in ('train','validation')})
        self.assertNotIn('execution_error',result['oracle_relations'])
        for template in result['per_template']:
            records=[r for r in result['cases'] if r['template']==template]
            self.assertTrue(any(r['kind']=='mutated' and r['oracle_relation']=='differs' for r in records),template)
            self.assertTrue(any(r['kind']=='equivalent' and r['oracle_relation']=='matches' for r in records),template)
        for record in result['cases']:
            if record['kind'] in ('reference','equivalent'): self.assertEqual(record['oracle_relation'],'matches')
            self.assertNotIn('expected',record['checks'])
        for verifier in ('original','contract_v2'):
            total=result['summary'][verifier]
            self.assertEqual(total['wrong_queries']+total['correct_queries'],result['variants'])
            self.assertEqual(total['true_positive']+total['false_negative'],result['oracle_relations']['differs'])
            self.assertEqual(result['per_family']['reference'][verifier]['false_positive'],0)
        # A correct algebraic average can trigger the unchanged lexical concern.
        self.assertGreater(result['per_family']['average_expression']['original']['false_positive'],0)
        self.assertGreater(result['per_family']['count_expression']['contract_v2']['false_positive'],0)
        test=next(t for t in corpus() if t['split']=='test')
        with self.assertRaises(ValueError): query_variants(test,'SELECT 1',['x'])
        with self.assertRaises(ValueError): mutation_audit([test])
        one=mutation_audit([corpus()[0],corpus()[1]])
        self.assertEqual(one['task_instances'],1)

    def test_family_counts_use_oracle_labels_and_report_inert_and_broken_mutants(self):
        cases=[dict(kind='mutated',oracle_relation='differs',checks={'original':{'flagged':True}}),
               dict(kind='mutated',oracle_relation='matches',checks={'original':{'flagged':True}}),
               dict(kind='mutated',oracle_relation='execution_error',checks={})]
        counts=family_counts(cases,'original')
        self.assertEqual((counts['recall'],counts['false_alarm_rate'],counts['inert_mutations'],counts['execution_errors']),(1,1,1,1))
        self.assertIsNone(family_counts([],'original')['recall'])

    def test_audit_cli_writes_v2_and_rejects_primary_output(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            target=Path(tmp)/'audit.json'
            with patch('sys.argv',['audit','--output',str(target)]),patch('study.mutation_audit.corpus',return_value=[corpus()[0]]), \
                 contextlib.redirect_stdout(io.StringIO()): audit_cli()
            data=json.loads(target.read_text());self.assertEqual(data['task_instances'],1)
            self.assertIn('study/mutation_audit.py',data['source_hashes'])
        with patch('sys.argv',['audit','--output','results/main/mutation_audit.json']), \
             patch('study.mutation_audit.mutation_audit') as audit,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): audit_cli()
        audit.assert_not_called()


if __name__=='__main__': unittest.main()
