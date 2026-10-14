"""Offline corpus authoring checks; no controller/model test outcomes."""
from copy import deepcopy
import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pilot.environment import query, correct
from study.data import corpus as v1_corpus, public, report_from_result
from study.data_v2 import (GROUPS, REFERENCE_SQL, corpus, database, order_records,
                          inventory_records, oracle, equivalent_sql, wrong_sql)
from study.runtime import Session, SMALL
from study.controller import features
from study.experiment import main, study_corpus, study_groups, template_list_hash, validate_test_configuration


class CorpusV2Tests(unittest.TestCase):
    def setUp(self):
        guard=patch('study.runtime.api',side_effect=AssertionError('Corpus checks must never call a model'))
        guard.start();self.addCleanup(guard.stop)

    def test_all_reference_equivalent_and_wrong_sql_against_record_oracles(self):
        self.assertNotIn('REFERENCE_SQL',inspect.getsource(oracle))
        self.assertNotIn('query(',inspect.getsource(oracle))
        for task in corpus():
            con,data=database(task['seed'],schema=task['schema_variant'])
            try:
                self.assertEqual(oracle(task['template'],data),task['expected'])
                for sql in (REFERENCE_SQL[task['template']],equivalent_sql(task['template'])):
                    report=report_from_result(query(con,sql),task['contract'])
                    self.assertTrue(correct(report,task['expected']),task['id'])
                report=report_from_result(query(con,wrong_sql(task)),task['contract'])
                self.assertFalse(correct(report,task['expected']),task['id'])
            finally: con.close()

    def test_splits_versions_seed_coverage_and_wording_grouping(self):
        tasks=corpus();self.assertEqual(len(tasks),480)
        self.assertEqual({split:len(names) for split,names in GROUPS.items()},
                         {'train':8,'validation':4,'test':8,'schema_shift':4})
        old_seen={'january','region_net','large_net'}
        self.assertTrue(old_seen<=set(GROUPS['train']));self.assertFalse(old_seen & set(GROUPS['test']))
        self.assertFalse(set(GROUPS['test']) & {t['template'] for t in v1_corpus()})
        for template in {t['template'] for t in tasks}:
            rows=[t for t in tasks if t['template']==template]
            self.assertEqual(len(rows),20);self.assertEqual({r['seed'] for r in rows},set(range(10)))
            self.assertEqual(len({r['split'] for r in rows}),1)
            self.assertEqual(len({r['prompt'] for r in rows}),2)
        self.assertEqual(study_corpus('v1'),v1_corpus());self.assertEqual(study_corpus('v2'),tasks)
        self.assertEqual(study_groups('v2'),GROUPS)
        self.assertIn('distinct_customers',GROUPS['train']);self.assertNotIn('completion_rate',GROUPS['train'])
        self.assertIn('completion_rate',GROUPS['validation']);self.assertNotIn('distinct_customers',GROUPS['validation'])
        for load in (study_corpus,study_groups):
            with self.assertRaises(ValueError): load('invalid')

    def test_seed_changes_structure_and_previously_constant_answers(self):
        records=[order_records(seed) for seed in range(10)]
        for field in ('status','discount','order_date'):
            patterns=[tuple(r[field] is None if field=='discount' else r[field] for r in data['orders']) for data in records]
            self.assertGreater(len(set(patterns)),1,field)
        for template in ('completion_rate','missing_count','date_window','summary'):
            values=[oracle(template,data) for data in records]
            if template=='summary': values=[r['count'] for r in values]
            else: values=[next(iter(r.values())) for r in values]
            self.assertGreater(len(set(values)),1,template)
        self.assertEqual(order_records(3),order_records(3))
        self.assertEqual(inventory_records(3),inventory_records(3))
        with self.assertRaises(ValueError): oracle('invalid',records[0])
        with self.assertRaises(ValueError): oracle('invalid',inventory_records(0))

    def test_schema_environments_mutations_and_public_boundary(self):
        tasks=corpus()
        for schema in ('orders','inventory'):
            task=next(t for t in tasks if t['schema_variant']==schema)
            s=Session(task);s.state['sql']=REFERENCE_SQL[task['template']];s.execute()
            self.assertIsNone(s.state['error']);self.assertTrue(s.result()['success'])
            s.check();self.assertFalse(s.state['checks']['flagged'])
        for mutation in ('add_cancelled','zero_missing_discount'):
            con,data=database(0,mutation);con.close()
            if mutation=='add_cancelled': self.assertEqual(len(data['orders']),84)
            else: self.assertTrue(all(r['discount'] is not None for r in data['orders']))
        for settings in ({'schema':'invalid'},{'schema':'inventory','mutation':'add_cancelled'},{'mutation':'invalid'}):
            with self.assertRaises(ValueError): database(0,**settings)
        task=tasks[0];changed=deepcopy(task)
        changed.update(id='PRIVATE_CANARY',template='PRIVATE_CANARY',split='PRIVATE_CANARY',seed=999,expected={'PRIVATE_GOLD':123})
        self.assertEqual(features(Session(task)),features(Session(changed)))
        response={'message':{'content':'{"sql":"SELECT 1"}'}}
        with patch('study.runtime.api',return_value=response) as api:
            Session(changed).generate(SMALL)
        body=json.dumps(api.call_args.args[1])
        self.assertNotIn('PRIVATE_CANARY',body);self.assertNotIn('PRIVATE_GOLD',body)
        self.assertEqual(set(public(task)),{'prompt','schema','contract'})
        for task in tasks:
            if task['contract']['aggregation']=='ratio':
                self.assertEqual(task['contract']['valid_range'],[0,100])
                self.assertIsInstance(task['contract']['denominator'],str)

    def test_net_probe_respects_explicit_non_net_sum_contract(self):
        task=next(t for t in corpus() if t['template']=='discount_by_region')
        s=Session(task,verifier='contract_v2');s.state['sql']=REFERENCE_SQL[task['template']];s.execute();s.check()
        self.assertFalse(s.state['checks']['flagged'])
        self.assertFalse(any(e['check']=='preserve_net_value' for e in s.state['checks']['evidence']))
        task=next(t for t in corpus() if t['template']=='net_total')
        s=Session(task,verifier='contract_v2');s.state['sql']=REFERENCE_SQL[task['template']];s.execute();s.check()
        self.assertFalse(s.state['checks']['flagged'])
        self.assertTrue(any(e['check']=='preserve_net_value' for e in s.state['checks']['evidence']))

    def test_v2_freeze_hashes_both_unseen_splits_and_is_read_only_on_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=['experiment','freeze','--corpus','v2','--output',tmp]
            with patch('sys.argv',args),patch('study.experiment.V2_ROOT',root),contextlib.redirect_stdout(io.StringIO()): main()
            raw=(root/'configuration.json').read_bytes();config=json.loads(raw)
            self.assertEqual(config['corpus_version'],'v2')
            self.assertEqual(config['dataset_sha256'],hashlib.sha256((root/'corpus.json').read_bytes()).hexdigest())
            for split in ('test','schema_shift'):
                self.assertEqual(config[split+'_templates_sha256'],template_list_hash(sorted(GROUPS[split])))
                validate_test_configuration(config,corpus(),split)
            with patch('sys.argv',args),patch('study.experiment.V2_ROOT',root),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main()
            self.assertEqual((root/'configuration.json').read_bytes(),raw)
            with patch('sys.argv',['experiment','evaluate','--corpus','v2','--split','test','--output',tmp]), \
                 patch('study.experiment.V2_ROOT',root),patch('study.experiment.provenance') as provider,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main()
            provider.assert_not_called()


if __name__=='__main__': unittest.main()
