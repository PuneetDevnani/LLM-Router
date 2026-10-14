"""Pre-inference amendments: offline authoring fixtures and mocked inference."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from study.data_v2 import corpus,REFERENCE_SQL
from study.runtime import Session,SMALL
from study.experiment import configuration,run_policy
from study.generation_cache import GenerationCache,request_bytes,request_sha256,record_sha256
from study.preinference import (ROOT,VERIFIERS,PROBES,digest,value_hash,check_preinference,
                               constraint_audit,reference_audit,targeted_design_variants,detector_audit,amend,main)
from study.verifier import insert_probe_orders,change_discount_values,contract_boundary_probes,nonnegative_range_probe


class AmendmentTests(unittest.TestCase):
    def task(self,name='net_total',seed=0):
        return next(t for t in corpus() if t['template']==name and t['seed']==seed and t['id'].endswith('w0'))

    def response(self,sql):
        return {'message':{'content':json.dumps({'sql':sql})},'load_duration':1_000_000_000,
                'prompt_eval_duration':500_000_000,'eval_duration':1_500_000_000,'prompt_eval_count':12,'eval_count':8}

    def checked(self,task,sql,version='contract_v3'):
        s=Session(task,verifier=version);s.state['sql']=sql;s.execute();s.check();return s

    def test_all_v2_references_never_flagged_by_any_verifier_all_wordings(self):
        with patch('study.runtime.api',side_effect=AssertionError('No model calls')):
            for task in corpus():
                for v in VERIFIERS:
                    s=self.checked(task,REFERENCE_SQL[task['template']],v)
                    self.assertIsNone(s.state['error'],task['id'])
                    self.assertTrue(s.result()['success'],task['id'])
                    self.assertFalse(s.state['checks']['flagged'],(task['id'],v,s.state['checks']))
        for name in ('discount_by_region','missing_count'):
            s=self.checked(self.task(name),REFERENCE_SQL[name],'contract_v2')
            self.assertFalse(any(e['check']=='preserve_net_value' for e in s.state['checks']['evidence']))
        t=deepcopy(self.task());t['contract'].pop('net_value')
        s=self.checked(t,REFERENCE_SQL['net_total'],'contract_v2')
        self.assertFalse(any(e['check']=='preserve_net_value' for e in s.state['checks']['evidence']))

    def test_omission_constraints_change_at_least_seventy_percent_of_seeds(self):
        coverage=constraint_audit()
        self.assertEqual(set(coverage['month_totals']),{'window','start','end'})
        for values in coverage.values():
            for value in values.values():
                self.assertEqual(value['seeds'],10);self.assertGreaterEqual(value['fraction'],.7)
        answers=[t['expected']['regions'] for t in corpus() if t['template']=='having_regions' and t['id'].endswith('w0')]
        self.assertTrue(any(answers));self.assertTrue(any(not a for a in answers))

    def test_new_probes_detect_design_faults_and_accept_alternatives(self):
        variants=[]
        for name in ('january','date_window','summary','large_net','net_total','region_total'):
            task=self.task(name);variants+= [(task,family,sql) for family,sql in targeted_design_variants(task)]
            self.assertFalse(self.checked(task,REFERENCE_SQL[name]).state['checks']['flagged'])
        expected={'inclusive_end_boundary':'date_boundary_records','strict_net_boundary':'net_count_boundary_records',
                  'cancelled_null_disjunction':'cancelled_null_discount','spurious_discount_dependence':'discount_irrelevance',
                  'negative_result':'public_nonnegative_range'}
        for task,family,sql in variants:
            evidence=self.checked(task,sql).state['checks']['evidence']
            self.assertTrue(any(e['check']==expected[family] and e['status']=='violation' for e in evidence),family)
        alt=REFERENCE_SQL['region_total'].replace("o.status='completed'","o.status!='cancelled'")
        self.assertFalse(self.checked(self.task('region_total'),alt).state['checks']['flagged'])
        for task in corpus(): self.assertIsInstance(task['contract']['discount_irrelevant'],bool)
        # Probe permission comes only from the contract, never a SQL keyword.
        t=deepcopy(self.task('region_total'));t['contract']['discount_irrelevant']=False
        e=self.checked(t,REFERENCE_SQL['region_total']).state['checks']['evidence']
        self.assertFalse(any(p['check']=='discount_irrelevance' for p in e))
        with self.assertRaises(ValueError): targeted_design_variants(self.task('month_totals'))

    def test_nonnegative_probe_optional_group_labels_null_and_all_policy_contracts(self):
        output={'columns':['region','value'],'rows':[[-100,2],['west',0],['north',None]]}
        self.assertIsNone(nonnegative_range_probe(output,{'grouped':True}))
        self.assertEqual(nonnegative_range_probe(output,{'grouped':True,'nonnegative':True})['status'],'consistent')
        output['rows'][1][1]=-1
        self.assertEqual(nonnegative_range_probe(output,{'grouped':True,'nonnegative':True})['status'],'violation')
        self.assertEqual(nonnegative_range_probe({'rows':[[None]]},{'nonnegative':True})['status'],'inconclusive')
        self.assertEqual(nonnegative_range_probe({'rows':[[float('inf')]]},{'nonnegative':True})['status'],'violation')
        self.assertIsNone(nonnegative_range_probe(output,{'nonnegative':False}))
        task=self.task();bodies=[]
        controller=SimpleNamespace(choose=lambda *a,**kw:('accept',[]))
        router=SimpleNamespace(choose=lambda public:(SMALL,[]))
        for policy in ('small','large','rules','escalate_default','escalate_fresh','learned','table','learned_model_only','task_routing'):
            with patch('study.runtime.api',return_value=self.response(REFERENCE_SQL['net_total'])) as api:
                run_policy(task,policy,controller=controller,router=router)
            bodies.append(api.call_args_list[0].args[1]['messages'])
        self.assertTrue(all(b==bodies[0] for b in bodies));self.assertIn('nonnegative',json.dumps(bodies[0]))
        for version in ('original','contract_v2'):
            s=self.checked(task,'SELECT -1',version)
            self.assertFalse(any(e['check']=='public_nonnegative_range' for e in s.state['checks']['evidence']))

    def test_probe_transforms_preserve_null_positions_and_use_public_boundaries(self):
        s=Session(self.task());con,_=s.environment(0)
        try:
            nulls=con.execute('SELECT id FROM orders WHERE discount IS NULL').fetchall()
            change_discount_values(con)
            self.assertEqual(con.execute('SELECT id FROM orders WHERE discount IS NULL').fetchall(),nulls)
            insert_probe_orders(con,[('2026-01-15','cancelled',123,None)])
            self.assertEqual(con.execute('SELECT status,amount,discount FROM orders ORDER BY id DESC LIMIT 1').fetchone(),('cancelled',123,None))
        finally: con.close()
        evidence={'evidence':[]};contract=deepcopy(s.public['contract']);contract['monetary_threshold']=5000
        s.state['sql']=REFERENCE_SQL['net_total'];s.execute()
        contract_boundary_probes(evidence,s.state['sql'],s.state['output'],0,contract,s.environment)
        self.assertEqual(evidence['evidence'],[]) # A sum is not a count-boundary contract.

    def test_cache_live_and_replayed_sql_usage_and_database_execution(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            path=Path(tmp)/'cache';sql=REFERENCE_SQL['net_total']
            s=Session(self.task(),generation_cache=True,cache_directory=path)
            with patch('study.runtime.api',return_value=self.response(sql)) as provider: s.generate(SMALL)
            provider.assert_called_once();body=provider.call_args.args[1]
            self.assertEqual(request_bytes(body),json.dumps(body).encode())
            self.assertEqual(request_sha256(body),hashlib.sha256(request_bytes(body)).hexdigest())
            s2=Session(self.task(seed=1),generation_cache=True,cache_directory=path)
            with patch('study.runtime.api',side_effect=AssertionError('Cache hit must not infer')): s2.generate(SMALL)
            self.assertEqual(s.state['sql'],s2.state['sql']);self.assertEqual(s.trace[0]['usage'],s2.trace[0]['usage'])
            self.assertNotEqual(s.state['report'],s2.state['report']);self.assertTrue(s2.result()['success'])
            self.assertEqual(s2.result()['replayed_model_effort_seconds'],3)
            self.assertEqual((s2.result()['cache_hits'],s2.result()['live_model_calls']),(1,0))
            self.assertGreaterEqual(s2.effort(),3)
            transient=Session(self.task(),fault='transient',generation_cache=True,cache_directory=path)
            with patch('study.runtime.api',side_effect=AssertionError('No inference')): transient.generate(SMALL)
            self.assertEqual(transient.state['error'],'tool_timeout');transient.apply('retry')
            self.assertTrue(transient.result()['success']);self.assertEqual(transient.operations,2)
            # Returned records are isolated; no mutable response aliases.
            cache=GenerationCache(path);record=cache.lookup(body);record['response']['message']['content']='broken'
            self.assertEqual(cache.lookup(body)['record_sha256'],record_sha256(cache.lookup(body)))
            cache.bind({'models':{'tag':'digest'},'version':'mock'});cache.bind({'models':{'tag':'digest'},'version':'mock'})
            with self.assertRaises(ValueError): cache.bind({'models':{'tag':'different'}})
            for change in ({'model':'another'},{'options':{**body['options'],'seed':1}},{'options':{**body['options'],'temperature':.7}},
                           {'messages':body['messages']+[{'role':'user','content':'feedback'}]}):
                self.assertIsNone(cache.lookup({**body,**change}))
            f=path/(request_sha256(body)+'.json');raw=json.loads(f.read_text());raw['response']['eval_duration']=0;f.write_text(json.dumps(raw))
            with self.assertRaises(ValueError): cache.lookup(body)
            with self.assertRaises(RuntimeError),patch('study.runtime.api',side_effect=AssertionError('No inference')):
                Session(self.task(),generation_cache=True,cache_directory=path).generate(SMALL)
        with self.assertRaises(ValueError): GenerationCache('results/main')

    def test_default_cache_off_and_consistency_replays_its_distinct_requests(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            path=Path(tmp)/'cache';task=self.task();sql=REFERENCE_SQL['net_total']
            default=Session(task,cache_directory=path)
            with patch('study.runtime.api',return_value=self.response(sql)): default.generate(SMALL)
            self.assertFalse(path.exists());self.assertNotIn('cache_hit',default.trace[0])
            for run in range(2):
                s=Session(task,consistency_verification=True,generation_cache=True,cache_directory=path)
                with patch('study.runtime.api',return_value=self.response(sql)) as provider:
                    s.generate(SMALL);s.check_consistency()
                self.assertEqual(provider.call_count,4 if run==0 else 0)
                result=s.result();self.assertEqual(result['logical_model_calls'],4)
                self.assertEqual(result['cache_hits'],0 if run==0 else 4)
                self.assertGreaterEqual(result['effort_seconds'],12)
            with patch('study.runtime.api',return_value=self.response(sql)):
                result=run_policy(task,'small',generation_cache=True,cache_directory=path)
            self.assertTrue(result['generation_cache'])
            self.assertEqual(default.last_request_sha256,result['trace'][0]['request_sha256'])

    def test_cache_does_not_store_provider_outages_and_replays_invalid_actions(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            path=Path(tmp)/'cache';body={'model':'test'};cache=GenerationCache(path)
            with self.assertRaises(OSError): cache.generate(body,lambda:(_ for _ in ()).throw(OSError('provider unavailable')))
            self.assertFalse(path.exists())
            with self.assertRaises(RuntimeError): cache.generate(body,lambda:self.response('SELECT 1'),cache_only=True)
            for run in range(2):
                s=Session(self.task(),generation_cache=True,cache_directory=path)
                with patch('study.runtime.api',return_value={'message':{'content':'not JSON'}}) as api: s.generate(SMALL)
                self.assertEqual(api.call_count,1 if run==0 else 0)
                self.assertTrue(s.state['error'].startswith('invalid_model_action'))

    def test_consistency_inherits_nonfinite_guard_and_oracle_sums_cache_counters(self):
        s=Session(self.task(),verifier='contract_v3',consistency_verification=True)
        responses=[self.response(REFERENCE_SQL['net_total'])]*3+[self.response('SELECT 1e999')]
        with patch('study.runtime.api',side_effect=responses): s.generate(SMALL);s.check_consistency()
        self.assertTrue(s.state['checks']['flagged'])
        self.assertEqual(s.state['checks']['samples'][-1]['error'],'non-finite SQL result')
        json.dumps(s.result(),allow_nan=False)
        from study.workflow import oracle_outcome,cache_accounting
        a=dict(success=False,policy='small',fault='clean',effort_seconds=3,calls=1,operations=1,wall_seconds=.1,
               logical_model_calls=1,live_model_calls=0,cache_hits=1,replayed_model_effort_seconds=3)
        b=dict(a,success=True,policy='large',effort_seconds=4,live_model_calls=1,cache_hits=0,replayed_model_effort_seconds=0)
        with patch('study.workflow.run_policy',side_effect=[a,b]): result=oracle_outcome(self.task(),0,'clean',{'generation_cache':True})
        self.assertEqual((result['logical_model_calls'],result['live_model_calls'],result['cache_hits']),(2,1,1))
        self.assertEqual(result['replayed_model_effort_seconds'],3)
        accounting=cache_accounting([result])[0];self.assertEqual(accounting['logical_model_calls'],2)
        self.assertEqual(accounting['charged_effort_seconds'],7)

    def test_live_sample_is_twenty_fixed_requests_with_cache_bypass_and_mismatch_reporting(self):
        from study.workflow import RUNTIME,live_rerun_sample,live_sample_summary
        from study.headroom import frozen_development,timing_tasks
        from study.test_workflow import WorkflowTests
        from study.freeze_protocol import validate_freeze
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();runtime={**RUNTIME,'generation_cache':True,'cache_directory':str(root/'cache')}
            tasks,_=frozen_development(ROOT/'results/v2/corpus_v2');chosen=timing_tasks(tasks)
            # First fills are mocked provider calls; the twenty later calls
            # deliberately bypass replay and include one disagreeing response.
            for task in chosen:
                for model in (runtime['small_model'],runtime['large_model']):
                    s=Session(task,max_calls=1,**{k:v for k,v in runtime.items() if k!='max_iterations'})
                    with patch('study.runtime.api',return_value=self.response(REFERENCE_SQL[task['template']])): s.generate(model)
            spec=WorkflowTests().frozen_fixture(root,runtime)
            provider={k:spec[k] for k in ('models','provider_version')}
            responses=[self.response(REFERENCE_SQL[t['template']]) for t in chosen for _ in range(2)];responses[0]=self.response('SELECT 1')
            with patch('study.workflow.verify_provider',return_value=provider),patch('study.runtime.api',side_effect=responses) as live:
                result=live_rerun_sample(root);self.assertEqual(live.call_count,20)
                self.assertEqual(live_rerun_sample(root),json.loads(json.dumps(result)));self.assertEqual(live.call_count,20)
            self.assertEqual(result['sql_mismatches'],1);self.assertEqual(result['result_mismatches'],1)
            self.assertTrue(all(r['same_request'] for r in result['rows']))
            summary=live_sample_summary(root,spec);self.assertEqual(summary['requests'],20);self.assertEqual(summary['sql_mismatches'],1)
            validate_freeze(root) # Live sample did not replace any cache record.
            rows=result['rows'];result['rows']=rows[:-1];(root/'LIVE_RERUN_SAMPLE.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError): live_sample_summary(root,spec)
            result['rows']=rows;(root/'LIVE_RERUN_SAMPLE.json').write_text(json.dumps(result))
            cached=next((root/'cache').glob('[0-9a-f]*.json'));record=json.loads(cached.read_text());record['recorded_wall_seconds']+=1
            record['record_sha256']=record_sha256(record);cached.write_text(json.dumps(record))
            with self.assertRaises(ValueError): validate_freeze(root)

    def test_final_validation_guard_rejects_sparse_duplicate_and_missing_baseline(self):
        from study.test_workflow import WorkflowTests
        from study.freeze_protocol import validate_selection
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();setting,_=WorkflowTests().selection_fixture(root,'table')
            path=root/'table/validation.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
            sparse=[r for r in rows if r['policy']=='large'];grid=[r for r in rows if r['policy']=='table']
            # Each task survives at one point, and every point survives in union.
            sparse+= [grid[i*9+(i%9)] for i in range(40)]
            for invalid in (sparse,rows+[grid[0]],[r for r in rows if r['policy']!='large']):
                path.write_text('\n'.join(map(json.dumps,invalid))+'\n')
                with self.assertRaises(ValueError): validate_selection(setting,ROOT/'results/v2/corpus_v2')

    def test_reference_audit_and_development_family_report_have_correct_populations(self):
        with patch('study.runtime.api',side_effect=AssertionError('No model calls')):
            refs=reference_audit();self.assertEqual(refs['instances'],240);self.assertEqual(refs['distinct_sql_strings'],24)
            for v in VERIFIERS: self.assertEqual(refs['summary'][v]['false_positive'],0)
            result=detector_audit()
        self.assertEqual(result['task_instances'],120);self.assertEqual(set(result['probe_ablations']),set(PROBES))
        self.assertEqual(set(result['by_family_partition']),{'design','evaluation'})
        self.assertTrue(all(c['split'] in ('train','validation') for c in result['cases']))
        self.assertTrue(all(len(r['sql']) for r in result['additional_design_cases']['cases']))

    def test_amendment_guard_snapshot_and_preregistration_hashes(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();dataset=root/'dataset';dataset.mkdir()
            archive=ROOT/'results/v2/amendments/2026-10-13/corpus_before'
            for name in ('configuration.json','PREREGISTRATION.json','corpus.json'): shutil.copyfile(archive/name,dataset/name)
            design=root/'amendments/2026-10-13/PROBE_DESIGN.json';design.parent.mkdir(parents=True);design.write_text('{}')
            with patch('study.preinference.V2',root),patch('study.runtime.api',side_effect=AssertionError('No inference')):
                result=amend(dataset);self.assertEqual(amend(dataset),result)
                self.assertEqual(digest(dataset/'corpus.json'),result['dataset_sha256'])
                self.assertEqual(value_hash({k:v for k,v in result.items() if k!='preregistration_sha256'}),result['preregistration_sha256'])
                self.assertEqual(digest(ROOT/'V2_PROTOCOL.md'),result['protocol_sha256'])
                self.assertEqual(len(json.loads((dataset/'template_hashes.json').read_text())),24)
                self.assertTrue((root/'amendments/2026-10-13/corpus_before/corpus.json').exists())
                model=root/'profile';model.mkdir();(model/'profile.jsonl').write_text('{"model":"mock"}\n')
                metadata=model/'provenance.json';metadata.write_text(json.dumps({'dataset_sha256':'historical'}))
                check_preinference(dataset)
                metadata.write_text(json.dumps({'dataset_sha256':result['dataset_sha256']}))
                with self.assertRaises(ValueError): check_preinference(dataset)
                metadata.unlink();(model/'profile.jsonl').unlink()
                (root/'test.jsonl').write_text('');check_preinference()
                (root/'TEST_STARTED.json').write_text('{}')
                with self.assertRaises(ValueError): check_preinference()
                with self.assertRaises(ValueError): amend(dataset)

    def test_configuration_and_cli_record_cache_without_inference(self):
        from argparse import Namespace
        args=Namespace(corpus='v2',verifier='contract_v3',initial_model='small',fresh_escalation=True,
                       neutral_feedback=True,diversify_seed=True,generation_cache=True,cache_directory='results/v2/cache')
        cfg=configuration(args,'dataset');self.assertTrue(cfg['generation_cache'])
        self.assertEqual(cfg['cache_directory'],str((ROOT/'results/v2/cache').resolve()))
        with patch('sys.argv',['preinference','amend']),patch('study.preinference.amend',return_value={'ok':True}) as action,contextlib.redirect_stdout(io.StringIO()):
            main();action.assert_called_once()
        with patch('sys.argv',['preinference','audit','--output','results/main/audit.json']),patch('study.preinference.check_preinference'),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): main()

    def test_split_sign_amendment_preserves_immediate_snapshot_and_holdout_lists(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();dataset=root/'dataset'
            shutil.copytree(ROOT/'results/v2/amendments/2026-10-13/nonnegative-split/corpus_before',dataset)
            old=json.loads((dataset/'PREREGISTRATION.json').read_text())
            cfg=json.loads((dataset/'configuration.json').read_text())
            design=root/'amendments/2026-10-13/nonnegative-split/PROBE_DESIGN.json'
            design.parent.mkdir(parents=True);design.write_text('{}')
            with patch('study.preinference.V2',root):
                result=amend(dataset,revision='nonnegative-split')
                self.assertEqual(amend(dataset,revision='nonnegative-split'),result)
                with self.assertRaises(ValueError): amend(dataset,revision='invalid')
            self.assertEqual(result['previous_dataset_sha256'],old['dataset_sha256'])
            current=json.loads((dataset/'configuration.json').read_text())
            self.assertIn('completion_rate',current['groups']['validation'])
            for split in ('test','schema_shift'):
                self.assertEqual(current[split+'_templates'],cfg[split+'_templates'])
                self.assertEqual(current[split+'_templates_sha256'],cfg[split+'_templates_sha256'])
            self.assertEqual(len(result['amendments']),8)


if __name__=='__main__': unittest.main()
