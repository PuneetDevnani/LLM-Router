"""Workflow integration uses synthetic observations and mocked providers only."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from study.freeze_protocol import (ROOT,confined,digest,canonical,source_files,write_freeze,validate_freeze,
    validate_selection,validate_leaf,begin_test,claim_job,complete_job,verify_completed_result)
from study.workflow import (RUNTIME,BASELINES,initialise,verify_provider,profile,experiment_args,run_module,
    append_unique,develop,final_freeze,oracle_outcome,heldout,decomposition,analyse_results,main)
from study.headroom import frozen_development
from study.data_v2 import REFERENCE_SQL
from study.runtime import Session
from study.analyse import select_operating_point


class WorkflowTests(unittest.TestCase):
    def test_profile_timing_reuse_gate_and_provider_provenance(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();plan,tasks=initialise(root,ROOT/'results/v2/corpus_v2')
            self.assertEqual(initialise(root,plan['corpus_directory'])[0],plan)
            tags={'models':[{'name':RUNTIME['small_model'],'digest':'s'},{'name':RUNTIME['large_model'],'digest':'l'}]}
            with patch('study.workflow.api',side_effect=[tags,{'version':'mock'}]):
                provider=verify_provider(plan)
            calls=[]
            def generate(session,model,feedback='',sampling=None):
                calls.append((session.task['id'],model));session.calls+=1;session.state['model']=model
                good=model==RUNTIME['large_model'] or session.task['template'] in ('net_total','average_net','completion_rate','large_net')
                session.state['sql']=REFERENCE_SQL[session.task['template']] if good else 'SELECT 1';session.execute()
            with patch('study.workflow.verify_provider',return_value=provider),patch.object(Session,'generate',generate),contextlib.redirect_stdout(io.StringIO()):
                timing=profile(root,plan,tasks,True);self.assertEqual(len(calls),20)
                report=profile(root,plan,tasks);self.assertEqual(len(calls),240)
                self.assertTrue(report['suitable_pair']);profile(root,plan,tasks);self.assertEqual(len(calls),240)
            self.assertEqual(timing['timing_tasks'],10);self.assertGreater(timing['conservative_model_call_cap'],0)
            self.assertIn('--initial-routing',experiment_args(root,plan,'large','table'))
            with patch('study.workflow.subprocess.run') as runner,contextlib.redirect_stdout(io.StringIO()): run_module('study.experiment',['train'])
            self.assertTrue(runner.call_args.kwargs['check'])
            row={'task_id':'a','fault':'clean','trial':0,'policy':'large','penalty':0}
            path=root/'unique.jsonl';append_unique(path,[row,row]);append_unique(path,[row]);self.assertEqual(len(path.read_text().splitlines()),1)
            with self.assertRaises(ValueError): confined(ROOT/'results/main')
            self.assertTrue(source_files());self.assertEqual(canonical({'b':1,'a':2}),'{'+'"a":2,"b":1'+'}')

    def selection_fixture(self,root,kind):
        leaf=root/kind;leaf.mkdir();tasks,_=frozen_development(ROOT/'results/v2/corpus_v2')
        val=[t for t in tasks if t['split']=='validation'];policy='table' if kind=='table' else 'learned'
        rows=[]
        for task in val:
            rows.append(dict(task_id=task['id'],template=task['template'],split='validation',fault='clean',trial=0,policy='large',
                             penalty=0,threshold=0,controller_type='none',success=True,effort_seconds=2))
            for p in (0,.15,.3):
                for t in (0,.1,.2):
                    rows.append(dict(task_id=task['id'],template=task['template'],split='validation',fault='clean',trial=0,policy=policy,
                        penalty=p,threshold=t,controller_type=kind,success=True,effort_seconds=1+p))
        (leaf/'validation.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
        (leaf/'branches.jsonl').write_text('{"split":"train"}\n');sha=digest(leaf/'branches.jsonl')
        model=leaf/('table_controller.json' if kind=='table' else 'controller_logistic_v2.json')
        model.write_text(json.dumps({'training_branches_sha256':sha,'controller_type':kind}))
        (leaf/('training_cv_'+kind+'.json')).write_text(json.dumps({'training_branches_sha256':sha}))
        selected=select_operating_point(rows,policy=policy);selected['analyse_sha256']=digest(ROOT/'study/analyse.py')
        selector=leaf/'selection.json';selector.write_text(json.dumps(selected))
        return {'controller_file':str(model),'selection_file':str(selector),'controller_type':kind,'initial_model':'small',
                'penalty':selected['selected_penalty'],'threshold':selected['selected_threshold']},[model,selector,leaf/('training_cv_'+kind+'.json')]

    def frozen_fixture(self,root,runtime=None):
        settings={p:dict(initial_model='small',controller_type='none',penalty=0,threshold=0) for p in BASELINES};files=[]
        for policy,kind in [('table','table'),('learned','logistic_v2')]:
            setting,paths=self.selection_fixture(root,kind);settings[policy]=setting;files+=paths
        settings['learned_model_only']=dict(settings['learned'])
        router=root/'router.json';router.write_text('{"models":[]}');files.append(router)
        runtime=RUNTIME if runtime is None else runtime;models={}
        if runtime.get('generation_cache'):
            from study.generation_cache import GenerationCache
            models={runtime['small_model']:'small-digest',runtime['large_model']:'large-digest'}
            cache=GenerationCache(runtime['cache_directory']);cache.bind({'models':models,'provider_version':{'version':'mock'}})
            snapshot=root/'GENERATION_CACHE_FREEZE.json'
            snapshot.write_text(json.dumps({p.name:digest(p) for p in cache.directory.glob('*.json')}))
            files.extend([snapshot,cache.directory/'CONTEXT.json'])
        spec={'validation_complete':True,'corpus_directory':str(ROOT/'results/v2/corpus_v2'),'runtime':runtime,
              'models':models,'provider_version':{'version':'mock'},'policy_settings':settings,'files':files,'router_file':str(router),
              'schedule':{'seeds':10,'trials':3,'faults':['clean','transient'],'wordings':1},
              'criterion':{'tolerance':.1,'lower_mean_effort':True,'both_fault_conditions':True}}
        return write_freeze(root,spec)

    def test_validation_proof_freeze_hashes_and_one_attempt_ledger(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();spec=self.frozen_fixture(root)
            self.assertEqual(validate_freeze(root),spec);self.assertEqual(begin_test(root),spec);self.assertEqual(begin_test(root),spec)
            with self.assertRaises(ValueError): write_freeze(root,spec)
            with self.assertRaisesRegex(ValueError,'one-attempt ledger'): validate_leaf(root,SimpleNamespace())
            key=('task','clean',0,'small',0,0,'none');claim_job(root,'test',key)
            with self.assertRaises(ValueError): claim_job(root,'test',key)
            row={'success':False};verify_completed_result(root,'test',key,row)
            verify_completed_result(root,'test',key,row)
            with self.assertRaises(ValueError): verify_completed_result(root,'test',key,{'success':True})
            with self.assertRaises(ValueError): complete_job(root,'test',('unclaimed',),{})
            with self.assertRaises(ValueError): claim_job(root,'train',key)
            model=Path(spec['policy_settings']['learned']['controller_file']);model.write_text('{}')
            with self.assertRaises(ValueError): validate_freeze(root)
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();setting,_=self.selection_fixture(root,'table')
            validate_selection(setting,ROOT/'results/v2/corpus_v2')
            setting['threshold']=.1
            with self.assertRaises(ValueError): validate_selection(setting,ROOT/'results/v2/corpus_v2')

    def test_oracle_is_separate_and_decomposition_denominators_are_explicit(self):
        task={'expected':{'x':1}}
        base=dict(success=False,effort_seconds=2,calls=1,operations=1,wall_seconds=3,policy='small',trace=[],expected=task['expected'])
        good=dict(base,success=True,policy='large',effort_seconds=4)
        with patch('study.workflow.run_policy',side_effect=[base,good]): result=oracle_outcome(task,0,'clean',{})
        self.assertTrue(result['success']);self.assertEqual(result['effort_seconds'],6);self.assertEqual(result['calls'],2)
        self.assertEqual(result['policy'],'oracle_upper_bound')
        events=[dict(kind='verification',report={'x':0},error=None,flagged=True),dict(kind='verification',report={'x':1},error=None,flagged=False)]
        report=decomposition([dict(good,trace=events)])['large']
        self.assertEqual(report['true_positive'],1);self.assertEqual(report['true_negative'],1)
        self.assertEqual(report['correct_after_flag'],1);self.assertEqual(report['runs'],1)
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp,patch('study.workflow.validate_freeze',return_value={'policy_settings':{'small':{}}}):
            with self.assertRaises(ValueError): analyse_results(tmp)

    def test_development_refuses_failed_pair_and_cli_initialises_without_inference(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp).resolve();plan,tasks=initialise(root,ROOT/'results/v2/corpus_v2')
            with patch('study.workflow.profile',return_value={'complete':True,'suitable_pair':False}),patch('study.workflow.run_module') as runner:
                with self.assertRaises(ValueError): develop(root,plan,tasks)
            runner.assert_not_called()
            with patch('sys.argv',['workflow','init','--directory',str(root)]),patch('study.workflow.api') as provider,contextlib.redirect_stdout(io.StringIO()): main()
            provider.assert_not_called()
