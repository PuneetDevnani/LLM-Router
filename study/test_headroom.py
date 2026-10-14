"""Headroom and timing tests use mocked providers only."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study.headroom import frozen_development, timing_tasks, headroom_summary, timing_estimate, main
from study.data_v2 import corpus, REFERENCE_SQL
from study.experiment import main as experiment, run_policy
from study.runtime import Session, SMALL, LARGE
from study.controller import features


class HeadroomTests(unittest.TestCase):
    def freeze(self,root):
        with patch('sys.argv',['experiment','freeze','--corpus','v2','--output',str(root)]), \
             patch('study.experiment.V2_ROOT',root),contextlib.redirect_stdout(io.StringIO()): experiment()

    def test_only_development_selected_and_hash_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.freeze(root)
            tasks,config=frozen_development(root)
            self.assertEqual(len(tasks),120)
            self.assertEqual({t['split'] for t in tasks},{'train','validation'})
            chosen=timing_tasks(tasks);self.assertEqual(len(chosen),10)
            self.assertEqual(len({t['template'] for t in chosen}),10)
            self.assertEqual(sum(t['split']=='validation' for t in chosen),4)
            with self.assertRaises(ValueError): frozen_development(root,0)
            with self.assertRaises(ValueError): timing_tasks(tasks[:1])
            raw=(root/'corpus.json').read_bytes();(root/'corpus.json').write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'bytes changed'): frozen_development(root)

    def test_pair_gate_is_template_based_and_partial_profiles_do_not_pass(self):
        tasks=[dict(id=f't{i}',template=f't{i}',split='train') for i in range(10)]
        rows=[dict(task_id=t['id'],template=t['template'],trial=0,model=m,success=i<(5 if m=='small' else 8))
              for i,t in enumerate(tasks) for m in ('small','large')]
        report=headroom_summary(rows,tasks,'small','large')
        self.assertTrue(report['complete']);self.assertTrue(report['suitable_pair'])
        self.assertEqual(report['models']['small']['template_fraction'],.5)
        self.assertIsNone(headroom_summary(rows[:-1],tasks,'small','large')['suitable_pair'])
        with self.assertRaises(ValueError): headroom_summary(rows+[rows[0]],tasks,'small','large')
        estimate=timing_estimate([{'wall_seconds':2},{'wall_seconds':4}])
        self.assertEqual(estimate['mean_generation_trajectory_seconds'],3)
        self.assertEqual(estimate['headroom_profile_calls'],240)
        self.assertEqual(estimate['planned_counts']['primary_test_trajectories'],4320)
        with self.assertRaises(ValueError): timing_estimate([])

    def test_custom_model_roles_drive_actions_features_and_policy_generation(self):
        task=corpus()[0];s=Session(task,small_model='candidate-small',large_model='candidate-large',fresh_escalation=True)
        with patch.object(s,'generate') as generate: s.apply('escalate')
        self.assertEqual(generate.call_args.args[0],'candidate-large')
        s.state['model']='candidate-large';self.assertEqual(features(s)[16],1)
        self.assertNotIn('escalate_fresh',s.eligible())
        response={'message':{'content':json.dumps({'sql':REFERENCE_SQL[task['template']]})}}
        with patch('study.runtime.api',return_value=response) as api:
            result=run_policy(task,'large',small_model='candidate-small',large_model='candidate-large')
        self.assertEqual(api.call_args.args[1]['model'],'candidate-large');self.assertTrue(result['success'])

    def test_cli_timing_profile_reuse_and_resumability_never_reexecute_jobs(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            root=Path(tmp);freeze=root/'frozen';freeze.mkdir();self.freeze(freeze)
            output=root/'headroom'
            tasks,_=frozen_development(freeze,1);selected=timing_tasks(tasks)
            tags={'models':[{'name':SMALL,'digest':'small'},{'name':LARGE,'digest':'large'}]}
            args=['headroom','timing','--corpus-freeze',str(freeze),'--directory',str(output),'--seeds','1']
            sqls=[REFERENCE_SQL[t['template']] for t in selected for _ in range(2)]
            responses=[{'message':{'content':json.dumps({'sql':sql})}} for sql in sqls]
            with patch('sys.argv',args),patch('study.headroom.api',return_value=tags), \
                 patch('study.runtime.api',side_effect=responses) as provider,contextlib.redirect_stdout(io.StringIO()): main()
            self.assertEqual(provider.call_count,20)
            initial=(output/'timing.jsonl').read_bytes()
            with patch('sys.argv',args),patch('study.headroom.api',return_value=tags), \
                 patch('study.runtime.api') as provider,contextlib.redirect_stdout(io.StringIO()): main()
            provider.assert_not_called();self.assertEqual((output/'timing.jsonl').read_bytes(),initial)
            remaining=[t for t in tasks if t['id'] not in {t['id'] for t in selected}]
            responses=[{'message':{'content':json.dumps({'sql':REFERENCE_SQL[t['template']]})}} for t in remaining for _ in range(2)]
            args[1]='profile'
            with patch('sys.argv',args),patch('study.headroom.api',return_value=tags), \
                 patch('study.runtime.api',side_effect=responses) as provider,contextlib.redirect_stdout(io.StringIO()): main()
            self.assertEqual(provider.call_count,4)
            rows=[json.loads(line) for line in (output/'profile.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),24);self.assertEqual(sum(r.get('reused_from') is not None for r in rows),20)
            self.assertTrue(all(r['split'] in ('train','validation') for r in rows))
            summary=json.loads((output/'profile_summary.json').read_text());self.assertTrue(summary['complete'])
            # Perfect small-model headroom is also unsuitable under the fixed rule.
            self.assertFalse(summary['suitable_pair'])


if __name__=='__main__': unittest.main()
