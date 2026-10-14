"""Paired offline economics, including failed effort and oracle limitations."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from study.economics import aggregate,profile_economics,main


class EconomicsTests(unittest.TestCase):
    def rows(self):
        return [dict(task_id=str(i),template='a' if i<2 else 'b',split='train',model=m,
                     success=success,effort_seconds=cost,trial=0,fault='clean')
                for i,flags in enumerate(((True,False),(False,True),(False,False)))
                for m,success,cost in zip(('small','large'),flags,(1,4))]

    def test_policy_pairing_cost_success_and_template_counts(self):
        result=profile_economics(self.rows(),'small','large');p=result['policies']
        self.assertEqual(p['always_small']['successes'],1);self.assertEqual(p['always_large']['successes'],1)
        self.assertEqual(p['oracle_pre_routing']['total_effort_seconds'],6)
        self.assertEqual(p['oracle_pre_routing']['effort_per_success_seconds'],3)
        self.assertEqual(p['small_first_perfect_detector']['total_effort_seconds'],11)
        self.assertEqual(p['small_first_perfect_detector']['successes'],2)
        self.assertEqual(p['small_first_perfect_detector']['mean_model_calls'],5/3)
        self.assertEqual(result['per_template']['a']['models']['small']['runs'],2)
        self.assertIsNone(result['per_template']['b']['policies']['always_small']['effort_per_success_seconds'])
        with self.assertRaises(ValueError): aggregate([])

    def test_reject_incomplete_duplicate_holdout_and_invalid_cost(self):
        rows=self.rows()
        invalid=[rows[:-1],rows+[rows[0]],[]]
        for field,value in [('split','test'),('model','other'),('effort_seconds',float('nan')),
                            ('effort_seconds',-1),('effort_seconds',True),('success',2),('template','wrong')]:
            changed=deepcopy(rows);changed[0][field]=value;invalid.append(changed)
        for values in invalid:
            with self.assertRaises(ValueError): profile_economics(values,'small','large')
        with self.assertRaises(ValueError): profile_economics(rows,'small','small')

    def test_cli_no_inference_preserves_input_and_completed_output(self):
        with tempfile.TemporaryDirectory(dir='results/v2') as tmp:
            source=Path(tmp)/'profile.jsonl';source.write_text('\n'.join(map(json.dumps,self.rows()))+'\n')
            before=source.read_bytes();output=Path(tmp)/'economics.json'
            args=['economics',str(source),'--small-model','small','--large-model','large','--output',str(output)]
            with patch('study.runtime.api',side_effect=AssertionError('No model calls')),patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()): main()
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(json.loads(output.read_text())['policies']['oracle_pre_routing']['successes'],2)
            with patch('sys.argv',args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main()
            args[-1]='results/main/economics.json'
            with patch('sys.argv',args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main()
