"""Infrastructure checks only: these do not measure model quality."""
import argparse
import json
import unittest
from unittest.mock import patch
from environment import database, tasks, query, correct
from run import run_task


class PilotTests(unittest.TestCase):
    def test_expected_answers_independent_of_sql(self):
        for task in tasks():
            con, _ = database(task['seed'])
            if task['template'] in ('total', 'count'):
                fn = 'SUM(revenue)' if task['template'] == 'total' else 'COUNT(*)'
                value = query(con, f"SELECT {fn} FROM orders WHERE status='completed'")['rows'][0][0]
                actual = {task['template']: value}
            else:
                column = 'month' if task['template'] == 'monthly' else 'category'
                rows = query(con, f"SELECT {column}, SUM(revenue) FROM orders WHERE status='completed' GROUP BY {column}")['rows']
                actual = {task['template']: dict(rows)}
            self.assertTrue(correct(actual, task['expected']), task['id'])
            con.close()

    def test_writes_and_attach_rejected(self):
        con, _ = database(0)
        for sql in ['DELETE FROM orders', 'DROP TABLE orders', "ATTACH DATABASE ':memory:' AS other", 'PRAGMA user_version=1', 'SELECT randomblob(1000000000)']:
            with self.assertRaises(Exception):
                query(con, sql)
        self.assertEqual(query(con, 'SELECT COUNT(*) FROM orders')['rows'][0][0], 60)
        con.close()

    def test_wrong_or_nonfinite_reports_rejected(self):
        expected = {'total': 5}
        for report in [None, {'total': True}, {'total': float('nan')}, {'total': 6}, {'total': 5, 'extra': 1}]:
            self.assertFalse(correct(report, expected))

    def test_transient_retry_needs_no_extra_generation(self):
        task = tasks()[0]
        args = argparse.Namespace(model='fake-small', strong_model='fake-large', policy='rules',
                                  fault='transient', max_steps=6, endpoint='http://unused', trial=0)
        responses = [{'message': {'content': '{"action":"query","sql":"SELECT SUM(revenue) FROM orders WHERE status=\'completed\'"}'}},
                     {'message': {'content': '{"action":"finish","report":' + __import__('json').dumps(task['expected']) + '}'}}]
        with patch('run.request', side_effect=responses) as backend:
            result = run_task(task, args)
        self.assertTrue(result['success'])
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(result['trace'][0]['error'], 'tool_timeout')
        self.assertEqual(result['trace'][1]['recovery'], 'tool_retry')

    def test_persistent_outage_cannot_be_scored_as_success(self):
        task = tasks()[0]
        args = argparse.Namespace(model='fake', strong_model='fake', policy='fixed',
                                  fault='outage', max_steps=2, endpoint='http://unused', trial=0)
        responses = [{'message': {'content': '{"action":"query","sql":"SELECT 1"}'}},
                     {'message': {'content': '{"action":"finish","report":' + __import__('json').dumps(task['expected']) + '}'}}]
        with patch('run.request', side_effect=responses):
            self.assertFalse(run_task(task, args)['success'])

    def test_invalid_json_escalates_without_ground_truth_feedback(self):
        task = tasks()[0]
        args = argparse.Namespace(model='small', strong_model='large', policy='escalate',
                                  fault='clean', max_steps=3, endpoint='http://unused', trial=0)
        responses = [{'message': {'content': 'not json'}},
                     {'message': {'content': json.dumps({'action': 'query', 'sql': "SELECT SUM(revenue) FROM orders WHERE status='completed'"})}},
                     {'message': {'content': json.dumps({'action': 'finish', 'report': task['expected']})}}]
        with patch('run.request', side_effect=responses) as backend:
            result = run_task(task, args)
        self.assertTrue(result['success'])
        self.assertEqual([c.args[2]['model'] for c in backend.call_args_list], ['small', 'large', 'large'])
        feedback = backend.call_args_list[-1].args[2]['messages']
        self.assertFalse(any('expected' in m['content'] for m in feedback))


if __name__ == '__main__':
    unittest.main()
