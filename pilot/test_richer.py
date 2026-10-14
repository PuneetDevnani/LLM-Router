import argparse
import json
import unittest
from unittest.mock import patch
from environment import query, correct
from richer import database, tasks
from checks import equivalent, verify
from run import run_task

# SQL reference implementations are for evaluator verification only, never sent to the agent.
SQL = {
    'net_total': "SELECT SUM(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'",
    'january': "SELECT SUM(amount) FROM orders WHERE status='completed' AND order_date>='2026-01-01' AND order_date<'2026-02-01'",
    'date_window': "SELECT COUNT(*) FROM orders WHERE status='completed' AND order_date>='2026-01-10' AND order_date<'2026-01-21'",
    'region_total': "SELECT c.region,SUM(o.amount) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region",
    'region_net': "SELECT c.region,SUM(o.amount-COALESCE(o.discount,0)) FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' GROUP BY c.region",
    'distinct_customers': "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE status='completed'",
    'average_net': "SELECT AVG(amount-COALESCE(discount,0)) FROM orders WHERE status='completed'",
    'missing_count': "SELECT COUNT(*) FROM orders WHERE status='completed' AND discount IS NULL",
    'completion_rate': "SELECT 100.0*SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)/COUNT(*) FROM orders",
    'discount_rate': "SELECT 100.0*SUM(CASE WHEN COALESCE(discount,0)>0 THEN 1 ELSE 0 END)/COUNT(*) FROM orders WHERE status='completed'",
    'large_net': "SELECT COUNT(*) FROM orders WHERE status='completed' AND amount-COALESCE(discount,0)>=5000",
    'summary': "SELECT COUNT(*),SUM(amount) FROM orders WHERE status='completed' AND order_date>='2026-01-01' AND order_date<'2026-02-01'",
}


class RichTests(unittest.TestCase):
    def test_all_independent_reference_answers_and_eligible_invariances(self):
        self.assertEqual(len(tasks()),72)
        for task in tasks():
            con,_ = database(task['seed'])
            sql = SQL[task['template']]
            output = query(con,sql)
            if task['template'] in ('region_total','region_net'):
                report = {'regions':dict(output['rows'])}
            elif task['template'] == 'summary':
                report = dict(zip(['count','total'],output['rows'][0]))
            else:
                report = {next(iter(task['expected'])):output['rows'][0][0]}
            self.assertTrue(correct(report,task['expected']),task['id'])
            evidence = verify(sql,output,task['seed'],task['contract'],database)
            self.assertFalse(evidence['flagged'],task['id'])
            self.assertFalse(any(e['status']=='inconclusive' for e in evidence['evidence']))
            con.close()

    def test_equivalent_sql_not_keyword_gated(self):
        task=tasks()[0]
        con,_=database(0)
        sql="SELECT SUM(CASE WHEN status != 'cancelled' THEN amount-IFNULL(discount,0) ELSE 0 END) FROM orders"
        output=query(con,sql)
        self.assertTrue(correct({'net_total':output['rows'][0][0]},task['expected']))
        self.assertFalse(verify(sql,output,0,task['contract'],database)['flagged'])
        con.close()

    def test_omitted_filter_and_wrong_null_semantics_detected(self):
        con,_=database(0)
        for sql in ['SELECT SUM(amount) FROM orders', "SELECT AVG(amount-discount) FROM orders WHERE status='completed'"]:
            evidence=verify(sql,query(con,sql),0,{'excludes_cancelled':True,'null_discount_as_zero':True},database)
            self.assertTrue(evidence['flagged'])
        con.close()

    def test_consistently_wrong_aggregation_can_pass(self):
        con,_=database(0)
        sql="SELECT MAX(amount) FROM orders WHERE status='completed'"
        output=query(con,sql)
        self.assertFalse(verify(sql,output,0,{'excludes_cancelled':True},database)['flagged'])
        self.assertFalse(correct({'january':output['rows'][0][0]},tasks()[1]['expected']))
        con.close()

    def test_order_ignored_but_duplicates_and_shape_preserved(self):
        a={'columns':['x'],'rows':[[1],[2],[2]]}
        self.assertTrue(equivalent(a,{'columns':['alias'],'rows':[[2.0],[1.0],[2.0]]}))
        self.assertFalse(equivalent(a,{'columns':['x'],'rows':[[1],[2]]}))
        self.assertFalse(equivalent(a,{'columns':['x'],'rows':[[1],[2,2],[2]]}))

    def test_semantic_escalation_and_gold_not_in_feedback(self):
        task=tasks()[0]
        args=argparse.Namespace(model='small',strong_model='large',policy='fixed',checks='metamorphic',
                                semantic_response='escalate',fault='clean',max_steps=4,endpoint='unused',trial=0)
        responses=[{'message':{'content':json.dumps({'action':'query','sql':'SELECT SUM(amount) FROM orders'})}},
                   {'message':{'content':json.dumps({'action':'query','sql':SQL['net_total']})}},
                   {'message':{'content':json.dumps({'action':'finish','report':task['expected']})}}]
        with patch('run.request',side_effect=responses) as backend:
            result=run_task(task,args)
        self.assertTrue(result['success'])
        self.assertTrue(result['trace'][0]['verification']['flagged'])
        self.assertEqual([c.args[2]['model'] for c in backend.call_args_list],['small','large','large'])
        self.assertNotIn('expected',result['public_prompt'])


if __name__=='__main__':
    unittest.main()
