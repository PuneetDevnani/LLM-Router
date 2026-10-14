"""Local Ollama baseline runner. No external API or third-party Python package."""
import argparse
import json
import platform
import statistics
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from environment import database, tasks, query, correct
from checks import verify
import richer

SYSTEM = '''You are a data-analysis agent. Database schema: orders(id INTEGER, month TEXT, category TEXT, status TEXT, revenue INTEGER). Revenue is an integer in rupees. status values are exactly 'completed' and 'cancelled'; comparisons are case-sensitive. month values use YYYY-MM format. Categories are books, electronics, and clothing. You can issue read-only SQL. Apply all filters requested by the task and use aggregate functions when totals are requested.
Return exactly one JSON action per turn:
{"action":"query","sql":"SELECT ..."}
or {"action":"finish","report":{...}}.
Use the query tool before finishing. Observe tool results and recover from errors. Your final report must match the requested structure. Do not explain outside JSON. /no_think'''


def request(endpoint, path, payload=None):
    req = urllib.request.Request(endpoint + path, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def run_task(task, args):
    factory = richer.database if 'schema' in task else database
    con, _ = factory(task['seed'])
    system = SYSTEM if 'schema' not in task else 'You are a data-analysis agent. Database schema: ' + task['schema'] + '\n' + SYSTEM.split('\n',1)[1]
    public_prompt = task['prompt']
    if 'contract' in task:
        public_prompt += '\nPublic task contract (provided to all baselines): ' + json.dumps(task['contract'])
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': public_prompt}]
    trace, report, queried = [], None, False
    model = args.model
    pending_retry = None
    started = time.perf_counter()
    try:
        for turn in range(args.max_steps):
            # Transient tool retry re-executes the failed operation without another model call.
            if pending_retry is not None:
                event = {'turn': turn, 'model': None, 'recovery': 'tool_retry', 'action': pending_retry}
                action, pending_retry = pending_retry, None
            else:
                before = time.perf_counter()
                response = request(args.endpoint, '/api/chat', {
                    'model': model, 'messages': messages, 'stream': False, 'format': 'json',
                    'think': False, 'keep_alive': 0,
                    'options': {'num_ctx': 2048, 'num_predict': 384, 'temperature': 0.2, 'seed': args.trial}})
                content = response['message']['content']
                event = {'turn': turn, 'model': model, 'wall_seconds': time.perf_counter() - before,
                         'content': content, 'usage': {k: response.get(k) for k in (
                             'total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration',
                             'prompt_eval_count', 'eval_count')}}
                messages.append({'role': 'assistant', 'content': content})
                try:
                    action = json.loads(content)
                    if not isinstance(action, dict):
                        raise ValueError('action must be a JSON object')
                except (json.JSONDecodeError, ValueError) as exc:
                    event['error'] = 'invalid_json'
                    event['detail'] = str(exc)
                    trace.append(event)
                    messages.append({'role': 'user', 'content': 'Invalid JSON action. Return a query or finish object.'})
                    if args.policy == 'escalate':
                        model = args.strong_model
                    continue
                event['action'] = action

            try:
                if action.get('action') == 'finish':
                    if not queried:
                        raise ValueError('query the database before finishing')
                    report = action.get('report')
                    trace.append(event)
                    break
                if action.get('action') != 'query':
                    raise ValueError('unknown action; use query or finish')
                # Faults apply to tool operations, never reveal their hidden schedule to the agent.
                query_index = sum(e.get('action', {}).get('action') == 'query' and e.get('recovery') != 'tool_retry' for e in trace)
                transient = args.fault == 'transient' and query_index == 0 and event.get('recovery') != 'tool_retry'
                if args.fault == 'outage' or transient:
                    raise TimeoutError('database temporarily unavailable')
                output = query(con, action.get('sql'))
                queried = True
                event['tool_result'] = output
                messages.append({'role': 'user', 'content': 'Tool result: ' + json.dumps(output)})
                if getattr(args, 'checks', 'off') == 'metamorphic' and 'contract' in task:
                    evidence = verify(action['sql'], output, task['seed'], task['contract'], factory)
                    event['verification'] = evidence
                    public_evidence = {'flagged':evidence['flagged'],
                                       'checks':[{'check':e['check'],'status':e['status']} for e in evidence['evidence']],
                                       'interpretation':evidence['interpretation']}
                    messages.append({'role':'user','content':'Verification evidence: ' + json.dumps(public_evidence)})
                    if evidence['flagged']:
                        response_policy = getattr(args, 'semantic_response', 'observe')
                        if response_policy in ('repair','escalate'):
                            messages.append({'role':'user','content':'A task-specific invariance check was violated. Inspect the query and repair it before submitting a report. The check may be imperfect.'})
                        if response_policy == 'escalate':
                            model = args.strong_model
            except (ValueError, sqlite3.Error, TimeoutError) as exc:
                event['error'] = 'tool_timeout' if isinstance(exc, TimeoutError) else 'invalid_action'
                event['detail'] = str(exc)
                messages.append({'role': 'user', 'content': 'Tool error: ' + str(exc)})
                if args.policy == 'rules' and isinstance(exc, TimeoutError) and event.get('recovery') != 'tool_retry':
                    pending_retry = action
                elif args.policy in ('rules', 'escalate'):
                    model = args.strong_model
            trace.append(event)
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        trace.append({'error': 'backend_failure', 'detail': str(exc)})
    finally:
        con.close()
    return {'task_id': task['id'], 'template': task['template'], 'trial': args.trial,
            'policy': args.policy, 'fault': args.fault, 'model': args.model,
            'strong_model': args.strong_model, 'success': queried and correct(report, task['expected']),
            'report': report, 'expected': task['expected'], 'seconds': time.perf_counter() - started,
            'system_prompt':system, 'public_prompt':public_prompt, 'trace': trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3:0.6b')
    p.add_argument('--strong-model', default='qwen3:1.7b')
    p.add_argument('--policy', choices=['fixed', 'escalate', 'rules'], default='fixed')
    p.add_argument('--suite', choices=['pilot','rich'], default='pilot')
    p.add_argument('--checks', choices=['off','metamorphic'], default='off')
    p.add_argument('--semantic-response', choices=['observe','repair','escalate'], default='observe')
    p.add_argument('--fault', choices=['clean', 'transient', 'outage'], default='clean')
    p.add_argument('--endpoint', default='http://127.0.0.1:11434')
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=6)
    p.add_argument('--output', default='results/pilot.jsonl')
    args = p.parse_args()
    if args.limit < 1 or args.max_steps < 1:
        p.error('limit and max-steps must be positive')
    if args.suite != 'rich' and args.checks != 'off':
        p.error('metamorphic checks currently require --suite rich')
    if args.checks == 'off' and args.semantic_response != 'observe':
        p.error('semantic repair/escalation requires --checks metamorphic')
    try:
        inventory = request(args.endpoint, '/api/tags')
    except urllib.error.URLError as exc:
        p.exit(2, f'Ollama is unavailable at {args.endpoint}: {exc}\nStart ollama serve first.\n')
    names = {m['name'] for m in inventory['models']}
    required = {args.model} | ({args.strong_model} if args.policy != 'fixed' or args.semantic_response == 'escalate' else set())
    if required - names:
        p.exit(2, 'Missing local models: ' + ', '.join(sorted(required - names)) + '\n')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a previous experiment.
    outcomes = []
    with output.open('x') as f:
        suite = richer.tasks() if args.suite == 'rich' else tasks()
        for task in suite[:args.limit]:
            result = run_task(task, args)
            result['config'] = vars(args)
            result['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                                 'ollama': request(args.endpoint, '/api/version')}
            result['model_inventory'] = inventory
            f.write(json.dumps(result) + '\n')
            f.flush()
            outcomes.append(result)
            print(f"{task['id']}: success={result['success']} time={result['seconds']:.1f}s", flush=True)
    if outcomes:
        print(json.dumps({'tasks': len(outcomes), 'success_rate': statistics.mean(r['success'] for r in outcomes),
                          'median_seconds': statistics.median(r['seconds'] for r in outcomes)}, indent=2))


if __name__ == '__main__':
    main()
