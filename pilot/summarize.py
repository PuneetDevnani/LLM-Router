"""Summarize saved actual-model runs without changing their traces."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'No completed task records in {path}')
    outcomes = collections.defaultdict(list)
    errors = collections.Counter()
    inference_seconds, loading_seconds, input_tokens, output_tokens = 0, 0, 0, 0
    verification_seconds, probes, flagged_queries = 0, 0, 0
    for row in rows:
        outcomes[row['template']].append(row['success'])
        for event in row['trace']:
            verification = event.get('verification', {})
            verification_seconds += verification.get('seconds',0)
            probes += len(verification.get('evidence',[]))
            flagged_queries += int(verification.get('flagged',False))
            if 'error' in event:
                errors[event['error']] += 1
            usage = event.get('usage', {})
            inference_seconds += sum(usage.get(k) or 0 for k in ['prompt_eval_duration', 'eval_duration']) / 1e9
            loading_seconds += (usage.get('load_duration') or 0) / 1e9
            input_tokens += usage.get('prompt_eval_count') or 0
            output_tokens += usage.get('eval_count') or 0
    return {'file': str(path), 'tasks': len(rows), 'successes': sum(r['success'] for r in rows),
            'median_task_seconds': statistics.median(r['seconds'] for r in rows),
            'total_task_seconds': sum(r['seconds'] for r in rows),
            'inference_seconds': inference_seconds, 'loading_seconds': loading_seconds,
            'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'verification_seconds':verification_seconds, 'verification_probes':probes,
            'flagged_queries':flagged_queries,
            'errors': dict(errors), 'success_by_template': {k: f'{sum(v)}/{len(v)}' for k, v in outcomes.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(p) for p in args.files], indent=2))
