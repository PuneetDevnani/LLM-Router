"""Offline detector audit; reference answers never enter live verification."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from study.data import corpus
from study.runtime import Session
from pilot.environment import correct
from study.experiment import read_rows


def main():
    tasks={t['id']:t for t in corpus()}; counts={'true_positive':0,'false_positive':0,'true_negative':0,'false_negative':0}; details=[]
    for record in read_rows('results/main/training_states.jsonl'):
        if record.get('stage','initial')!='initial' or record['fault']!='clean' or record['state']['error']: continue
        s=Session(tasks[record['task_id']]); s.state=deepcopy(record['state'])
        incorrect=not correct(s.state['report'],s.task['expected']); s.check(); flagged=s.state['checks']['flagged']
        key=('true_positive' if incorrect else 'false_positive') if flagged else ('false_negative' if incorrect else 'true_negative')
        counts[key]+=1
        details.append({'task_id':record['task_id'],'incorrect':incorrect,'flagged':flagged,'checks':s.state['checks']})
    tp,fp,fn=counts['true_positive'],counts['false_positive'],counts['false_negative']
    result={'scope':'clean initial training states with executable, structurally formattable outputs; development diagnostic only',
            'counts':counts,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'details':details}
    Path('results/main/detector_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))


if __name__=='__main__': main()
