"""Immutable scientific configuration and one-attempt held-out job ledger."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'results/v2'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def source_files():
    names=('runtime.py','generation_cache.py','controller.py','data.py','data_v2.py','verifier.py','experiment.py',
           'analyse.py','task_routing.py','freeze_protocol.py','workflow.py')
    return ([ROOT/'study'/n for n in names]+sorted(p for p in (ROOT/'pilot').glob('*.py') if not p.name.startswith('test'))+
            [ROOT/n for n in ('V2_PROTOCOL.md','TASK_ROUTING_PROTOCOL.md','COMPLETION_PROTOCOL.md')])


def validate_selection(setting,corpus_directory):
    """Freeze only a complete nine-point, 40-task validation selection."""
    from study.headroom import frozen_development
    from study.analyse import select_operating_point
    model=confined(setting['controller_file']);leaf=model.parent
    selector=confined(setting['selection_file']);record=json.loads(selector.read_text())
    kind=setting['controller_type'];policy='table' if kind=='table' else 'learned'
    rows=[json.loads(line) for line in (leaf/'validation.jsonl').read_text().splitlines()]
    tasks,_=frozen_development(corpus_directory)
    expected={t['id'] for t in tasks if t['split']=='validation'}
    candidates=[r for r in rows if r['policy']==policy and r.get('controller_type')==kind]
    baseline=[r for r in rows if r['policy']=='large']
    if any(r['trial']!=0 or r['fault']!='clean' for r in candidates+baseline):
        raise ValueError('Use all 40 validation tasks, one clean trial')
    keys=[(r['task_id'],r['penalty'],r.get('threshold',0.0)) for r in candidates]
    required={(task,p,t) for task in expected for p in (0,.15,.3) for t in (0,.1,.2)}
    if len(keys)!=len(required) or set(keys)!=required:
        raise ValueError('Complete unique 40-task validation grid required at every point')
    if len(baseline)!=len(expected) or {r['task_id'] for r in baseline}!=expected:
        raise ValueError('Complete unique large-model validation baseline required')
    selected=select_operating_point(candidates+baseline,policy=policy)
    if {(c['penalty'],c['threshold']) for c in selected['candidates']}!={(p,t) for p in (0,.15,.3) for t in (0,.1,.2)}:
        raise ValueError('Validation grid differs from prospective protocol')
    if record.get('analyse_sha256')!=digest(ROOT/'study/analyse.py'): raise ValueError('Selection analysis source differs')
    for name in ('selected_penalty','selected_threshold','selection_mode','candidates'):
        if record.get(name)!=selected[name]: raise ValueError('Selection differs from recorded validation evidence')
    if setting['penalty']!=selected['selected_penalty'] or setting['threshold']!=selected['selected_threshold']:
        raise ValueError('Frozen setting is not the validation choice')
    fitted=json.loads(model.read_text());training_sha=digest(leaf/'branches.jsonl')
    cv=json.loads((leaf/('training_cv_'+kind+'.json')).read_text())
    if fitted.get('training_branches_sha256')!=training_sha or cv.get('training_branches_sha256')!=training_sha:
        raise ValueError('Fitted controller or CV differs from preserved training branches')


def confined(path):
    path=Path(path).resolve()
    if not path.is_relative_to(V2): raise ValueError('Scientific artifacts must be under results/v2')
    return path


def write_freeze(directory,spec):
    root=confined(directory)
    if (root/'FREEZE.json').exists() or (root/'FREEZE.md').exists(): raise ValueError('Existing final freeze is preserved')
    if not spec.get('validation_complete'): raise ValueError('Complete training and validation before final freeze')
    required={'corpus_directory','runtime','models','provider_version','policy_settings','files','schedule','criterion','router_file'}
    if required-set(spec): raise ValueError('Incomplete final freeze specification')
    policies={'small','large','escalate_default','escalate_fresh','rules','task_routing','table','learned','learned_model_only'}
    if set(spec['policy_settings'])!=policies: raise ValueError('Freeze the complete nine-policy comparison')
    for setting in spec['policy_settings'].values():
        if 'controller_file' in setting: validate_selection(setting,spec['corpus_directory'])
    spec=dict(spec,version=1,dispatcher='study.workflow',
              code_git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files()})
    corpus=confined(spec['corpus_directory']);config=json.loads((corpus/'configuration.json').read_text())
    spec.update(dataset_sha256=config['dataset_sha256'],test_templates_sha256=config['test_templates_sha256'],
                schema_shift_templates_sha256=config['schema_shift_templates_sha256'])
    spec['files']={str(confined(p).relative_to(ROOT)):digest(p) for p in spec['files']}
    root.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(spec,indent=2,allow_nan=False)+'\n';sha=hashlib.sha256(raw.encode()).hexdigest()
    (root/'FREEZE.json').write_text(raw)
    (root/'FREEZE.md').write_text('# Final controller freeze\n\nCode git hash: '+spec['code_git_hash']+
        '\n\nManifest SHA256: '+sha+'\n\nTest split SHA256: '+spec['test_templates_sha256']+
        '\n\nSchema shift SHA256: '+spec['schema_shift_templates_sha256']+
        '\n\nSelected policy settings:\n\n```json\n'+json.dumps(spec['policy_settings'],indent=2)+'\n```\n\n'+
        'Benefit criterion: '+canonical(spec['criterion'])+'\n\nThe workflow ledger permits one attempt per held-out trajectory. '+
        'Provider failures stop the study; attempted jobs cannot be silently retried. V1 remains primary; v2 is exploratory.\n')
    return validate_freeze(root)


def validate_freeze(directory):
    root=confined(directory);path=root/'FREEZE.json';spec=json.loads(path.read_text())
    if spec.get('version')!=1 or spec.get('dispatcher')!='study.workflow': raise ValueError('Unsupported final freeze')
    if digest(path) not in (root/'FREEZE.md').read_text(): raise ValueError('Final manifest hash mismatch')
    actual={str(p.relative_to(ROOT)):digest(p) for p in source_files()}
    if actual!=spec['source_hashes']: raise ValueError('Scientific source changed after final freeze')
    for name,sha in spec['files'].items():
        if digest(confined(ROOT/name))!=sha: raise ValueError('Frozen fitted artifact changed: '+name)
    if spec['runtime'].get('generation_cache'):
        snapshot_path=root/'GENERATION_CACHE_FREEZE.json'
        cache=confined(spec['runtime']['cache_directory'])
        if str(snapshot_path.relative_to(ROOT)) not in spec['files'] or str((cache/'CONTEXT.json').relative_to(ROOT)) not in spec['files']:
            raise ValueError('Generation cache snapshot/context omitted from freeze')
        for name,sha in json.loads(snapshot_path.read_text()).items():
            if Path(name).name!=name or digest(cache/name)!=sha: raise ValueError('Pre-freeze cache record changed')
    required_files={str(confined(s[k]).relative_to(ROOT)) for s in spec['policy_settings'].values()
                    for k in ('controller_file','selection_file') if k in s}
    required_files.add(str(confined(spec['router_file']).relative_to(ROOT)))
    if not required_files<=set(spec['files']): raise ValueError('Fitted artifacts omitted from final hash manifest')
    corpus=confined(spec['corpus_directory']);raw=(corpus/'corpus.json').read_bytes()
    config=json.loads((corpus/'configuration.json').read_text());tasks=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=spec['dataset_sha256']: raise ValueError('Frozen corpus changed')
    from study.experiment import validate_test_configuration
    for split in ('test','schema_shift'):
        validate_test_configuration(config,tasks,split)
        if config[split+'_templates_sha256']!=spec[split+'_templates_sha256']: raise ValueError('Frozen split changed')
    if spec['criterion']!={'tolerance':.1,'lower_mean_effort':True,'both_fault_conditions':True}:
        raise ValueError('Pre-registered criterion changed')
    if spec['schedule']!={'seeds':10,'trials':3,'faults':['clean','transient'],'wordings':1}:
        raise ValueError('Pre-registered held-out schedule changed')
    marker=root/'TEST_STARTED.json'
    if marker.exists() and json.loads(marker.read_text())['manifest_sha256']!=digest(path):
        raise ValueError('Manifest differs from the study that already started')
    return spec


def validate_leaf(directory,args):
    validate_freeze(directory)
    raise ValueError('Held-out runs must use study.workflow test so the one-attempt ledger cannot be bypassed')


def begin_test(directory):
    root=confined(directory);spec=validate_freeze(root);marker=root/'TEST_STARTED.json'
    record={'manifest_sha256':digest(root/'FREEZE.json'),'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with marker.open('x') as stream: stream.write(json.dumps(record,indent=2)+'\n')
    except FileExistsError:
        if json.loads(marker.read_text())['manifest_sha256']!=record['manifest_sha256']: raise ValueError('Different test already started')
    return spec


def claim_job(directory,split,key):
    if split not in ('test','schema_shift'): raise ValueError('Held-out split required')
    root=confined(directory);ledger=root/'ledger'/split;ledger.mkdir(parents=True,exist_ok=True)
    token=hashlib.sha256(canonical(key).encode()).hexdigest();path=ledger/(token+'.started.json')
    record={'key':list(key),'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with path.open('x') as stream: stream.write(json.dumps(record)+'\n')
    except FileExistsError: raise ValueError('This held-out job was already attempted; do not rerun it: '+canonical(key))
    return token


def complete_job(directory,split,key,result):
    if split not in ('test','schema_shift'): raise ValueError('Held-out split required')
    root=confined(directory);token=hashlib.sha256(canonical(key).encode()).hexdigest();ledger=root/'ledger'/split
    if not (ledger/(token+'.started.json')).exists(): raise ValueError('Unclaimed held-out result')
    with (ledger/(token+'.completed.json')).open('x') as stream:
        stream.write(json.dumps({'key':list(key),'result_sha256':hashlib.sha256(canonical(result).encode()).hexdigest()})+'\n')


def verify_completed_result(directory,split,key,result):
    """Recover a written outcome's checkpoint without repeating its trajectory."""
    root=confined(directory);token=hashlib.sha256(canonical(key).encode()).hexdigest();ledger=root/'ledger'/split
    if not (ledger/(token+'.started.json')).exists(): raise ValueError('Result has no attempt ledger entry')
    path=ledger/(token+'.completed.json')
    if not path.exists(): complete_job(root,split,key,result)
    record=json.loads(path.read_text())
    if record['key']!=list(key) or record['result_sha256']!=hashlib.sha256(canonical(result).encode()).hexdigest():
        raise ValueError('Recorded held-out result was altered')
