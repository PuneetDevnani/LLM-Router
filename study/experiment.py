"""Resumable local experiments. Gold labels are used only for offline learning/scoring."""
import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from study.data import corpus, GROUPS
from study.runtime import Session, api, SMALL, LARGE, fixed_continuation, lexical_checks
from study.controller import Controller, TableController, CONTROLLER_TYPES, leave_one_template_out_cv, features,FEATURES,REVISION_FEATURES


V2_ROOT=Path('results/v2')
CONTROLLER_FILES={'logistic_v1':'controller.json','table':'table_controller.json','logistic_v2':'controller_logistic_v2.json'}
POLICIES={'small','large','rules','escalate_default','escalate_fresh','task_routing','table','learned','learned_model_only','learned_no_effort','learned_no_history'}

def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()] if Path(path).exists() else []


def study_corpus(version='v1'):
    if version=='v1': return corpus()
    if version=='v2':
        from study.data_v2 import corpus as v2_corpus
        return v2_corpus()
    raise ValueError('Unknown corpus version')


def study_groups(version='v1'):
    if version=='v1': return GROUPS
    if version=='v2':
        from study.data_v2 import GROUPS as v2_groups
        return v2_groups
    raise ValueError('Unknown corpus version')


def provenance():
    files=[Path('study')/name for name in ('runtime.py','generation_cache.py','controller.py','data.py','data_v2.py','verifier.py','experiment.py','task_routing.py')]
    files+=sorted(p for p in Path('pilot').glob('*.py') if not p.name.startswith('test'))
    return {'ollama':api('/api/version'),'models':api('/api/tags'),
            'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}


def job_key(row):
    """Legacy rows are the zero-threshold operating point."""
    return (row['task_id'],row['fault'],row['trial'],row['policy'],
            row['penalty'],row.get('threshold',0.0),
            row.get('controller_type','table' if row['policy']=='table' else 'logistic_v1')
            if row['policy'].startswith('learned') or row['policy']=='table' else 'none')


def evaluation_jobs(tasks,faults,trials,policies,penalties,thresholds,controller_type='logistic_v1'):
    return [(t,fault,trial,policy,float(penalty),float(threshold),
             'table' if policy=='table' else controller_type if policy.startswith('learned') else 'none')
            for t in tasks for fault in faults for trial in range(trials)
            for policy in policies
            for penalty in (penalties if policy.startswith('learned') or policy=='table' else [0.0])
            for threshold in (thresholds if policy.startswith('learned') or policy=='table' else [0.0])]


def template_list_hash(templates):
    """Canonical hash of an ordered frozen template list."""
    if (not isinstance(templates,list) or not templates or any(not isinstance(t,str) or not t for t in templates)
            or len(set(templates))!=len(templates)):
        raise ValueError('Nonempty, unique test-template list required')
    return hashlib.sha256(json.dumps(templates,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def validate_test_configuration(config,tasks,split='test'):
    """Authorize only the frozen corpus test families, excluding seen v1 families."""
    if split not in ('test','schema_shift'): raise ValueError('Unknown held-out split')
    templates=config.get(split+'_templates')
    if config.get(split+'_templates_sha256')!=template_list_hash(templates):
        raise ValueError('Frozen test-template hash mismatch')
    actual={t['template'] for t in tasks if t['split']==split}
    if actual!=set(templates): raise ValueError('Corpus test templates differ from the frozen list')
    development={t['template'] for t in tasks if t['split']!=split}
    if development & set(templates): raise ValueError('Frozen test templates overlap development templates')
    groups=config.get('groups',{})
    if split in groups and set(groups[split])!=set(templates):
        raise ValueError('Configuration groups disagree with frozen test templates')
    if set(templates) & set(groups.get('train',[])+groups.get('validation',[])):
        raise ValueError('Configuration test templates overlap train/validation')
    # The old split is immutable evidence; derive its burned families from it.
    seen=json.loads(Path('results/main/configuration.json').read_text())['groups']['test']
    if set(templates) & set(seen): raise ValueError('Seen v1 test templates cannot be v2 test templates')
    return templates


def configuration(args,dataset_sha256):
    """Record runtime switches explicitly, including their v1-compatible defaults."""
    version=getattr(args,'corpus','v1');groups=study_groups(version)
    test_templates=sorted(groups['test'])
    result={'corpus_version':version,'groups':groups,'test_templates':test_templates,'test_templates_sha256':template_list_hash(test_templates),
            'max_calls':3,'max_tool_operations':6,'context':4096,
            'formatter':'deterministic declared output schema','wording_variants':2,
            'verifier':args.verifier,'initial_model':args.initial_model,
            'small_model':getattr(args,'small_model',SMALL),'large_model':getattr(args,'large_model',LARGE),
            'fresh_escalation':args.fresh_escalation,'neutral_feedback':args.neutral_feedback,
            'diversify_seed':args.diversify_seed,
            'generation_cache':getattr(args,'generation_cache',False),
            'cache_directory':str(Path(getattr(args,'cache_directory','results/v2/generation_cache')).resolve()) if getattr(args,'generation_cache',False) else None,
            'consistency_verification':getattr(args,'consistency_verification',False),
            'consistency_k':getattr(args,'consistency_k',3),'initial_routing':getattr(args,'initial_routing',False),
            'max_iterations':getattr(args,'max_iterations',5),
            'evidence_status':'post-hoc exploratory; original template families already inspected',
            'default_decoding':{'temperature':0.2,'num_predict':384,
                                'seed_rule':'trial+calls (calls includes current generation)' if args.diversify_seed else 'trial'},
            'dataset_sha256':dataset_sha256}
    if 'schema_shift' in groups:
        result['schema_shift_templates']=sorted(groups['schema_shift'])
        result['schema_shift_templates_sha256']=template_list_hash(result['schema_shift_templates'])
        result['corpus_source_sha256']=hashlib.sha256(Path('study/data_v2.py').read_bytes()).hexdigest()
    return result


def run_policy(task,policy,trial=0,fault='clean',controller=None,penalty=0.1,threshold=0.0,
               verifier='original',controller_type='logistic_v1',fresh_escalation=False,neutral_feedback=False,
               diversify_seed=False,initial_model='small',small_model=SMALL,large_model=LARGE,router=None,
               consistency_verification=False,consistency_k=3,initial_routing=False,max_iterations=5,
               generation_cache=False,cache_directory='results/v2/generation_cache'):
    if policy not in POLICIES: raise ValueError('Unknown policy: '+policy)
    if initial_model not in ('small','large'): raise ValueError('Unknown initial model')
    s=Session(task,trial,fault,verifier=verifier,
              fresh_escalation=fresh_escalation or policy=='escalate_fresh',
              neutral_feedback=neutral_feedback and policy!='escalate_default',diversify_seed=diversify_seed,
              small_model=small_model,large_model=large_model,consistency_verification=consistency_verification,
              consistency_k=consistency_k,initial_routing=initial_routing,
              generation_cache=generation_cache,cache_directory=cache_directory)
    routed_large=policy=='large' or ((policy.startswith('learned') or policy=='table') and initial_model=='large')
    if policy=='task_routing':
        if router is None: raise ValueError('Fitted task router required')
        model,values=router.choose(s.public)
        if model not in (small_model,large_model): raise ValueError('Task-router model pair differs from runtime roles')
        routed_large=model==large_model
        s.trace.append({'kind':'routing','values':values,'selected_model':model})
    if initial_routing and (policy.startswith('learned') or policy=='table'):
        action,values=controller.choose(s,penalty,accept_threshold=threshold)
        s.trace.append({'kind':'prediction','values':values});s.apply(action);routed_large=action=='route_large'
    else: s.generate(large_model if routed_large else small_model)
    for _ in range(max_iterations):
        eligible=s.eligible()
        if policy.startswith('learned') or policy=='table':
            mode={'learned':'joint','table':'joint','learned_model_only':'model_only','learned_no_effort':'no_effort','learned_no_history':'no_history'}[policy]
            action,values=controller.choose(s,penalty,mode,accept_threshold=threshold)
            s.trace.append({'kind':'prediction','values':values})
        elif s.state['error']=='tool_timeout' and 'retry' in eligible:
            action='retry'
        elif s.state['error']:
            same_small=policy=='small' or (policy=='task_routing' and not routed_large)
            action=('repair' if same_small else 'escalate') if s.calls<s.max_calls else 'stop'
        elif policy=='rules' and not s.state['checks']:
            action='verify'
        elif policy=='rules' and s.state['checks'].get('flagged') and 'escalate' in eligible:
            action='escalate_fresh' if 'escalate_fresh' in eligible else 'escalate'
        elif policy=='rules' and 'verify_consistency' in eligible:
            action='verify_consistency'
        elif policy=='escalate_default' and s.state['model']==small_model and 'escalate' in eligible:
            action='escalate'
        elif policy=='escalate_fresh' and s.state['model']==small_model and 'escalate_fresh' in eligible:
            action='escalate_fresh'
        else: action='accept'
        s.apply(action)
        if action in ('accept','stop'): break
    else:
        s.state['report']=None
        s.trace.append({'kind':'termination','reason':'controller_iteration_budget'})
    result=s.result(); result.update(policy=policy,penalty=penalty,threshold=threshold,verifier=verifier,
                                    fresh_escalation=s.fresh_escalation,neutral_feedback=s.neutral_feedback,
                                    diversify_seed=s.diversify_seed,
                                    generation_cache=generation_cache,
                                    consistency_verification=consistency_verification,consistency_k=consistency_k,
                                    initial_routing=initial_routing,max_iterations=max_iterations,
                                    initial_model='large' if routed_large else 'small',
                                    small_model=small_model,large_model=large_model,corpus_version=task.get('corpus_version','v1'),
                                    evidence_status=('exploratory frozen v2 held-out trajectory' if task.get('corpus_version')=='v2' and task['split'] in ('test','schema_shift') else
                                                     'exploratory v2 development' if task.get('corpus_version')=='v2' else 'post-hoc exploratory; original task families already inspected'),
                                    controller_type='table' if policy=='table' else controller_type if policy.startswith('learned') else 'none')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['freeze','collect','train','evaluate'])
    p.add_argument('--output',default='results/v2')
    p.add_argument('--corpus',choices=['v1','v2'],default='v1')
    p.add_argument('--small-model',default=SMALL)
    p.add_argument('--large-model',default=LARGE)
    p.add_argument('--router',help='Fitted task router; defaults to task_router.json in the output directory')
    p.add_argument('--controller',choices=sorted(CONTROLLER_TYPES),default='logistic_v1')
    p.add_argument('--cross-validate',action='store_true')
    p.add_argument('--split',choices=['train','validation','test','schema_shift'],default='validation')
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--wordings',type=int,default=1)
    p.add_argument('--trials',type=int,default=1)
    p.add_argument('--policies',default='small,large,rules,learned,learned_model_only')
    p.add_argument('--faults',default='clean')
    p.add_argument('--penalties',default='0.1')
    p.add_argument('--thresholds',default='0.0')
    p.add_argument('--verifier',choices=['original','contract_v2','contract_v3'],default='original')
    p.add_argument('--fresh-escalation',action='store_true')
    p.add_argument('--neutral-feedback',action='store_true')
    p.add_argument('--diversify-seed',action='store_true')
    p.add_argument('--generation-cache',action='store_true',help='Replay exact requests and charge recorded model usage')
    p.add_argument('--cache-directory',default='results/v2/generation_cache')
    p.add_argument('--consistency-verification',action='store_true')
    p.add_argument('--consistency-k',type=int,default=3)
    p.add_argument('--initial-routing',action='store_true')
    p.add_argument('--max-iterations',type=int,default=5)
    p.add_argument('--initial-model',choices=['small','large'],default='small',help='Initial collection and learned/table policy model; use a separate directory for each route')
    p.add_argument('--stage',choices=['initial','checked','repaired','consistency','routing'],default='initial')
    args=p.parse_args(); out=Path(args.output)
    if not 2<=args.consistency_k<=5 or args.max_iterations<1: p.error('Invalid consistency or iteration budget')
    if args.stage=='consistency' and not args.consistency_verification: p.error('Consistency stage requires --consistency-verification')
    if args.stage=='routing' and not args.initial_routing: p.error('Routing stage requires --initial-routing')
    if set(args.policies.split(','))-POLICIES: p.error('Unknown policy')
    router_path=Path(args.router) if args.router else out/'task_router.json'
    if args.mode=='evaluate' and 'task_routing' in args.policies.split(',') and not router_path.exists():
        p.error('Fit a training-only task router before evaluating task_routing')
    try:
        penalties=[float(v) for v in args.penalties.split(',')]
        thresholds=[float(v) for v in args.thresholds.split(',')]
    except ValueError: p.error('Penalties and thresholds must be numeric lists')
    if any(not math.isfinite(v) or v<0 for v in penalties): p.error('Penalties must be finite and nonnegative')
    if any(not math.isfinite(v) or not 0<=v<=1 for v in thresholds): p.error('Thresholds must be finite probabilities')
    if len(set(penalties))!=len(penalties) or len(set(thresholds))!=len(thresholds): p.error('Grid values must be unique')
    if out.resolve()==Path('results/main').resolve():
        # A completed legacy stream can resume as a read-only no-op. No new
        # revised outcome or provenance may be written into the primary study.
        if (args.mode=='evaluate' and args.verifier=='original' and not args.fresh_escalation
                and not args.neutral_feedback and not args.diversify_seed and args.initial_model=='small' and args.corpus=='v1'
                and not args.consistency_verification and not args.initial_routing and args.max_iterations==5 and not args.generation_cache
                and args.small_model==SMALL and args.large_model==LARGE):
            tasks=[t for t in corpus() if t['split']==args.split and t['seed']<args.seeds
                   and int(t['id'][-1])<args.wordings]
            done={job_key(r) for r in read_rows(out/(args.split+'.jsonl'))}
            jobs=evaluation_jobs(tasks,args.faults.split(','),args.trials,args.policies.split(','),penalties,thresholds,args.controller)
            if jobs and all((t['id'],fault,trial,policy,penalty,threshold,controller_type) in done
                            for t,fault,trial,policy,penalty,threshold,controller_type in jobs):
                print('All legacy jobs already complete; frozen records and provenance unchanged.'); return
        p.error('results/main is the frozen primary study. Use a separate --output directory for revised work.')
    if not out.resolve().is_relative_to(V2_ROOT.resolve()):
        p.error('New outputs must be under results/v2.')
    if args.mode=='evaluate' and args.split in ('test','schema_shift'):
        config_path=out/'configuration.json'
        if not config_path.exists(): p.error('Freeze the hashed test-template list in configuration.json first.')
        try: validate_test_configuration(json.loads(config_path.read_text()),study_corpus(args.corpus),args.split)
        except (ValueError,KeyError) as exc: p.error(str(exc))
        if args.corpus=='v2' and (not (out/'FREEZE.md').exists() or not (out/'FREEZE.json').exists()):
            p.error('V2 held-out inference requires the final controller FREEZE.md and FREEZE.json; a dataset freeze is insufficient.')
        if args.corpus=='v2':
            from study.freeze_protocol import validate_leaf
            validate_leaf(out,args)
    out.mkdir(parents=True,exist_ok=True)
    if args.mode=='freeze':
        if (out/'configuration.json').exists() or (out/'corpus.json').exists():
            p.error('Existing frozen configuration is preserved; use a new output directory.')
        frozen_tasks=study_corpus(args.corpus)
        corpus_path=out/'corpus.json'; corpus_path.write_text(json.dumps(frozen_tasks,indent=2)+'\n')
        (out/'configuration.json').write_text(json.dumps(configuration(args,hashlib.sha256(corpus_path.read_bytes()).hexdigest()),indent=2)+'\n')
        print('Frozen',len(frozen_tasks),'instances in',len({t['template'] for t in frozen_tasks}),'grouped templates.'); return
    if args.mode=='train':
        if (out/'validation.jsonl').exists() or (out/'test.jsonl').exists():
            p.error('Training would replace a controller already used for evaluation. Use a new directory.')
        rows=read_rows(out/'branches.jsonl')
        if not rows: raise ValueError('Collect training branches first')
        if any(r.get('split')!='train' for r in rows): raise ValueError('Training branches only')
        if any(r.get('corpus_version','v1')!=args.corpus for r in rows): raise ValueError('Training corpus version mismatch')
        model_path=out/CONTROLLER_FILES[args.controller]
        if model_path.exists(): p.error('Existing fitted controller is preserved; use a new v2 directory.')
        controller=CONTROLLER_TYPES[args.controller].train(rows)
        controller.model['training_branches_sha256']=hashlib.sha256((out/'branches.jsonl').read_bytes()).hexdigest()
        controller.model['controller_type']=args.controller
        cv=leave_one_template_out_cv(rows) if args.cross_validate or args.controller=='logistic_v2' else None
        cv_path=out/('training_cv_'+args.controller+'.json')
        if cv is not None and cv_path.exists(): raise ValueError('Existing cross-validation output is preserved')
        controller.save(model_path)
        if cv is not None:
            cv_path.write_text(json.dumps({'training_branches_sha256':controller.model['training_branches_sha256'],
                                         'comparisons':cv},indent=2)+'\n')
        print('Trained',args.controller,'on',len(rows),'executed training branches.'); return
    config_path=out/'configuration.json'
    frozen_test_sha256=template_list_hash(sorted(study_groups(args.corpus)['test']))
    if config_path.exists():
        previous_config=json.loads(config_path.read_text())
        if previous_config.get('corpus_version','v1')!=args.corpus: raise ValueError('Frozen corpus version changed')
        defaults={'fresh_escalation':False,'neutral_feedback':False,'diversify_seed':False,'generation_cache':False,
                  'consistency_verification':False,'consistency_k':3,'initial_routing':False,'max_iterations':5,
                  'initial_model':'small','verifier':'original','small_model':SMALL,'large_model':LARGE}
        if any(previous_config.get(k,default)!=getattr(args,k) for k,default in defaults.items()):
            raise ValueError('Runtime options changed in frozen configuration. Use a new output directory.')
        if args.generation_cache and previous_config.get('cache_directory')!=str(Path(args.cache_directory).resolve()):
            raise ValueError('Generation cache changed; use a new output directory')
        if 'test_templates' in previous_config or 'test_templates_sha256' in previous_config:
            frozen_test_sha256=template_list_hash(previous_config.get('test_templates'))
            if previous_config.get('test_templates_sha256')!=frozen_test_sha256:
                raise ValueError('Frozen test-template hash mismatch')
    # Fail closed on incompatible resumes; do not replace a stream's provenance.
    metadata=deepcopy(provenance()); metadata['verifier']=args.verifier
    metadata.update(fresh_escalation=args.fresh_escalation,neutral_feedback=args.neutral_feedback,
                    diversify_seed=args.diversify_seed,test_templates_sha256=frozen_test_sha256,corpus_version=args.corpus,
                    small_model=args.small_model,large_model=args.large_model)
    metadata.update(consistency_verification=args.consistency_verification,consistency_k=args.consistency_k,
                    initial_routing=args.initial_routing,max_iterations=args.max_iterations,
                    generation_cache=args.generation_cache,cache_directory=str(Path(args.cache_directory).resolve()) if args.generation_cache else None)
    if args.initial_model!='small': metadata['initial_model']=args.initial_model
    metadata['evidence_status']='post-hoc exploratory; original template families already inspected'
    provenance_path=out/'provenance.json'
    if provenance_path.exists():
        previous=json.loads(provenance_path.read_text())
        if previous.get('corpus_version','v1')!=args.corpus: raise ValueError('Corpus version changed; use a new directory')
        if previous.get('small_model',SMALL)!=args.small_model or previous.get('large_model',LARGE)!=args.large_model:
            raise ValueError('Model pair changed; use a new directory')
        if previous.get('source_hashes')!=metadata['source_hashes'] or previous.get('verifier','original')!=args.verifier:
            raise ValueError('Source or verifier changed. Use a new output directory; existing records were not modified.')
        if previous.get('test_templates_sha256',frozen_test_sha256)!=frozen_test_sha256:
            raise ValueError('Frozen test templates changed since provenance was recorded. Use a new output directory.')
        if (previous.get('fresh_escalation',False)!=args.fresh_escalation
                or previous.get('neutral_feedback',False)!=args.neutral_feedback
                or previous.get('diversify_seed',False)!=args.diversify_seed
                or previous.get('initial_model','small')!=args.initial_model):
            raise ValueError('Runtime options changed. Use a new output directory.')
        if any(previous.get(k,default)!=metadata[k] for k,default in [('consistency_verification',False),('consistency_k',3),('initial_routing',False),('max_iterations',5)]):
            raise ValueError('Verification/routing budgets changed; use a new directory')
        if previous.get('generation_cache',False)!=args.generation_cache or previous.get('cache_directory')!=metadata['cache_directory']:
            raise ValueError('Generation cache changed; use a new directory')
    else:
        if any((out/name).exists() for name in ['branches.jsonl','validation.jsonl','test.jsonl']):
            raise ValueError('Existing evidence has no provenance. Use a new output directory.')
        provenance_path.write_text(json.dumps(metadata,indent=2)+'\n')
    if args.generation_cache:
        from study.generation_cache import GenerationCache
        installed={m['name']:m['digest'] for m in metadata['models']['models']}
        GenerationCache(args.cache_directory).bind({'provider_version':metadata['ollama'],
            'models':{m:installed[m] for m in (args.small_model,args.large_model)}})
    if not config_path.exists():
        dataset=json.dumps(study_corpus(args.corpus),indent=2)+'\n'
        config_path.write_text(json.dumps(configuration(args,hashlib.sha256(dataset.encode()).hexdigest()),indent=2)+'\n')
    if args.mode=='evaluate':
        # Protect a resumed stream against replacing its fitted predictors.
        models={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                for name in CONTROLLER_FILES.values() if (out/name).exists()}
        if 'task_routing' in args.policies.split(','):
            models['task_routing']=hashlib.sha256(router_path.read_bytes()).hexdigest()
        model_path=out/'evaluation_models.json'
        if model_path.exists() and json.loads(model_path.read_text())!=models:
            raise ValueError('Fitted controllers changed. Use a new directory for revised predictions.')
        if not model_path.exists(): model_path.write_text(json.dumps(models,indent=2)+'\n')
    ts=[t for t in study_corpus(args.corpus) if t['split']==('train' if args.mode=='collect' else args.split) and t['seed']<args.seeds and int(t['id'][-1])<args.wordings]
    rng=random.Random(20261012); rng.shuffle(ts)
    if args.mode=='collect':
        marker=out/'INFERENCE_STARTED.json'
        if not marker.exists(): marker.write_text(json.dumps({'dataset_sha256':json.loads(config_path.read_text()).get('dataset_sha256')})+'\n')
        path=out/'branches.jsonl'; existing=read_rows(path)
        states_path=out/'training_states.jsonl'
        states={(r['task_id'],r['fault'],r['trial'],r.get('stage','initial')):r for r in read_rows(states_path)}
        done={(r['task_id'],r['fault'],r['trial'],r.get('stage','initial'),r['action']) for r in existing}
        with path.open('a') as f:
            for task in ts:
                for fault in args.faults.split(','):
                    for trial in range(args.trials):
                        state_key=(task['id'],fault,trial,args.stage)
                        s=Session(task,trial,fault,verifier=args.verifier,fresh_escalation=args.fresh_escalation,
                                  neutral_feedback=args.neutral_feedback,diversify_seed=args.diversify_seed,
                                  small_model=args.small_model,large_model=args.large_model,
                                  consistency_verification=args.consistency_verification,consistency_k=args.consistency_k,initial_routing=args.initial_routing,
                                  generation_cache=args.generation_cache,cache_directory=args.cache_directory)
                        if state_key in states:
                            saved=states[state_key]
                            s.state=deepcopy(saved['state']); s.calls=saved['calls']; s.verification_calls=saved.get('verification_calls',0); s.operations=saved['operations']; s.trace=deepcopy(saved['trace'])
                        else:
                            original=states.get((task['id'],fault,trial,'initial')) if args.stage!='routing' else None
                            if original:
                                s.state=deepcopy(original['state']); s.calls=original['calls']; s.verification_calls=original.get('verification_calls',0); s.operations=original['operations']; s.trace=deepcopy(original['trace'])
                            elif args.stage!='routing': s.generate(args.large_model if args.initial_model=='large' else args.small_model)
                            stage_action={'checked':'verify','repaired':'repair','consistency':'verify_consistency'}.get(args.stage)
                            if stage_action:
                                if stage_action not in s.eligible():
                                    print('Skipped',task['id'],fault,args.stage,'stage action ineligible',flush=True)
                                    continue
                                s.apply(stage_action)
                            saved={'task_id':task['id'],'fault':fault,'trial':trial,'stage':args.stage,'state':s.state,'calls':s.calls,'operations':s.operations,'trace':s.trace,
                                   'verification_calls':s.verification_calls,
                                   'fresh_escalation':s.fresh_escalation,'neutral_feedback':s.neutral_feedback,'diversify_seed':s.diversify_seed,
                                   'generation_cache':s.generation_cache,'cache_directory':s.cache_directory if s.generation_cache else None}
                            with states_path.open('a') as sf: sf.write(json.dumps(saved)+'\n')
                            states[state_key]=saved
                        if all((task['id'],fault,trial,args.stage,a) in done for a in s.eligible()): continue
                        candidates=s.eligible(); rng.shuffle(candidates)
                        for action in candidates:
                            key=(task['id'],fault,trial,args.stage,action)
                            if key in done: continue
                            branch=s.clone(); branch.apply(action)
                            if action in ('accept','stop'): result=branch.result()
                            else: result=fixed_continuation(branch)
                            row={'task_id':task['id'],'template':task['template'],'split':'train','fault':fault,'trial':trial,'stage':args.stage,
                                 'corpus_version':args.corpus,'schema_variant':task.get('schema_variant','orders'),
                                 'small_model':args.small_model,'large_model':args.large_model,
                                 'fresh_escalation':s.fresh_escalation,'neutral_feedback':s.neutral_feedback,'diversify_seed':s.diversify_seed,
                                 'generation_cache':s.generation_cache,'cache_directory':s.cache_directory if s.generation_cache else None,
                                 'consistency_verification':args.consistency_verification,'consistency_k':args.consistency_k,'initial_routing':args.initial_routing,
                                 'feature_names':REVISION_FEATURES if args.consistency_verification else FEATURES,
                                 'features':features(s,feature_names=REVISION_FEATURES if args.consistency_verification else FEATURES),'action':action,'success':result['success'],
                                 'remaining_effort_seconds':result['effort_seconds'],'initial_state':s.state,
                                 'initial_trace':s.trace,'branch_result':result}
                            f.write(json.dumps(row)+'\n'); f.flush(); done.add(key)
                        print('Collected',task['id'],fault,'branches',len(candidates),flush=True)
        return
    controller=CONTROLLER_TYPES[args.controller].load(out/CONTROLLER_FILES[args.controller]) if 'learned' in args.policies else None
    table_controller=TableController.load(out/'table_controller.json') if 'table' in args.policies.split(',') else None
    router=None
    if 'task_routing' in args.policies.split(','):
        from study.task_routing import TaskRouter
        router=TaskRouter.load(router_path)
        if router.model['models']!=[args.small_model,args.large_model]:
            raise ValueError('Task-router model pair differs from evaluation configuration')
        if router.model.get('dataset_sha256')!=json.loads(config_path.read_text())['dataset_sha256']:
            raise ValueError('Task-router training corpus differs from evaluation configuration')
    path=out/(args.split+'.jsonl'); rows=read_rows(path)
    marker=out/'INFERENCE_STARTED.json'
    if not marker.exists(): marker.write_text(json.dumps({'dataset_sha256':json.loads(config_path.read_text()).get('dataset_sha256')})+'\n')
    done={job_key(r) for r in rows}
    jobs=evaluation_jobs(ts,args.faults.split(','),args.trials,args.policies.split(','),
                         penalties,thresholds,args.controller)
    rng.shuffle(jobs)
    with path.open('a') as f:
        for task,fault,trial,policy,penalty,threshold,controller_type in jobs:
            key=(task['id'],fault,trial,policy,penalty,threshold,controller_type)
            if key in done: continue
            options={}
            if args.fresh_escalation: options['fresh_escalation']=True
            if args.neutral_feedback: options['neutral_feedback']=True
            if args.diversify_seed: options['diversify_seed']=True
            if args.generation_cache: options.update(generation_cache=True,cache_directory=args.cache_directory)
            if args.initial_model!='small': options['initial_model']=args.initial_model
            if args.small_model!=SMALL: options['small_model']=args.small_model
            if args.large_model!=LARGE: options['large_model']=args.large_model
            if args.consistency_verification: options['consistency_verification']=True;options['consistency_k']=args.consistency_k
            if args.initial_routing: options['initial_routing']=True
            if args.max_iterations!=5: options['max_iterations']=args.max_iterations
            if policy=='task_routing': options['router']=router
            result=run_policy(task,policy,trial,fault,table_controller if policy=='table' else controller,
                              penalty,threshold,verifier=args.verifier,controller_type=controller_type,**options)
            f.write(json.dumps(result)+'\n'); f.flush(); done.add(key)
            print(args.split,task['id'],policy,fault,'success',result['success'],flush=True)


if __name__=='__main__': main()
