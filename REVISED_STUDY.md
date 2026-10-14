# Exploratory controller and recovery revisions

Primary results retain the original failed-controller findings. Revised methods have no new model-run results. The twelve original template families have already been inspected; experiments on them are exploratory. Do not score contract_v2 on the old test families january, region_net or large_net. The current experiment CLI blocks that test split. Moving those families into v2 training and creating a fresh hashed test split belongs to the separate corpus phase; existing templates and splits remain unchanged here.

## Compatible action vocabularies and recovery

The global six-action ACTIONS list remains unchanged. Every fitted controller stores its own action list; encoding, prediction and selection use that list. Old controller.json files retain their original dimensions, scores and default choices. A controller that lacks escalate_fresh ignores that action even when the session offers it. New table or logistic training includes it when collected branches contain it. Template cross-validation preserves the training vocabulary across folds, including actions absent from a particular fold.

Session defaults use anchored escalation and the original feedback. --fresh-escalation enables escalate_fresh on error-free states while calls remain; it generates with the large model and empty feedback. From a large-model state it is eligible only with --diversify-seed. Execution errors use anchored escalation so feedback carries the error; flagged, executable states can use fresh escalation. --neutral-feedback separately replaces unflagged, error-free correction feedback with “Solve the task again from the request and contract.” Neither flag is enabled by default.

The escalate_fresh policy starts small, retries a transient tool failure directly, then escalates an executable candidate without previous SQL. Execution failures escalate with feedback. escalate_default retains previous-SQL feedback even if neutral feedback is requested. The duplicate large_first policy has been removed. --initial-model large now controls both collection and learned/table policy initial generation; fixed baselines retain their own routes. No learned task-type router is implemented in this phase.

## Training and validation

TableController smooths action × execution-error/current-large/flagged-check cells toward pooled action rates with prior strength 2. Logistic_v2 standardizes base features before shared/action/interaction encoding, shrinks interactions more strongly, and fits Platt scaling to out-of-fold training logits. Nested leave-one-template-out CV reports prediction metrics on training templates only. Logistic_v1 retains the original numerical method.

Train every controller before creating validation.jsonl. Fitted files, branches and evaluation populations must remain compatible; training after validation is intentionally blocked. The completion script now trains table and logistic_v2 with --cross-validate before evaluating any validation jobs. It stops after validation and never runs test. The commands below describe future execution; no collection or evaluation was launched in this fix phase.

```sh
python -m study.experiment collect --output results/v2/recovery --fresh-escalation --neutral-feedback --seeds 5
python -m study.experiment collect --output results/v2/recovery --fresh-escalation --neutral-feedback --seeds 5 --stage checked
python -m study.experiment collect --output results/v2/recovery --fresh-escalation --neutral-feedback --seeds 5 --stage repaired
python -m study.experiment train --output results/v2/recovery --controller table --cross-validate
python -m study.experiment train --output results/v2/recovery --controller logistic_v2 --cross-validate
python -m study.experiment evaluate --output results/v2/recovery --fresh-escalation --neutral-feedback --split validation --controller table --policies small,large,escalate_default,escalate_fresh,learned --penalties 0,0.05,0.15,0.3,0.6 --thresholds 0,0.05,0.1,0.2,0.3
python -m study.experiment evaluate --output results/v2/recovery --fresh-escalation --neutral-feedback --split validation --controller logistic_v2 --policies learned --penalties 0,0.05,0.15,0.3,0.6 --thresholds 0,0.05,0.1,0.2,0.3
python -m study.analyse select --directory results/v2/recovery --controller logistic_v2 --selection-rule v2
```

Before full collection, the protocol still requires a separate ten-task timing batch and a total-cost estimate. These commands are not a substitute for that step or the fresh-corpus freeze.

Job identity includes controller type, penalty and threshold; old rows without threshold mean 0.0 and old learned rows default to logistic_v1. Paired comparisons accept controller_type to select one controller and still reject duplicate operating points. Selection filters controller type; v2 selection flags degenerate success ties instead of rewarding inaction. Threshold gating removes accept and stop only while a supported recovery action remains; it does not guarantee calibrated correctness after recovery is exhausted.

Collection does not save checked/repaired states if verify/repair was ineligible. Provenance hashes runtime.py, controller.py, data.py, verifier.py, experiment.py and pilot execution modules, including environment.py and checks.py. Tests, dashboard, figures and offline analysis are excluded. Runtime flags and initial collection model are also checked when resuming. A source or configuration change requires a new directory.

configuration.json stores test_templates and their canonical SHA-256. Test evaluation requires an existing matching hash, exact agreement with the corpus test families, no train/validation overlap, and no seen v1 test families. This permits future fresh corpora once frozen; no real test was run in this phase. Freeze refuses to overwrite an existing configuration. selection.json and table_selection.json record analyse_sha256.

The mutation audit records distinct SQL strings per family and a committed ten-family design/ten-family evaluation allocation. The allocation alternates alphabetically sorted existing families and is fixed for future verifier changes. All family outcomes were already observed before this split, so its evaluation partition is retrospective. Component ablations remove each of eight keyword rules with the net-value probe kept or removed, and remove the probe separately. They recompute flags from recorded evidence without changing either verifier or selecting a rule set. Full, design and evaluation reports are written to results/v2/mutation_audit.json.

## Detector evidence and offline diagnostics

contract_v2 remains exploratory and corpus-specific. It adds net-value-preserving probes and lexical concerns to the original checker. No new checks were designed from old test failures in this phase. missing_count currently has null_discount_as_zero=False, so its legitimate discount IS NULL reference does not receive the net-preservation probe. A regression executes every train/validation reference query over all ten seeds and both wordings and requires flagged=False.

The diagnostic reports saved initial states, reference SQL and saved correct large-model outputs separately. Specificity is measured on reference SQL and correct large outputs; an empty correct-output population yields null specificity. Reference executions test false alarms on supplied correct queries, not model competence, and repeated wordings are not independent model predictions. Seen v1 test families are excluded before executing checks. Saved large outputs are deduplicated by task, trial and SQL.

```sh
python -m study.diagnostics --directory results/v2/recovery --verifier contract_v2
```

This writes results/v2/recovery/detector_diagnostic.json. Reading original development evidence is also allowed with an explicit output under results/v2; it does not score the old test split. The diagnostic includes small-only validation success and initial small accept-branch success by template, without inference. These counts can motivate a task-type routing comparison, but cannot establish its future test performance.

For original saved demonstrations:

```sh
python -m study.demo --template january --policy learned --replay --directory results/main
```

Live learned demos need both the chosen fitted model and validation selection under results/v2. The CLI explains missing artifacts rather than failing with a file error. New live demonstrations and diagnostics are exploratory; neither changes the frozen report's primary result.
