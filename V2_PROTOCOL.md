# Exploratory v2 corpus and experiment protocol

This addendum is dated 13 October 2026. V1 remains the primary study result. V2 is exploratory; never pool its measurements with results/main. The original v1 templates, splits, scoring function and published evidence remain unchanged.

## Dataset and holdouts

study/data_v2.py defines a separate corpus version. Each template has ten seeds, two wording variants, an independent Python-record oracle, reference SQL, equivalent SQL and deliberately wrong SQL tests. Authoring checks validate those fixtures; they are not model test outcomes.

| Split | Templates | Instances with both wordings | Primary tasks using wording 0 |
| --- | ---: | ---: | ---: |
| Training | 8 old | 160 | 80 |
| Validation | 4 old | 80 | 40 |
| Fresh test | 8 new order-analysis | 160 | 80 |
| Schema shift | 4 new inventory | 80 | 40 |

Training contains net_total, january, region_total, region_net, average_net, missing_count, distinct_customers and large_net. Validation contains date_window, completion_rate, discount_rate and summary. The three burned v1 test families are training-only in this version.

The fresh order templates cover top-N regions, cancelled/total ratio, customers without orders using LEFT JOIN, discount sums by region, monthly date-window grouping, HAVING counts, maximum per customer and weighted net average. Inventory adds warehouse stock value, low-stock distinct products, top stock-value products and warehouse unit share.

Seed changes now affect statuses, null positions, dates, customer participation and values. Three initial completed orders ensure each requested region has a defined observation; remaining status patterns vary by seed. References are validated independently on all ten seeds. Ratio contracts include a human-authored 0–100 range and denominator description supplied unchanged to every policy. These fields contain no gold counts or labels.

Freeze the complete corpus and both template lists before inference under that corpus revision:

```sh
python -m study.experiment freeze --corpus v2 --output results/v2/corpus_v2
```

configuration.json records corpus_version, dataset_sha256, corpus_source_sha256, test_templates_sha256 and schema_shift_templates_sha256. Freeze never overwrites an existing configuration. Model prompts and features exclude corpus/template IDs, split labels, database seeds and gold answers.

## Headroom and timing gate

Run a ten-task timing batch: six training and four validation templates, both candidate models, one fresh generation and execution per model. There is no controller, verifier or recovery. Print the estimated full cost before launching collection.

```sh
python -m study.headroom timing --corpus-freeze results/v2/corpus_v2 --directory results/v2/headroom/qwen3-06-17
python -m study.headroom profile --corpus-freeze results/v2/corpus_v2 --directory results/v2/headroom/qwen3-06-17 --seeds 10
```

Profile only train/validation tasks. Timing observations are reused once in the profile, rather than rerun. Logs are append-only and completed jobs resume without re-execution. Model digests, source hashes, decoding and corpus hashes are recorded. Provider failures stop the process and are not scored as model quality.

The predefined headroom gate requires the small model to achieve at least 50% correctness within 30–70% of development templates, and the large model to have higher overall development correctness. Report all per-template counts. Failure of the installed pair means no pair has been selected; do not silently relax the gate or tune prompts against the fresh test. Alternate installed tags can be profiled in a new directory with --small-model and --large-model. Downloads and resource requirements must be considered before choosing another pair.

The timing estimate is an upper-budget estimate, not a measured controller runtime prediction. It excludes extra consistency samples and must be refreshed if model tags or runtime budgets change.

## Collection, training and validation order

The following stages are pending until headroom is established and the verifier phase is complete. No full collection, validation or held-out model run is authorized by the dataset freeze itself.

Use separate directories for each initial model, controller type and runtime flag combination. A suggested layout is results/v2/runs/<pair>/<initial>/<controller>/fresh-<0|1>-neutral-<0|1>-diverse-<0|1>. Include verifier and corpus versions in each configuration. Generate an immutable snapshot per directory; never overwrite or append incompatible results. For the same initial-model/flag set, collect branches once and initialize controller directories from that preserved collection before evaluation.

Collect initial, checked, repaired, consistency and pre-generation routing stages on at least five training seeds. Skip stages whose action was not applied. Consistency uses k=3, temperature 0.7, and seeds trial+generation_calls+1 through +3; its recorded model and live execution cost is charged as verification effort. Do not assume repeated agreement means correctness.

Train table and logistic_v2 with --cross-validate before creating any validation.jsonl. Compare against logistic_v1 on training-only leave-one-template-out CV. Logistic_v2 excludes stop rows from fitting and calibration; predictor CV excludes stop rows for every method. Policy outcomes still count stopping as failure.

Validation uses all ten seeds and one wording: **40 tasks**, because this version has four validation templates. Use one trial, clean tools, and the small nine-point grid penalty={0,0.15,0.3} × threshold={0,0.1,0.2}. Select using select_operating_point and report selection_mode. A degenerate tie is no evidence for an operating point. Selection records analyse_sha256.

--initial-model controls learned/table initial generation. --small-model and --large-model bind runtime roles. Fresh escalation is allowed only in error-free states; from a large-model state it additionally requires diversify_seed. Execution-error escalation keeps previous SQL and the observed error. --neutral-feedback and --diversify-seed default off; enabled diversification uses trial+calls after incrementing calls. Each collected branch records the corpus, model pair and runtime switches. All baselines receive the same schema, public request and contract.

## Pre-registered success criterion

The primary revised controller is logistic_v2 at its validation-selected setting. For each clean/transient condition, require its success rate to be no more than 10 percentage points below the best simple baseline and its mean effort to be lower than that same baseline. Simple baselines are small, large, anchored escalation, fresh escalation, rules and task-type routing. At equal baseline success, the comparison baseline is the one with lowest mean effort. Both conditions must satisfy the criterion before claiming a useful benefit. Per-template raw counts must accompany every aggregate.

This is a descriptive practical tolerance, not a formal non-inferiority test. Model capability, family diversity and coarse cluster intervals limit the claim. Otherwise report a second negative result; do not change the tolerance after seeing the test.

## Final freeze and one-shot test

Only after collection, training, validation and all required baselines are implemented, write FREEZE.md and a machine-readable manifest with the clean code git hash, source hashes, model digests, controller hashes, selected penalties/thresholds, selection_mode, runtime flags/budgets, test and schema-shift split hashes, dataset hash and this protocol's hash. No controller hashes or selected settings exist yet; a dataset freeze is not this final freeze.

The final policy set is small, large, escalate_default, escalate_fresh, rules, task-type routing, table, logistic_v2 and its model-only restriction. Task-type routing uses public request/contract categories and training-only calibration. A constant routing outcome provides no evidence of useful task discrimination. The oracle-routing diagnostic is separate and labelled an upper bound on choosing between the two fixed baseline outcomes; it must not enter model features or policy comparisons.

Primary test: eight new templates × ten seeds × one wording × three trials × clean/transient × nine policies = 4,320 trajectories. Schema shift is separate: four inventory templates under the same schedule = 2,160 trajectories. Run every condition once; resume only jobs not yet executed, without replacing completed observations. Do not rerun adverse conditions or tune on either holdout.

Report per-template successes/runs, template-cluster intervals and a decomposition of detector sensitivity/false alarms, recovery success after flags, and final correctness. Preserve numerator/denominator populations and note shared tasks/states. Schema-shift, oracle upper bounds and retrospective mutation-family evaluations remain separate from primary comparisons. The historical mutation-family split was made after its outcomes were observed and cannot be described as a clean unseen holdout.

## Current state

Corpus authoring, task routing, revised probes, consistency verification, generation replay and the guarded one-shot workflow are implemented. Historical development profiling selected a pinned 1.7B/4B pair, but those profiles predate the amendments below. They are preserved and cannot be resumed or pooled with the revised corpus. Revised development collection, fitted controllers, final freeze, live replay sample and fresh held-out model results remain pending. Infrastructure tests do not establish model quality.

## Dated pre-inference amendments — 13 October 2026

These researcher-authorised amendments precede any inference under the amended requests/contracts and any fresh test or schema-shift inference. Earlier development inference already occurred and is historical evidence, not evidence for the amended configuration. The previous corpus, preregistration and source snapshots are preserved in results/v2/amendments/2026-10-13. A corpus amendment is refused after a final FREEZE.json, TEST_STARTED.json, nonempty held-out outcome stream, or development inference marker/stream matching the active amended dataset hash. Historical development streams with the previous hash remain allowed. Regenerate corpus/oracle fixtures, source and per-template hashes, and the preregistration payload hash; the ordered test-family lists remain unchanged. No held-out model outcomes informed these changes.

1. **13 October 2026 — Explicit net semantics.** preserve_net_value requires the public net_value field to be true and the existing sum/average/count eligibility. A missing field is false; NULL-discount handling alone does not imply net semantics. Weighted means are excluded because the intervention also changes their weights. Check every reference query on all ten seeds and both wordings for original, contract_v2 and contract_v3 across all four splits. This is authoring QA, not held-out model evaluation.
2. **13 October 2026 — Exercised constraints.** month_totals now uses 10 January inclusive to 20 February exclusive; having_regions requires at least 24 completed orders. Record-oracle tests require omitting the window, either endpoint independently, or HAVING to change the answer on at least 70% of seeds. Preserve both empty and non-empty HAVING answers. These changes were specified before amended-model inference.
3. **13 October 2026 — Opt-in generation replay.** --generation-cache defaults off; the prospective workflow enables it explicitly in a new run directory. The key is SHA256 of json.dumps(request_body).encode(), exactly matching the sent body: model tag, public messages, sampling options and decoding seed are included. Gold, task IDs, split and database seed remain absent. Cache records contain the complete response, recorded usage, original wall time and integrity hash; files are published atomically without replacement. Provider version and model digests bind the cache context. Freeze hashes the context and all existing records; subsequent misses append new records. SQL is parsed and executed on the current database for every logical call, including transient faults and consistency samples.
4. **13 October 2026 — Accounting and interpretation.** Charge recorded load_duration, prompt_eval_duration and eval_duration on both hits and misses; measure tools/probes live. This is replayed charged effort, not live inference latency or physical resource consumption. Report logical calls, live calls, cache hits, replayed model effort and observed wall time separately. Cache reuse couples generations across database seeds; seeds with the same public body are not independent model samples. First-fill loading/warming affects recorded durations. After final freeze and before either holdout, rerun the fixed ten development timing tasks with both models, trial 0 (20 live requests), bypassing cache. Compare SQL, executed results, usage and wall time; retain all mismatches and all original cached responses. This sample cannot tune rules/settings. Any mismatch limits claims to the cache-defined experiment; it cannot establish uncached deployment savings. A started live sample is never silently rerun after interruption.

   The nine-policy schedule has 4,320 primary and 2,160 schema-shift trajectories. The separate oracle has 480 primary and 240 schema-shift jobs, each executing two baseline candidate trajectories: 1,440 candidate trajectories. Total candidate trajectories are 7,920, excluding development and the 20-request live sample. Consistency samples and cache hits are model requests within trajectories, not additional policy trajectories. Count physical inference separately from this logical schedule.

   Read the unchanged ten-point criterion with template counts: eight equally weighted test families mean losing an entire family costs 12.5 points, already beyond the tolerance. Four validation or schema-shift families mean one family costs 25 points; at 40 validation tasks, ten points equals four task outcomes. Three trials/fault repeats do not create new independent families. Show all family numerators/denominators and do not treat the criterion as a formal non-inferiority test. At the historical nine-task validation size the tolerance was effectively exact match; the revised validation has 40 tasks.
5. **13 October 2026 — Development-designed probes.** PROBE_DESIGN.json records date-boundary insertions (before inclusive start and at exclusive end), below/at-threshold records for an eligible ungrouped net-count contract, cancelled rows with NULL discounts, and a discount-value intervention. The last runs only when the human-authored discount_irrelevant contract field is true; it preserves NULL positions. Every policy receives that same field. New probes run only in opt-in contract_v3; retain the net-value probe and existing v1 default verifier. Design examples use train/validation requests/contracts only. Keep the committed mutation-family allocation unchanged, report design and evaluation recall and false alarms separately, add distinct SQL counts, ablate each new probe, and report false alarms for all authored reference SQL. The evaluation families are retrospective and were seen before allocation; this is not a clean unseen detector test. Oracle-matching but semantically constrained mutants remain in the false-alarm population even if a counterfactual probe raises a concern. Do not change probes after reading evaluation-family results.
6. **13 October 2026 — Development split swap.** Move distinct_customers from validation into training and completion_rate from training into validation. Counts remain eight training and four validation families. Update every task's split, configuration groups, dataset/template hashes and preregistration payload. Preserve the immediately preceding corpus snapshot under amendments/2026-10-13/nonnegative-split/corpus_before. Test and schema-shift template lists and their hashes are unchanged. Earlier profiles retain their original split metadata and are historical only.
7. **13 October 2026 — Public nonnegative constraint.** Add an optional human-authored nonnegative=true contract field to sum and explicit net-value tasks in v2. This sign constraint is supplied identically to every policy, contains no gold magnitude, and defaults absent/disabled for v1. The opt-in contract_v3 verifier checks numeric report values against zero, excluding grouping labels; empty/NULL outputs are inconclusive. Negative or non-finite values violate the sign constraint. Register two development-only negative-result SQL forms before rerunning the existing retrospective design/evaluation-family audit. Report both partitions, the added design cases, removal of the sign probe and false alarms on all authored reference SQL; do not change the probe in response to evaluation-family results.
8. **13 October 2026 — Offline economics diagnostic.** study.economics reads a paired development profile without model calls and reports per-template model correctness and charged effort. Always-small and always-large retain their recorded costs. Oracle pre-routing selects a cheaper correct recorded candidate, or the cheaper candidate when both fail: this is a finite-sample hindsight diagnostic. Small-first with a perfect detector charges small on every task and fresh-large on small failures, assumes zero detector overhead, and has union correctness. It is an optimistic bound, not measured detector/recovery performance or anchored escalation. Include all failed effort in effort-per-success, retain profile provenance and original split fields, and label historical profiles as incompatible with the active amended corpus. Write results/v2/headroom/economics.json. No evidence from this diagnostic changes held-out settings or the success criterion.
