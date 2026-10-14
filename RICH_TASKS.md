# Richer development tasks and imperfect semantic checks

Implemented 12 October 2026. This is the second feasibility milestone, not a held-out research evaluation.

## Task suite

`pilot/richer.py` builds two related tables, customers and orders, and 72 tasks from 12 structural templates and six database seeds. The templates cover net revenue, month/date windows, region joins, distinct customers, average net amount, missing values, completion and discount percentages, thresholded counts, and combined summaries.

Reference answers are calculated from Python records, independently of SQL. SQL references exist only in infrastructure tests. Templates and seeds are retained as evaluation metadata and are not sent to the model. The first 12 tasks cover one instance of each template; these are development tasks and must not be reused as held-out test data.

Snapshots are exported under data/rich-development-public.jsonl and data/rich-development-evaluation.jsonl. Public records use opaque development IDs and omit reference answers and template labels; evaluation records map those IDs to templates and expected reports. The runner currently regenerates the same suite directly from code rather than reading these snapshots.

All rich-task baselines receive the same schema, natural-language request, and public structured contract. The contract flags are explicitly human-authored input semantics: whether cancelled records are excluded, and whether missing discounts mean zero. Automatic extraction from arbitrary user requests has not been implemented. This structured-input condition must be acknowledged in future comparisons.

## Two imperfect checks

1. Add cancelled records: for tasks explicitly excluding cancelled orders, execute the proposed SQL on a cloned database with additional cancelled orders. The result should be unchanged.
2. Replace missing discounts with zero: where the task explicitly defines missing discounts as zero, execute on a clone with those missing values replaced. The result should be unchanged.

The checks compare result multisets with numeric tolerance and preserve duplicates. They ignore row order and column aliases, which are not semantically relevant to these relations. They do not compare against expected answers. Failed probe execution is inconclusive rather than evidence that the candidate is wrong.

These relations can expose missing filters and wrong null treatment. They cannot establish that the query answered the requested question. For example, MAX(amount) on completed orders can remain unchanged under added cancelled records even when SUM(amount) was requested. Such false negatives are explicitly tested. More aggressive tests may also produce false alarms, particularly for order-sensitive or non-deterministic queries. This initial suite avoids those query classes.

No checker is claimed as novel. The contribution under investigation is choosing between verification and correction, as specified in RESEARCH_PROPOSAL.md.

## Runner options

Start the local server from the project root, using the existing model directory:

```sh
OLLAMA_MODELS="$PWD/.models" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_NO_CLOUD=1 ollama serve
```

Use distinct output filenames for each experiment:

```sh
python3 pilot/run.py --suite rich --limit 12 --output results/rich-small-clean-12.jsonl
python3 pilot/run.py --suite rich --model qwen3:1.7b --limit 12 --output results/rich-large-clean-12.jsonl
python3 pilot/run.py --suite rich --checks metamorphic --semantic-response escalate --limit 12 --output results/rich-check-escalate-12.jsonl
```

`--checks off` is the no-verification baseline. With `--checks metamorphic`, `--semantic-response observe` supplies evidence without forcing a correction; `repair` adds a correction instruction using the current model; `escalate` also switches to the stronger model after a violation. The controller remains rule-based, not learned. Model invocations and tool attempts still share the configured maximum step count.

Every query check logs its evidence and measured duration. A model may ignore feedback, submit an incorrect report after correct SQL, or keep repairing until the step limit. Final correctness is evaluated separately and never returned to the agent.

Current model serving remains cold per invocation (`keep_alive=0`). Context is limited to 2,048 tokens. The agent receives a compact summary of check names/statuses; full transformed results remain only in the log. Long tool outputs and repeated actions can still cause backend context truncation. Main-study code must add explicit context accounting before long runs. These development runs are for behaviour inspection, not definitive latency savings.

## Validation

Run `python3 -m unittest -v` from `pilot/`. Twelve checks currently cover the original environment and richer tasks: independent expected answers for all 72 instances, equivalent valid SQL, deliberate filter/null mistakes, a wrong query that passes the checker, multiset comparisons, and semantic escalation.

## Next milestone

Inspect actual-model failures; add independent checks for aggregation and date semantics without using hidden answers. Collect executed recovery branches from saved states only after the checks and richer tasks expose useful differences among actions. Keep verification accuracy, recovery effectiveness, and final report correctness separate.

## Actual-model development batch

One task per template (12 tasks, database seed 0), one trial per condition, completed locally on the M2 with existing quantized models. The three runs share the same task/schema/public-contract prompt and six-step limit. No models were downloaded during these runs.

| Condition | Correct reports | Median task seconds | Total task seconds |
|---|---:|---:|---:|
| Qwen3-0.6B, no checks | 3/12 | 2.44 | 38.37 |
| Qwen3-1.7B, no checks | 5/12 | 4.58 | 91.75 |
| Small model with checks and semantic escalation | 4/12 | 5.37 | 128.47 |

The checking baseline performed 25 probes, flagged seven query attempts, and spent approximately 0.11 seconds inside the probes. Most additional elapsed time came from extra inference and model loading, not the database checks. This run escalates only on semantic-check violations; SQL execution errors do not automatically switch models under its fixed policy. A combined execution-error and semantic-error recovery baseline remains to be tested.

Observed errors include selecting individual rows instead of aggregates, referencing region on the wrong table, hallucinating extra date filters, incorrect exclusive date boundaries, and generating PostgreSQL casts unsupported by the SQLite environment. Future prompt revisions should explicitly name the SQL dialect and be applied to all compared policies; their results must remain separate from this prompt condition. All current runs save their actual system/public prompts.

An offline audit of successful query executions from the two no-check runs flagged at least one query in three small-model tasks and one stronger-model task. Nine and seven final reports respectively were incorrect. These counts are not detector recall or precision: they use final task correctness rather than independently labelled per-query semantics, and some failed tasks never produced an executable query. The audit is not a live routed-model result.

In this sample, the checker rescued the average-net task but did not produce a useful overall time/success trade-off. No significance or generalization claim is supported by 12 development instances and one trial. Aggregate extraction errors, arbitrary extra predicates, and failed joins require additional observable evidence and stronger rule baselines before learning from recovery branches.

Saved evidence:
- results/rich-small-clean-12.jsonl
- results/rich-large-clean-12.jsonl
- results/rich-check-escalate-12.jsonl
- results/rich-offline-check-audit.json
- results/rich-development-summary.json
