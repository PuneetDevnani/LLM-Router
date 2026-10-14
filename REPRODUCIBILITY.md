# Reproducibility and evidence boundaries

V1 is the primary completed result. Its evidence is read-only in results/main; the saved evidence retains those findings. The active source has exploratory revisions. Do not execute current generation or analysis commands against results/main. Exact historical sources and model digests are in the preserved release package and provenance; use a separate checkout for source-identical historical reproduction. Already-seen v1 test families are not fresh v2 holdouts.

## Environment and checks

Python, numpy and Ollama are needed for live inference. The completed runs used Python 3.12.14 and numpy 2.3.5; Python 3.11 or later is the intended source target. Report builders also need python-docx, pypdf and a renderer; figures need matplotlib. The bundled runtime is separate from inference dependencies.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s pilot -p 'test*.py'
python -m unittest study.test_study
python -m unittest study.test_revision
```

study.test_study includes RevisionTests plus corpus, runtime, offline-policy, mutation-audit, split-guard and headroom tests. Mocked providers verify infrastructure, not model accuracy. The frozen-hash test protects published evidence, including the original detector diagnostic.

## Original result replay

```sh
python -m study.demo --template january --policy learned --replay --directory results/main
```

This is saved-trace replay, not fresh inference. Live learned demos require fitted v2 controllers and selection files; those revised model artifacts have not been collected yet. The old diagnostic source is retained inside the read-only archive by the user's explicit choice. The active no-argument diagnostic writes to results/v2.

## Exploratory v2

V2_PROTOCOL.md defines the versioned dataset and pre-registered gates. study/data_v2.py keeps old v1 definitions untouched and adds seeded structural variation, eight fresh order templates and four inventory templates. Train/validation use eight/four old templates; fresh test and schema shift use new families only. All tasks have independent record oracles, two wordings and ten seeds.

```sh
python -m study.experiment freeze --corpus v2 --output results/v2/corpus_v2
```

This freezes 480 instances and hashes both held-out template lists before model calls. Repeating freeze fails without modifying files. The corpus source and bytes are checked by the headroom tool. Current inference/controller defaults reproduce v1 behaviour; --corpus v2 is explicit.

Start the local server when conducting the separately planned profiling stage:

```sh
OLLAMA_MODELS="$PWD/.models" OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_NO_CLOUD=1 ollama serve
```

Then run the ten-task timing batch and inspect the cost estimate before a full profile or collection:

```sh
python -m study.headroom timing --corpus-freeze results/v2/corpus_v2 --directory results/v2/headroom/qwen3-06-17
python -m study.headroom profile --corpus-freeze results/v2/corpus_v2 --directory results/v2/headroom/qwen3-06-17 --seeds 10
```

Both models generate once per development task; there is no controller, verifier or recovery. Timing records are reused once. Completed JSONL jobs resume without re-execution. Provider outages propagate. A failing pair remains a negative finding and does not justify changing fresh test templates.

## Runtime switches and future experiments

--initial-model small|large controls collection and learned/table initial generation. --small-model/--large-model configure model roles; defaults remain Qwen3-0.6B/Qwen3-1.7B. --fresh-escalation enables clean/flagged regeneration from the request; execution errors retain feedback. Large-to-large fresh regeneration requires --diversify-seed. Diversified seeds are trial+calls after calls increments. --neutral-feedback avoids asserting an unflagged error-free query is wrong. All switches default to v1 behaviour and appear in configuration, provenance and collected branches. Changing source, model pair, corpus or switches requires a new directory.

After a suitable pair and verifier are established, collect initial/checked/repaired states on at least five training seeds. Train table and logistic_v2 with --cross-validate before validation. Keep directories separate by initial model, controller and flags. Validation uses ten seeds: four templates mean 40 tasks. Use the nine-point grid in V2_PROTOCOL.md and select_operating_point; report selection_mode and analyse_sha256.

Task-type routing is implemented in study/task_routing.py; its fixed rule is in TASK_ROUTING_PROTOCOL.md. After a suitable-pair profile is complete, fit with `python -m study.task_routing --profile-directory <chosen-profile> --output results/v2/routing/task_router.json`. The CLI excludes validation rows, uses training-template OOF Platt calibration, and preserves an existing fit. Evaluate policy task_routing with --router pointing to that artifact. Its model pair and dataset must match evaluation, and its hash enters evaluation_models.json.

Before the fresh test, write FREEZE.md with the clean code git hash, fitted controller/router hashes, selected settings, model digests and frozen split hashes. study.workflow implements the development/freeze/live-sample/one-shot/summarise sequence and its attempted-job ledger. Do not run a partial held-out comparison. Fresh test and schema shift must be scored once, with three trials and clean/transient conditions, raw template counts and descriptive cluster intervals. Never pool v2 with v1.

13 October 2026 pre-inference amendments are recorded individually in V2_PROTOCOL.md. `python -m study.preinference amend --revision nonnegative-split` regenerates the amended corpus, independent-oracle fixtures, per-template hashes and PREREGISTRATION.json payload hash, preserving the prior corpus snapshot. It refuses amendments after a final freeze or held-out job. `python -m study.preinference audit --output results/v2/amendments/2026-10-13/nonnegative-split/mutation_audit.json` makes no model calls and writes a separate development mutation audit plus false alarms on all authored reference fixtures. Never use its held-out reference fixtures to tune against model outcomes.

`--generation-cache --cache-directory results/v2/<run>/generation_cache` is opt-in in experiment.py and recorded in configuration, provenance and branch rows. The prospective workflow enables it explicitly. Exact request bytes are hashed, responses/usage replayed, and SQL executed live for every database instance. Effort charges recorded model durations, including on hits; report cache hits, live/logical calls and observed wall time separately. Provider digests/version and pre-freeze cache contents are protected. After the final freeze, the `study.workflow live-sample` command runs twenty fixed development requests with cache bypass before `test`; it is a single-attempt sample and does not replace cached observations. No revised model inference was executed during the amendment phase.

## Offline exploratory tools

```sh
python -m study.offline_eval
python -m study.diagnostics --directory results/main --output results/v2/detector_diagnostic.json --verifier contract_v2
python -m study.mutation_audit
```

These make no model calls. Offline policy replay evaluates one recorded branch decision with its v1 continuation, not full controller deployment. Diagnostics report all development templates, including zero-run cells; correct large outputs deduplicate by task and SQL. Mutation audits report distinct SQL strings, retrospective locked family partitions, per-family error/false-alarm counts and component ablations. New probes or keyword removals require a separate verifier phase; historical family outcomes have already been seen.

## Documents and figures

The original 49-page report and supporting manuscript remain retained unchanged. The revised report is in results/v2/report/ as DOCX and PDF, with twenty-five references, updated pagination and separate exploratory analyses. Rebuilding requires rendering and visually checking every page, then recalculating the static contents map. Never regenerate the published evidence to make it match revised code.

The latest pre-inference split puts distinct_customers in training and completion_rate in validation. Holdout template lists and hashes are unchanged. Optional nonnegative contracts are supplied identically to all policies and checked only by contract_v3; original/v2 verifiers and missing-field behaviour are preserved. Run offline profile economics with `python -m study.economics <profile.jsonl> --small-model <tag> --large-model <tag> --output results/v2/headroom/economics.json`. Existing outputs are preserved; choose a new filename for a different profile. Historical profile costs are optimistic recombination diagnostics, not measured amended-controller performance.
