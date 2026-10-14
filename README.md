# Adaptive model routing with verification and recovery

A local SQL-agent routing, verification, and recovery project by Puneet Devnani.

This completed local prototype studies how a SQL-analysis agent chooses acceptance, verification, small-model repair, stronger-model escalation, direct tool retry, or stopping. A lightweight controller predicts final correctness and remaining measured effort from observable state features. It uses Qwen3-0.6B and Qwen3-1.7B through Ollama and requires no paid API.

**Main finding:** the learned controller did not improve the success–effort trade-off. On each primary held-out condition, the stronger-model baseline achieved 18/27 correct runs (66.7%) and the joint controller achieved 0/27. Lower effort accompanied correctness loss. This is a reproducible engineering and empirical failure-analysis project; it does not establish a generally superior router or a reliability guarantee.

## Deliverables

- `output/portfolio/dashboard.html`: self-contained offline results and recorded execution viewer; open directly in a browser.
- `results/main/summary.json`: measured results and descriptive uncertainty intervals.
- `results/main/test.jsonl`: 270 primary runs and 18 sensitivity runs.
- `results/main/controller.json`: fitted numerical controller.
- `PORTFOLIO_GUIDE.md`: demonstration and interview guide.
- `REPRODUCIBILITY.md` and `STUDY_PROTOCOL.md`: exact scope, conditions, and workflow.

The dashboard reads saved traces. Its displayed durations belong to those runs. Use the CLI below for fresh inference. Candidate, supervisor, and programme-chair signatures/designations/dates remain blank in the certificate.

## Quick start

Use Python 3.11 or later in a virtual environment. Install Ollama from its official distribution or Homebrew. From the extracted project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OLLAMA_MODELS="$PWD/.models" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_NO_CLOUD=1 ollama serve
```

Keep that terminal running. In a second terminal, activate the same virtual environment and download the models through the running server:

```sh
source .venv/bin/activate
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
python -m study.demo --template large_net --seed 1 --policy large --fault transient
python -m study.demo --template net_total --policy escalate_fresh
```

The first command demonstrates direct recovery from an injected first-query timeout. A verified original live run returned `{"count": 29}` correctly. The second starts small and regenerates with the large model from the public request alone; this new policy has no measured benchmark result yet. Model tags and decoding can change outcomes; stored digests and original traces establish the completed study's evidence. A live learned demo requires a trained controller and validation selection under results/v2; those revised artifacts have not been collected yet.

For a demo without downloading models:

```sh
python -m study.demo --template january --policy learned --replay --directory results/main
```

This prints **SAVED TRACE REPLAY**. It is not fresh inference. The small numerical controller still needs numpy installed.

## Implementation

| File | Responsibility |
| --- | --- |
| `study/data.py` | 240-instance synthetic corpus and template-grouped splits |
| `study/data_v2.py` | Separate 480-instance corpus, seeded structure and inventory holdout |
| `study/headroom.py` | Development-only timing batch and controller-free model-pair profile |
| `study/task_routing.py` | Public-contract routing fitted and calibrated on training profiles only |
| `study/runtime.py` | Local inference, bounded read-only SQL, checks, recovery and accounting |
| `study/controller.py` | Action-specific logistic success and ridge effort predictors |
| `study/experiment.py` | Resumable branch collection, training and evaluation |
| `study/analyse.py` | Validation-only penalty selection and template-cluster analysis |
| `study/diagnostics.py` | Offline development diagnostic of verifier misses |
| `study/demo.py` | Live CLI and explicitly labelled replay |
| `study/dashboard.py` | Standalone recorded-result dashboard |

Every policy receives the same schema, request, and human-authored structured contract. The deterministic formatter copies SQL results into the requested report shape and cannot fix incorrect SQL. Gold answers, template IDs, split labels and database seeds are excluded from model requests and controller features. V1 learned policies always start with the small model. Exploratory v2 additionally allows --initial-model large; this option does not change the primary results.

## Verification and reproduction

```sh
python -m unittest discover -s pilot -p 'test*.py'
python -m unittest study.test_study
```

The required suites currently pass **12 pilot tests and 123 study tests**. The study runner includes RevisionTests; the separate study.test_revision command passes 24 overlapping tests. Coverage includes the original 240 instances, all 480 v2 instances, independent oracles, constraint omissions, grouping, leakage, v1 request compatibility, cache replay/accounting, the live-sample guard, headroom, task routing, detector audits and offline economics. Mocked providers establish infrastructure behaviour, not model accuracy.

The study collected 292 executed action branches from 72 states, evaluated 63 validation trajectories, then froze penalty 0.3 before testing. Reproduction instructions, including collection stages and the resumable workflow, are in `REPRODUCIBILITY.md`. Use a separate output directory for changed configurations; completed records are intentionally skipped on resumption.

Measured-result plots are retained in `output/figures/`. The dashboard displays saved executions.


## Limitations and earlier development

Only three held-out template families, nine unique test tasks, and one wording variant were evaluated. Repeated runs are not 27 independent families. The reported cluster intervals are coarse. No-effort and history-masked modes perturb the fitted controller rather than retraining ablations. In a clean training-state diagnostic, the verifier missed all 16 executable wrong results. Public contracts are supplied, not automatically extracted. Measured effort is hardware-specific time, not monetary API cost, FLOPs or energy.

Earlier cold-serving development pilots remain in `pilot/`, `PILOT_FINDINGS.md`, and `RICH_TASKS.md`. Their prompts, formatting and serving conditions differ; do not pool their scores with the main study. `PROJECT_BLUEPRINT.md` and `RESEARCH_PROPOSAL.md` record earlier plans; the implemented scope is defined by `STUDY_PROTOCOL.md`.

PROJECT_DESCRIPTION.md gives a complete project explanation. The original targeted literature audit covers twelve closely related primary papers in `literature/LITERATURE_COMPARISON.md`. Routing, verification, fallback and recovery already have substantial prior art; this project claims an implemented local system and measured limitations, not priority for those ideas. Source links and hashes are in `literature/SOURCE_MANIFEST.json`.

The portable package excludes model weights, downloaded third-party papers, dependency caches and render intermediates. Model downloads are required for live use; the recorded dashboard works offline.

## Exploratory revision

Current development code adds acceptance thresholds, smoothed empirical-table and calibrated logistic controllers, fresh escalation, --initial-model routing for learned policies, and opt-in contract_v2 verification. See REVISED_STUDY.md for training order, validation grids, hashed split guards and evidence boundaries. The primary evidence retains the original failed-controller study. No revised held-out model accuracy is claimed.

The amended corpus and prospective workflow are in V2_PROTOCOL.md, with eight dated pre-inference amendment notes. Its eight fresh order templates and four inventory templates have hashed holdout lists. The earlier development profiles predate the amended contracts and remain separate. study.workflow provides timing, development, final freeze, a fixed live replay sample and the one-shot test ledger; the revised model experiments remain pending. Task-type routing is implemented; TASK_ROUTING_PROTOCOL.md records its rule before fitting. V2 uses four validation templates, so ten seeds give 40 tasks. Generation caching defaults off and reports replayed charged effort separately from live calls and wall time. REPRODUCIBILITY.md describes current flags and safe output paths.
