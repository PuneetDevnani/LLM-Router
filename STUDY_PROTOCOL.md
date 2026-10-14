# Main study protocol

Frozen design prepared on 12 October 2026 before held-out runs. Earlier pilot data are development history and are not pooled with this study.

## Scope

This is a resource-limited empirical study of a post-generation verification/recovery controller for structured-input SQL analysis agents. Initial generation always uses the small model for learned policies. It is not a general-purpose language-model router trained on arbitrary requests.

The corpus contains 240 instances: 12 structural templates, 10 database seeds, and two wording variants. All variants of a template stay in the same split. Six templates train the controller, three validate decisions, and three test transfer. Only one wording per seed is used in the primary experiment to avoid treating paraphrases as independent evidence. The extra wording variants remain available for supplementary checks.

Public input includes the schema, request, and human-authored task contract. The contract declares aggregation, grouping, date constraints, distinctness, null conventions, and report shape. These inputs are supplied to every policy. Gold answers, template names, split labels, and database seeds do not enter the model prompt or learned feature vector. Database seeds are used only to instantiate and reset the environment.

## Shared agent

Each model proposes read-only SQLite SQL in JSON. Queries execute through the bounded tool, and a shared deterministic formatter maps the result to the declared report schema. This removes a separate model call for copying numeric results into JSON. The formatter cannot repair incorrect SQL. Each trajectory permits at most three model calls, six main-tool attempts, and five controller decisions. Verification probes are separately counted and timed; they are not hidden free model calls.

Contexts are 4,096 tokens with bounded fresh request/feedback rather than an ever-growing transcript. Decoding uses temperature 0.2 and a recorded seed, with a 384-token output cap. Ollama keeps the active model warm; the server is limited to one loaded model. Actual loading durations remain included in effort. Experiment jobs are shuffled with a fixed seed to reduce systematic serving-order bias.

## Verification and decisions

Two database invariances are inherited from development. Fallible lexical concerns additionally inspect visible aggregation, grouping, ratio operations, distinctness, and date endpoints. Equivalent expressions can trigger false alarms, and passing all checks cannot establish correctness. These signals must be evaluated empirically.

Actions are accept, small-model repair, stronger-model escalation, verification followed by a fixed recovery continuation during data collection, direct transient retry, and explicit stop. Training samples execute each eligible action from the same saved state. For non-terminal actions the fixed continuation retries a transient error or escalates on an error/flag where budget permits. Labels are terminal correctness and measured incremental effort. Saved states are reused on resumption so new branches do not silently use a different initial output.

The controller fits L2 logistic regression for success and ridge regression for log1p effort using observable features and action interactions. It chooses predicted success minus a validated effort penalty. This is an empirical action-value approximation under the collection continuation, not an optimal-policy or formal reliability guarantee. Repeated deployment decisions can induce a distribution mismatch, tested through fresh trajectories.

## Experiments and analysis

Training collection starts with several seeds per training template under clean and transient-failure conditions. Validation compares effort penalties and selects one before testing. Held-out runs compare small-only, stronger-only, combined error/verification rules, the joint learned controller, and a learned model-only action restriction. Supplementary no-effort/no-history modes diagnose sensitivity; unless retrained, they are feature/action perturbations rather than clean retrained architectural ablations.

Predefined operating-point selection: among the candidate penalties, prefer the lowest mean validation effort whose point success rate is within 10 percentage points of the stronger-model validation success. If none qualifies, prefer highest validation success and break ties by lower effort. The 10-point tolerance is a practical selection rule for this small study, not a claim of statistical non-inferiority. It is defined before validation execution.

Primary fault conditions are clean tools and a first-operation transient timeout. Persistent outage tests verify termination separately. A timeout is injected at the first main-tool attempt for each affected trajectory; all policies face this same operation-index schedule. Verification operates on the database and is not called during an outage.

Deployment effort adds model loading, prompt evaluation, generation, and measured tool/check duration. End-to-end wall time, token counts, calls, model switches, and failed-run expenditure are also reported. Training collection cost is separate from deployment cost. Hardware-specific timings are not FLOP, energy, or monetary measures.

Repeated runs remain grouped by task and template. Report uncertainty and per-template outcomes; three held-out templates provide limited evidence of structural generalization. Avoid treating different seeds or wording variants as fully independent tasks. Do not claim equivalence from similar point estimates, and do not tune against held-out results.

Sample counts, selected penalty, exact executed commands, source hashes, model digests, and results will be written to the experiment summary. Any change after freezing is recorded as a new configuration rather than mixed with earlier runs.

## Exploratory v2 addendum — 13 October 2026

The text above records the original v1 design and remains the primary protocol. The active code now offers versioned v2 data and opt-in runtime revisions. V2_PROTOCOL.md is the prospective experimental workflow; v2 outputs are separate under results/v2 and must never be pooled with v1.

V2 has eight old training templates, four old validation templates, eight fresh test templates and four inventory schema-shift templates. It uses ten seeds and two wording variants; only wording 0 enters primary model experiments. Statuses, null positions, dates and participation now vary with seed. Independent record oracles are verified against reference/equivalent/wrong SQL before freezing. The three previously-scored test families are v2 training-only. Both new template lists, the corpus bytes and corpus source are hashed before inference.

--initial-model and explicit --small-model/--large-model tags define runtime roles. --fresh-escalation is for executable clean/flagged states; execution errors retain previous-SQL/error feedback. Fresh repeats from a large state require --diversify-seed. Its seed is trial+calls after increment. --neutral-feedback avoids an unsupported assertion of error. Defaults remain v1. Public ratio ranges/denominators are human-authored inputs to every baseline; gold, template IDs, split labels and database seeds remain excluded from prompts/features.

Headroom profiling is controller-free and development-only. Run a ten-task timing batch first and print the estimated full cost. The fixed pair gate is a small model correct on at least half of the seeds in 30–70% of development templates, with a stronger model having higher overall success. If the pair fails, report that and do not proceed to collection. Verifier revisions and task-type routing must be ready before the final controller freeze.

Collection must cover initial/checked/repaired stages and at least five training seeds, skipping ineligible stages. Train table and logistic_v2 with training-only nested CV before any validation run. V2 validation uses 40 tasks, not 30: four templates × ten seeds. Use the nine-point penalty/threshold grid and select_operating_point, with selection_mode and analysis source hash recorded.

The pre-registered benefit criterion is within ten percentage points of the best simple baseline at lower mean effort in each clean/transient condition, with per-template counts shown. This is descriptive, not formal non-inferiority. Write the clean code git hash, controller hashes, model digests, selected settings, protocol hash and both split hashes to FREEZE.md before the test. Run the full frozen policy schedule once, three trials, clean/transient; keep schema-shift and oracle diagnostics separate. Report detector/recovery/final-correctness populations and cluster intervals. An unfavourable outcome is a legitimate second negative result.

The researcher authorised pre-inference amendments on 13 October 2026: explicit net-value eligibility, exercised monthly/HAVING constraints, optional exact-request generation caching and development-designed boundary/cancelled-NULL/discount-irrelevance probes. V2_PROTOCOL.md records each dated amendment, accounting rules and template-level interpretation. PREREGISTRATION.json and corpus/template hashes are regenerated; earlier development profiles cannot be pooled with the revised configuration. Cache replay charges recorded model usage plus live tool/probe effort; it is not a live latency measurement. A fixed twenty-request development sample bypasses cache after final freeze, before held-out execution. Eight test families mean one whole family is 12.5 percentage points, larger than the ten-point descriptive tolerance. V1 sources/evidence/default behavior remain the primary study.

The existing mutation-family evaluation partition is retrospective because its outcomes were previously inspected. It must not be described as a new unseen holdout. This corpus/protocol phase does not alter the primary Chapter 5 findings or establish a revised controller benefit.
