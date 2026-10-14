# Adaptive Model Routing with Verification and Recovery

The project is titled **A Cost-Aware and Reliability-Guided Framework for Adaptive Model Routing in Tool-Using AI Agents**. It develops a local SQL-analysis agent and investigates whether a learned controller can select useful verification and recovery actions while balancing correctness and execution effort. The implementation and experiments are complete. The measured controller did not achieve the intended improvement: the stronger-model baseline obtained 18 correct runs out of 27 in each held-out tool condition, while the learned joint controller obtained zero.


## Problem and objective

An AI agent must do more than generate plausible text. When answering a database question, it must produce a query that represents the request, execute that query, handle failures, and return the required report. A query can run without errors yet compute the wrong value. A correct query can also fail because the tool temporarily times out. These cases need different recovery actions.

Always invoking a stronger model can consume more resources. Starting with a smaller model can be cheaper for an individual call but become expensive if repeated repair is required. Verification also consumes effort and can miss errors. The research question is whether observable execution evidence can support better decisions about accepting, checking, repairing, escalating, retrying, or stopping.

The implemented scope is a post-generation controller for structured SQL tasks. For learned policies, the small model always generates the initial query. The controller then chooses eligible actions. General-purpose routing before generation and automatic extraction of task contracts are outside the completed study.

## Technology and environment

The project uses Python, SQLite, NumPy, Ollama, Qwen3-0.6B, and Qwen3-1.7B. The language models run locally; no paid inference API is required. The main experiments used an Apple M2 laptop with 8 GB unified memory. Matplotlib produces measured-result plots. A standalone HTML dashboard displays stored results and execution traces.

The models are existing open-weight models. Their weights are not trained or fine-tuned. The learned components are small numerical predictors for action outcomes and remaining effort.

## Dataset and inputs

The synthetic sales database contains customers and orders, including customer regions, order dates, status, amounts, and nullable discounts. The corpus contains 240 instances: twelve task templates, ten database seeds, and two wording variants. Tasks cover totals, date filters, grouped joins, averages, distinct counts, missing values, percentages, thresholds, and combined summaries.

Templates are grouped into six training, three validation, and three test families. Related variants remain in the same split. The primary held-out experiment uses nine unique task instances from three test families, with three repeated trials per instance. Consequently, 27 runs per condition are repeated measurements over nine tasks, not 27 independent task families.

Every policy receives the same request, database schema, and human-authored structured contract. The contract specifies the required operation, grouping, null handling, dates, and output shape. Reference answers, task identifiers, template labels, split labels, and database seeds are excluded from live model requests and controller features.

## Runtime workflow

1. Receive the public request, schema, and structured contract.
2. Generate read-only SQLite SQL using the initial model.
3. Execute the query through a bounded tool and record its output or error.
4. Extract observable features from the request, SQL, result, checks, model identity, and remaining budget.
5. Use the policy to select an eligible action.
6. Execute verification, repair, escalation, or retry and return to the decision loop, or terminate through acceptance or stopping.
7. Format the final result deterministically and evaluate correctness independently after termination.
8. Save the complete execution trace and resource measurements.

The deterministic formatter copies scalar or grouped query results into the declared report shape. It rejects invalid shapes but cannot repair a semantically incorrect query.

## Controller actions

| Action | Behaviour |
| --- | --- |
| Accept | Terminate with the executable, formattable current result. |
| Verify | Collect additional evidence about the unchanged candidate. |
| Repair | Ask the small model to generate a corrected query. |
| Escalate | Ask the stronger model to generate a new query. |
| Retry | Repeat the existing query after a transient tool failure. |
| Stop | Terminate with an explicitly incomplete outcome. |

Each trajectory allows at most three model calls, six main-tool attempts, and five controller decisions. Verification probes are separately counted and timed. Eligibility is recalculated after state changes. A new query invalidates evidence collected for the previous query.

## Verification and independent scoring

Verification includes fallible lexical checks and database invariance probes. For example, adding cancelled orders should not change an answer that explicitly excludes cancelled orders. Replacing missing discounts with zero should preserve a result when the task defines that convention.

These checks provide evidence, not proof. Two different queries can preserve the same invariance while only one answers the request correctly. Passing a check or recovering from a timeout is therefore distinct from correct task completion.

Reference answers are calculated independently from Python records. The evaluator compares the final report with these answers only after termination, using declared report keys and numeric tolerances. Offline training can use correctness labels; deployment cannot see them.

## Learning and action selection

Collection produced 72 saved states and 292 actually executed action branches. Eligible actions were explored from copies of saved states, with a fixed bounded recovery continuation. Each branch supplied terminal correctness and remaining measured effort. Only 26 branches terminated correctly, leaving little positive training evidence.

The controller uses 22 observable features with action-specific interactions. An L2-regularized logistic regression estimates eventual correctness; ridge regression on log-transformed duration estimates remaining effort. Features are standardized using training data only.

The action score is:

**predicted correctness − lambda × predicted remaining effort / reference effort**

The controller selects the eligible action with the highest score. Validation compared penalties 0, 0.05, 0.15, and 0.3 and selected 0.3 under the predefined fallback rule. No candidate met the intended tolerance relative to stronger-model validation success. These estimates are empirical approximations under the collection continuation and do not supply calibration or optimal-policy guarantees.

## Experiments and results

Five primary policies were compared: small only, stronger only, recovery rules, the joint learned controller, and a restriction of that controller without verification. Conditions were clean tools and a timeout injected at the first main-tool operation. There are 270 primary trajectories, 63 validation trajectories, and 18 supplementary sensitivity trajectories. Persistent-outage termination is tested separately.

| Policy | Clean correctness | Clean mean effort | Timeout correctness | Timeout mean effort |
| --- | --- | --- | --- | --- |
| Small only | 0/27 | 2.17 s | 0/27 | 2.24 s |
| Stronger only | 18/27 | 10.65 s | 18/27 | 10.00 s |
| Recovery rules | 0/27 | 9.75 s | 0/27 | 8.08 s |
| Joint controller | 0/27 | 1.77 s | 0/27 | 1.55 s |
| Model-only restriction | 0/27 | 1.44 s | 0/27 | 1.59 s |

Effort adds recorded model loading, prompt processing, generation, and measured tool/check durations. End-to-end wall time is recorded separately. Seconds on this machine are not monetary cost, energy, or FLOPs.

The stronger model solved the January-sum and large-net-count families but failed the grouped regional-net family. The joint controller selected 36 acceptances and 18 stops across its 54 primary trajectories, with 27 additional timeout retries. It selected no verification, repair, or escalation in those trajectories.

## Failure interpretation and contribution

The controller spent less effort because it terminated with incorrect or incomplete outcomes. This does not establish successful cost reduction. One recorded query excluded missing discounts despite the contract treating them as zero, returning 19 instead of 25. Acceptance narrowly received the highest score despite a predicted success probability of approximately 0.012. The effort penalty made alternative recovery scores unattractive. Removing the penalty in a small sensitivity test still produced no correct reports, so the selected penalty alone does not explain the failure.

The verifier also missed all sixteen executable wrong results examined in its clean training-state diagnostic. This is a development diagnostic, not a comprehensive held-out verification benchmark. Limited model capability, scarce successful branches, imperfect checks, and differences between training continuation and deployed decisions constrain the findings.

The contribution is a reproducible local system, controlled comparisons, explicit separation of detection/recovery/final correctness, full execution traces, and evidence explaining the failed trade-off. The study does not establish a superior router, new constituent routing technique, monetary savings, or a general reliability guarantee.

## Literature and deliverables

The updated bibliography contains 24 entries: the original twelve closely related research papers and two software/model references, plus ten additional works. The additions include four Springer journal papers, a Nature paper on semantic entropy, ReAct, Toolformer, Reflexion, a cost-efficient cascade paper, and Spider. The new works are cited in the literature discussion. Spider and the cited uncertainty/agent methods were not substituted into the experiments.

Deliverables include source code, corpus and split definitions, measured-result plots, an offline dashboard, a live/replay CLI, and a reproducibility guide. Infrastructure tests validate engineering behaviour rather than model accuracy.

## Future work and submission

Future studies can improve semantic checks, collect more diverse states and successful recovery examples, introduce a validated acceptance threshold, test stronger model pools, and evaluate additional schemas and public benchmarks. Any controller revisions require fresh validation and held-out data rather than repeated optimization against the existing test set.

Supervisor review and certificate signatures, designations, and dates remain necessary before submission. For a viva, explain the architecture, inspect a saved incorrect query, demonstrate direct timeout retry, compare correctness with effort, and state the negative result clearly.
