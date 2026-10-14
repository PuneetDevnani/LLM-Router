# Recovery-Aware Model Routing for Tool-Using AI Agents under Compute Budgets

Status: research blueprint, not a report of completed work.
Prepared: 12 October 2026.

Progress: local baseline runner, 24-task feasibility dataset, bounded SQL tool, and six infrastructure tests implemented. Initial actual-model runs completed with 12/24 correct for Qwen3-0.6B and 24/24 for Qwen3-1.7B. A four-task timeout-recovery check completed. See PILOT_FINDINGS.md for evidence and limitations. A targeted full-text comparison of 12 papers is now available in literature/LITERATURE_COMPARISON.md. The learned router and main held-out experiments remain outstanding.

Scope revision: RESEARCH_PROPOSAL.md is the current detailed design and supersedes overlapping planning suggestions below, which are retained as the initial scaffold. The candidate contribution now concerns joint verification/recovery selection using observable evidence and predicted remaining effort. Model routing, fallback, cost inversions, and gold-free semantic checking already have prior work. The targeted review does not certify novelty.

Second feasibility milestone: a 72-task development suite across 12 templates and two related tables, two imperfect semantic invariance checks, and rule-based semantic escalation are implemented. Twelve infrastructure tests pass. See RICH_TASKS.md for the structured-input condition and checker limitations. No learned controller has been trained.

## Scope and research question

Develop a local data-analysis agent that selects between two open-weight language models and chooses a recovery response when an action fails. No paid APIs are required. Language-model training is outside the initial scope; the learned component is a small routing/recovery predictor.

Research question: Does accounting for expected recovery effort improve the task-success versus compute trade-off relative to selecting models using immediate inference effort alone?

Secondary questions:
- Which observable failure categories benefit from model escalation, argument repair, or a tool retry?
- How much of the apparent saving from a smaller model is consumed by validation, retries, and model switching?
- Does the policy transfer to held-out task templates and changed tool-failure rates?

Novelty is provisional. Step-level model routing, cost-aware reliability, and self-healing tool routing already exist. The proposed distinction is an empirically learned estimate of remaining recovery effort in stateful tool-use workflows. A full-text comparison must establish whether this distinction is new and substantive before the final contribution is claimed.

## Application

Use a sandboxed data-analysis environment with fixed CSV files and a SQLite database. Start with tools for inspecting table schemas, executing read-only SQL, performing bounded calculations, and saving structured reports to an isolated output directory. Avoid unrestricted code execution in the first version.

Example: Calculate monthly revenue excluding cancelled orders, identify the highest-growth category, and save a report with the underlying aggregates.

Task families: aggregation, joins and filtering, missing-value handling, date calculations, and multi-step report generation. Each task has an evaluator checking required values, constraints, and output artifacts. Equivalent valid tool sequences must be accepted.

## Proposed method

At each decision point, observe the user request, execution history, tool schemas, recent error messages, remaining budget, context length, and progress indicators.

Before a generation, select a model. After validation or execution indicates failure, select an eligible recovery response: regenerate arguments, retry a transient tool operation, escalate the model, or stop explicitly. An alternative-tool response is an extension only if an equivalent tool exists.

Train lightweight predictors for completion probability and remaining inference effort for candidate decisions from pilot trajectories and controlled branches. Start with transparent features and simple logistic/regression models. Compare against hand-written rules before considering more complex learners.

Choose a decision using predicted completion probability minus a tunable penalty for predicted remaining inference effort. Tune the penalty and decision thresholds on validation tasks; report multiple operating points rather than only one favourable setting. Treat probability estimates as empirical predictions, not reliability guarantees.

Estimated effort must include expected future repairs and retries. Measured end-to-end latency additionally includes tool time, routing, validation, and model loading. Keep these measurements distinct to avoid counting the same time twice.

The policy cannot see the fault injector's hidden category, expected answer, or final evaluator result at inference time. Validation success is a signal, not proof of semantic correctness.

## Hardware plan

Confirmed laptop: MacBook Air M2 with 8 GB unified memory. Start the memory and tool-use pilot with 4-bit quantized Qwen3-0.6B and Qwen3-1.7B, loading one model at a time. The 0.6B model may be too weak for useful tool execution; this must be checked rather than assumed. Qwen3-4B is an optional stronger candidate after memory profiling locally or in Colab. Pin exact model revisions, quantization, context limits, prompts, and inference settings.

Run development and the demo on the MacBook Air M2. Start with a short context budget (approximately 2,048 tokens) and bounded output lengths, then adjust after profiling. Account for model weights, context cache, runtime overhead, and memory used by macOS and other applications; weight size alone does not establish fit. Avoid sustained memory swapping during benchmark runs. Use free Colab for supplementary inference runs when available, with resumable checkpoints. The submission deadline remains pending user confirmation.

Run primary latency comparisons on a consistent hardware/backend configuration. If models cannot remain resident together, include loading/switching overhead in end-to-end measurements. Report cold-start and warm execution separately. Separate M2 and Colab results.

## Pilot and continuation criteria

1. Create 20-30 objectively scored tasks spanning the initial task families.
2. Run each model alone, then a simple small-first escalation strategy.
3. Inspect representative successful and failed traces.
4. Measure runtime, memory, tool-call validity, and final task success.
5. Continue if there is useful variation in model capability, observable recoverable failures, and a manageable runtime. Revise tasks or models if both mostly fail or one dominates without a meaningful compute trade-off.

The pilot is a feasibility study, not evidence of publication-level performance.

## Main experiment

Provisional target: 100-150 tasks, scaled after profiling the pilot. Split by underlying task template into training, validation, and test partitions. Related paraphrases and database variants remain in the same partition. Three repetitions per condition are an initial target, subject to compute availability and measured variance.

Conditions:
- Clean tools with naturally occurring model errors.
- Controlled transient tool failures.
- Persistent tool failures where a valid completion may be impossible.
- Malformed or incomplete tool responses.
- Held-out combinations or rates of failures.

Invalid arguments generated naturally by a model should be reported separately from deliberately corrupted actions. Do not conflate infrastructure faults and model reasoning errors.

Use reproducible fault schedules keyed to tool/operation and seed. Log which faults were actually encountered because different policies may follow different trajectories. Score ordinary completion, recovery on recoverable cases, and explicit stopping on impossible cases separately.

Baselines:
1. Smaller model only.
2. Stronger model only.
3. Smaller-first with a fixed bounded escalation rule.
4. Failure-specific hand-written recovery rules.
5. Learned routing based on immediate inference effort, with the same recovery machinery.
6. Proposed recovery-aware policy.

Keep prompts, tool schemas, task limits, and access to recovery options comparable. Add literature baselines where compatible and feasible; explain adaptations.

Ablations: remove predicted recovery effort; remove execution-history features; remove failure-specific recovery choices. Do not turn these into separate unrelated systems.

Metrics:
- Final task success with task-level uncertainty intervals.
- Total model inference time and end-to-end latency, including failed tasks.
- Median and upper-percentile latency.
- Token counts, peak memory, model switches, and recovery attempts.
- Completion after detected recoverable failure.
- Compute spent on unsuccessful runs.
- Calibration of predicted completion probabilities.

Use paired task comparisons and bootstrap over tasks, keeping repeated runs grouped by task. Examine success versus compute curves. Define any acceptable success loss before examining final test results. Monetary savings and energy savings are not measured claims in this project.

## Data collection cautions

Only the selected decision's outcome is observed in an ordinary trajectory. Do not assign outcomes to untried models or recovery options. To compare candidate choices, branch from saved environment states on a controlled subset, then execute the branches to completion. Count this collection workload separately from deployment inference effort.

Cached trajectories can support training and debugging. Final policy evaluation must execute fresh full trajectories; replaying cached answers does not measure adaptive-agent success or latency.

## Deliverables

- Research report following the supplied institutional formatting guidelines.
- Research-paper manuscript with verified related work and actual experimental results.
- Reproducible source repository with installation instructions, pinned configurations, benchmark runner, and tests for tool/evaluator behaviour.
- Task dataset, fault definitions, execution logs, and analysis notebooks where redistribution permits.
- Local demo showing model choice, tool actions, failures, recovery, and final outcome.
- Short demonstration video and an evidence-based portfolio description.

## Work sequence

1. Literature comparison and hardware/tool-use pilot.
2. Task environment, evaluators, and baseline agents.
3. Trace collection and lightweight predictors.
4. Recovery-aware policy and validation tuning.
5. Fresh held-out experiments, ablations, and error analysis.
6. Report, paper, reproducibility check, and portfolio demo.

Calendar dates will be set after the submission deadline is known. Prioritize the literature review and pilot before writing a large amount of project documentation.

## Starting references

These sources identify overlap; listing them does not establish novelty or independent replication of their findings.

- RouteLLM: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html
- AgentRouter: https://arxiv.org/abs/2609.22951
- Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents: https://arxiv.org/abs/2603.01548
- A Communication-Theoretic Framework for LLM Agents: Cost-Aware Adaptive Reliability: https://arxiv.org/abs/2605.09121
- RACER: https://proceedings.mlr.press/v306/hao26c.html
- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3-4B model card: https://huggingface.co/Qwen/Qwen3-4B
- Colab resource limitations: https://research.google.com/colaboratory/faq.html

## Literature review acceptance check

For each close paper, record its decision unit, observable inputs, model/recovery choices, optimization objective, actual cost accounting, datasets, baselines, limitations, and released artifacts. Read the method and experiments, not just the abstract. If existing work already matches the proposed method, revise the contribution before the main implementation.
