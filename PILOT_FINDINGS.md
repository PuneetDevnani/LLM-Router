# Initial local pilot findings

Date: 12 October 2026.

## What was executed

Actual local inference on the user's MacBook Air M2 (8 GB unified memory), using Ollama 0.40.3 and its downloaded quantized Qwen3 models. No paid API was used. Model details and digests are saved in the run records.

The pilot has 24 tasks: six database seeds for each of four templates. Both models received the same documented schema and prompt. This is a feasibility sample, with one trial per task, no held-out evaluation, and no trained router.

| Model | Correct tasks | Median end-to-end task time | Total inference time | Total model-loading time |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 12/24 | 2.50 seconds | 22.04 seconds | 40.23 seconds |
| Qwen3-1.7B | 24/24 | 4.39 seconds | 42.74 seconds | 62.53 seconds |

Times come from the saved executions. Inference time sums Ollama prompt evaluation and output generation durations. Loading time sums Ollama load durations. End-to-end time includes HTTP and tool overhead. Models were unloaded after every invocation, so these are conservative cold-execution measurements, not optimized serving benchmarks. The small-model run overlapped with downloading the larger model; rerun comparisons without background downloads before making latency claims. Peak memory and swapping were not measured.

## Observed failure pattern

The smaller model passed every total-revenue and count task. It failed every monthly and category aggregation task. The stronger model passed all four task families in this sample.

Importantly, neither clean run logged SQL execution errors. Incorrect model queries could execute successfully and produce well-formed reports. Therefore execution-error detection alone would not trigger escalation in these cases. The final evaluator knows the correct answer; the agent does not receive that answer during a run.

This motivates investigating observable semantic checks, such as checking requested filters or aggregation requirements, without relying on hidden expected answers. Whether these checks generalize and whether a learned recovery policy improves on simple rules remain open research questions.

## Prompt development

An earlier four-task smoke run with less schema-value documentation produced 0/4 correct answers. The shared prompt was then updated to state actual categorical values, case sensitivity, date format, and general instructions to honour task filters and aggregation requests. A separate four-task smoke run achieved 2/4, followed by the 24-task runs above. Preserve all runs; do not combine the old and new prompt conditions.

The original smoke-run system prompt was:

```text
You are a data-analysis agent. Database schema: orders(id INTEGER, month TEXT, category TEXT, status TEXT, revenue INTEGER). Revenue is an integer in rupees. You can issue read-only SQL.
Return exactly one JSON action per turn:
{"action":"query","sql":"SELECT ..."}
or {"action":"finish","report":{...}}.
Use the query tool before finishing. Observe tool results and recover from errors. Your final report must match the requested structure. Do not explain outside JSON. /no_think
```

Subsequent runs store their system prompt and runtime metadata directly in each result record.

## Infrastructure validation

Six automated checks pass: independent expected-answer validation across all tasks, rejected SQL writes/unsafe operations, rejected incorrect/nonfinite reports, tool retry without extra generation, persistent-outage failure scoring, and malformed-action escalation. These use mocked responses where appropriate and are not model-quality evidence.

An additional actual-model run used the failure-specific-rule baseline on four tasks with a timeout injected at the first query. All four timed-out queries were retried without another model generation and returned tool results. Final correctness remained 2/4: infrastructure recovery did not fix the smaller model's semantic mistakes. This is a functional recovery check, not a statistically supported comparison between recovery policies.

## Saved evidence

- `results/small-clean.jsonl`: original four-task smoke run.
- `results/small-clean-v2.jsonl`: documented-schema smoke run.
- `results/small-clean-24.jsonl`: full small-model pilot.
- `results/large-clean-24.jsonl`: full stronger-model pilot.
- `results/rules-transient-4.jsonl`: actual-model transient-tool-retry check.

Summarize locally with:

```sh
python3 pilot/summarize.py results/small-clean-24.jsonl results/large-clean-24.jsonl
```

## Decision

Continue with this model pair for prototype development. Expand task diversity before main experiments; template identity almost completely explains the current success difference, so a fixed task-type rule may perform as well as a learned policy here. Complete the full-text novelty comparison before claiming a new method. Do not use this pilot's results as proof of recovery-aware routing improvements or publishable generalization.
