# Project demonstration and interview guide

Project: A Cost-Aware and Reliability-Guided Framework for Adaptive Model Routing in Tool-Using AI Agents.
Author: Puneet Devnani.

## What the project demonstrates

The project implements local language-model inference, a bounded SQLite tool, independent evaluators, observable verification, explicit retry/repair/escalation/stop decisions, small numerical success/effort predictors, grouped data splits, resumable experiments, and trace-based analysis. The scientific question is whether the learned decisions improve the correctness-effort frontier; improvement must be supported by the recorded results.

Do not describe it as a production-ready autonomous agent or claim a cost-saving percentage that is not supported at comparable task success. A system that spends less by returning more wrong answers has not solved the reliability objective.

## Five minute demonstration

1. Explain the user request and public task contract. Show that reference answers are absent from controller input.
2. Run a local large_net task with the stronger-model policy. Show the model's SQL, tool response, chosen actions, and final report.
3. Repeat with a transient timeout. Explain why the failed operation can be retried without another model generation.
4. Replay an explicitly labelled held-out trace with a semantic failure. Show why successful execution and passed checks do not prove correctness.
5. Open the saved experimental summary and compare success with measured deployment effort. Explain a negative finding and the experiment that would test an improvement.

Commands from the project root, after starting Ollama and installing numpy:

```sh
python -m study.demo --template large_net --seed 1 --policy large
python -m study.demo --template large_net --seed 1 --policy large --fault transient
python -m study.demo --template january --policy learned --replay
```

Replay reads a recorded run. It does not invoke a model or measure new latency. The live demo appends a separate result under results/demo and does not alter the main test set.

## Suggested project description

Built a reproducible local framework for verification and recovery in tool-using language-model agents. Implemented bounded SQL execution, independent task scoring, executed recovery branches, lightweight success/effort predictors, and repeated comparisons with fixed-model and rule-based policies. Documented semantic detector failures and the limits of learned routing under constrained hardware.

Add measured result numbers only after checking results/main/summary.json. Include the model pair, hardware, and task conditions alongside any metric.

## Questions to prepare for

- Why does a cheaper model not always yield a cheaper successful task?
- What information can the controller observe before making a decision?
- Why is a verifier result evidence rather than ground truth?
- How were untried actions labelled? Explain the executed-branch method.
- What differs between the training continuation and deployed decisions?
- Why are all variants of a template kept within one data split?
- Why does the model-only restriction differ from a retrained ablation?
- Why can a static rule beat a learned controller on a small dataset?
- What makes the timings hardware-specific?
- What did the failed experiments teach you, and how would you design a fresh evaluation?

## Portfolio contents

Include the source code, small fitted controller, corpus and split definitions, exact experiment commands, selected traces, final summary/figures. Model weights and third-party literature PDFs are retained locally and should not be copied into a public repository. Download models from their official distribution and comply with their licences.

The supervisor certificate remains unsigned. Do not present a signature or institutional approval until it has actually been provided.
