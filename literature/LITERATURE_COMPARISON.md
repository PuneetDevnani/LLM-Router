# Literature comparison and novelty audit

Reviewed on 12 October 2026. This is a targeted review of 12 close primary papers, not an exhaustive systematic review. Method, evaluation, and relevant limitation sections were inspected from full texts. Reported findings are the authors' claims, not independently reproduced results. Local PDFs and extracted text are retained in `sources/`; the source manifest records download URLs and file hashes.

## Decision

Retain the research area, but revise the proposed contribution. Neither step-level routing, fallback, verification-guided escalation, nor the observation that cheap models can become expensive through repeated errors is new. SQL semantic checking without a gold answer and reducing verification calls through early exit also have prior work.

Candidate contribution: a lightweight, state-conditioned controller that jointly chooses an additional verification operation or a recovery action, using predictions of final completion probability and remaining measured deployment effort. Evaluate this specifically on multi-step data-analysis agents with silent semantic errors, explicit tool failures, and imperfect verifiers. This combination is a research hypothesis; this review does not certify priority or publishability.

## Comparison matrix

| Work | Decision and evidence | Cost/evaluation focus | Consequence for our project |
|---|---|---|---|
| [1] RouteLLM | Pre-generation strong/weak choice from prompt preferences | Strong-model use and benchmark quality | A learned cheap/strong router is established |
| [2] FrugalGPT | Score candidate responses and continue a model cascade | Cumulative API charges under a budget | Cheap-first escalation and response scorers are established |
| [3] AutoMix | Noisy self-verification informs accept/escalate decisions | Generation plus verification cost | Verification-guided escalation is established |
| [4] AgentRouter | Step classifier using task/context features; end-of-trajectory fallback | Completion quality and API spend | Step routing with fallback is established |
| [5] BAAR | History-conditioned binary model choices; SFT and policy optimization | Sequential success-cost frontier and budgets | Future trajectory cost and cost inversions are established concerns |
| [6] Self-Healing Router | Health signals reweight a manually constructed tool graph | Mock control-flow correctness and LLM-call counts | Tool recovery without a model call is established |
| [7] AgentCodec | Select reliability techniques from estimated quality/cost | Curated and standard reasoning/code tasks | Cost-aware selection of reliability techniques is established |
| [8] RACER | Calibrate model subsets using router scores | Misrouting risk versus number of invoked models | Risk calibration is established; guarantees require assumptions |
| [9] xRouter | Learned LM delegates to external models or answers | Reward uses success and cumulative invocation cost | Learned cost-aware orchestration is established |
| [10] SQLHD | Two-stage metamorphic SQL error detection | Detection metrics and verifier-call overhead | Gold-free checking and heuristic early exit are established |
| [11] MRSQLGen | Select transformations from error patterns and compare results | SQL error detection, token usage and runtime | Targeted semantic checks are established |
| [12] ReflexGrad | Progress signals gate local refinement versus causal diagnosis | In-episode recovery and additional model calls | History-guided recovery is established |

## Evidence notes

### [1] RouteLLM — Ong et al.

The reviewed manuscript learns preference-based routing between strong and weak models. It studies similarity-weighted ranking, matrix factorization, BERT, and causal-LM classifiers; evaluation includes MMLU, GSM8K, and MT-Bench. The principal decision chooses one model before generating an answer. It does not provide the proposed menu of tool retry, semantic check, repair, and termination. Compare a locally trained prompt router, clearly identified as an adaptation rather than a reproduction of the pretrained system. Evidence: Sections 3–5, especially Section 4.2. [Primary manuscript](https://arxiv.org/pdf/2406.18665).

### [2] FrugalGPT — Chen, Zaharia, and Zou

The cascade uses a learned correctness scorer and thresholds to accept an answer or call another model. Its budget objective includes charges accumulated across calls. Experiments cover HEADLINES, OVERRULING, and CoQA. This already combines reliability estimation and cost-sensitive escalation; adding a small classifier is insufficient novelty. Our comparison should include a calibrated fixed cascade with the same model pool and available verifier operations. Evidence: Section 3, cascade formulation; Section 4, experimental setup. [Primary manuscript](https://arxiv.org/pdf/2305.05176).

### [3] AutoMix — Aggarwal et al.

AutoMix generates with a smaller model, samples context-grounded self-verification, and uses thresholds or a POMDP meta-router to retain the answer or escalate. Its accounting explicitly includes verification. The reviewed version evaluates context-grounded tasks and discusses limitations when verifier information is weak. It does not support claiming that noisy verification or non-LLM routing is our invention. Test a self-verification threshold baseline and count all verification calls. Evidence: Sections 4.1–4.2, 5.2, and Limitations. [Reviewed v4](https://arxiv.org/pdf/2310.12963v4).

### [4] AgentRouter — Paul and Nandy

The step router uses task type, reasoning depth, tool requirements, output format, and accumulated context. Its algorithm includes frontier fallback after trajectory evaluation. The text says evaluation uses held-out ground truth for the fallback signal, with a judge substituted in production. This distinction motivates measuring practical detector errors, but does not prove the work lacks production validation. Claimed evaluation spans 2,400 held-out trajectories. Verify its publication status separately; manuscript formatting alone is insufficient. Evidence: Section 3, Algorithm 1, Section 4. [Reviewed v1](https://arxiv.org/pdf/2609.22951v1).

### [5] Budget-Aware Agentic Routing (BAAR) — Zhang et al.

BAAR treats routing as a sequential decision using interaction history, selects small or large models, and uses boundary-guided SFT followed by policy optimization. Experiments cover SciWorld, ALFWorld, and AppWorld. Appendix F explicitly discusses cost inversions from weak-model loops or lengthy incorrect reasoning. Therefore accounting for future mistakes cannot be the sole contribution. Our narrower difference is an explicit verification/recovery action menu and low-cost offline predictors, tested against model-only routing. Evidence: Sections 3–5, Appendices A, E, F. [Reviewed v1](https://arxiv.org/pdf/2602.21227v1).

### [6] Graph-Based Self-Healing Tool Routing — Bholani

Runtime monitors alter tool-graph weights; shortest-path rerouting handles outages and escalates when alternatives disappear. All 19 evaluation scenarios use mock tools and a deterministic mock LLM. The paper explicitly distinguishes control-flow validation from semantic correctness and lists perfect module accuracy as a limitation. Our study can examine actual model outputs and noisy checks, but real inference by itself is not a new algorithm. A bounded direct tool retry must be a baseline. Evidence: Sections 2, 4.1, 5.4. [Reviewed v1](https://arxiv.org/pdf/2603.01548v1).

### [7] AgentCodec — Omidvar and Akhlaghi

The framework implements multiple reliability mechanisms and a semantic nearest-neighbor router selecting a technique from estimated quality and cost. The reviewed evaluation includes 69 curated tasks and 300 hard benchmark items, with task-grouped repeated trials. Candidate techniques include refinement and redundant verification. Thus choosing among reliability techniques with a cost penalty is established. Our proposed setting makes decisions during tool execution and measures the consequences of detected and undetected semantic failures. Evidence: Sections 4–5, particularly 5.4; Appendices D and E. [Reviewed v1](https://arxiv.org/pdf/2605.09121v1).

### [8] RACER — Hao et al.

RACER calibrates model sets to control the risk of excluding all correct models while minimizing selected-set size. It evaluates several knowledge/reasoning benchmarks; an appendix reports parallel-generation latency. Its guarantees rely on calibration assumptions, including exchangeability, and concern model coverage rather than guaranteed final-answer correctness. We should report empirical completion-risk calibration without claiming a distribution-free guarantee. Number of model calls is insufficient as our sole local compute metric. Evidence: Sections 2, 4–5, Limitations, Appendix D.3. [Official paper](https://raw.githubusercontent.com/mlresearch/v306/main/assets/hao26c/hao26c.pdf).

### [9] xRouter — Qian et al.

A trained router LM answers or delegates through tool calls; the engine handles retries, caching, and response validation. Its reward uses success and total invocation cost, and multi-turn accounting discourages immediate savings that harm later decisions. The reported setup trains a 7B router and limits interactions to three turns. This is beyond our training budget. Use a lightweight, model-only learned baseline and describe the architectural comparison without claiming exact replication. Evidence: Sections 3–5, particularly 4.1–4.2. [Reviewed v1](https://arxiv.org/pdf/2510.08439v1).

### [10] SQLHD — Yang et al.

SQLHD checks schema linking and logical synthesis through metamorphic relations without requiring reference SQL at inference. It evaluates derived Spider/BIRD datasets using GLM-4 and ChatGPT. Section 6.2 discusses verifier overhead and a heuristic that exits early to reduce calls. Thus both gold-free verification and early-exit verification have prior art. Detector outputs should be treated as fallible evidence. Publication metadata in this preprint contains placeholders; cite the arXiv version until authoritative bibliographic details are checked. Evidence: Sections 3, 4.2, 6.2. [Reviewed v1](https://arxiv.org/pdf/2512.22250v1).

### [11] MRSQLGen — Lin et al.

MRSQLGen retrieves likely error patterns, generates targeted metamorphic prompts, and checks result relationships using majority violations. It evaluates Spider/BIRD with multiple models, reports token/runtime overhead, and analyzes incorrect rule selection and detection failures. Selecting relevant checks from query characteristics is therefore established. Our candidate difference must concern learned allocation between checking and correction under a shared remaining budget, with final completion as the outcome. Adapted check baselines must identify implementation differences. Evidence: Sections 3, 5–8. [Author-hosted FSE paper](https://matafeiyanll.github.io/paper/FSE26-MRSQLGen.pdf).

### [12] ReflexGrad — Kadu and Krishnan

The reviewed version uses consecutive low progress scores to trigger causal diagnosis, otherwise applying fast textual refinement, with a cooldown after slow activation. It evaluates actual agent recovery and analyzes erroneous evaluator signals. Its cost discussion counts extra model calls; ALFWorld is the principal validated environment. This prevents claiming that trace-aware within-episode recovery is new. Compare a progress/stagnation-triggered recovery rule, distinguishing an inexpensive observable rule from the paper's LM-based evaluator. Evidence: Sections 3.1–3.3, 6.6–8. [Reviewed v4](https://arxiv.org/pdf/2511.14584v4).

## Claims to avoid

- First adaptive model router for tool-using agents.
- First router with fallback or verification.
- First observation that cheap models can incur greater total cost.
- First SQL checker that does not need correct reference answers.
- Guaranteed reliability merely because probabilities are calibrated empirically.
- New algorithm solely because models are local or smaller.

## What remains uncertain

This review has not established that the exact joint controller is absent from all related work. Citation chasing should next inspect unified routing/cascading, adaptive verification, learned repair selection, and budgeted test-time computation. A final paper must update this search near submission. If a prior method matches the proposed controller, present an empirical study or revise the method rather than assert algorithmic novelty.

## Acceptance gate

Proceed to the main experiment only if (a) the method has a precise distinction from the close papers above, (b) pilot branch data show that different actions are useful in different observable states, and (c) a learned controller has a plausible advantage over calibrated task-type and failure-specific rules. Otherwise simplify or reposition the contribution.
