# Task-type routing baseline — 13 October 2026

This rule is fixed before fitting the router and before any fresh held-out inference. It supplements V2_PROTOCOL.md without changing its recorded hash or success criterion.

Route before any SQL exists, using only five public contract fields: aggregation, grouped output, distinctness, date-window requirement and net-value requirement. No template identity, split, database seed, gold result or candidate SQL enters this category.

Fit empirical success and effort by category and model on the chosen pair's clean training profile only. Back off to each model's global training mean with a prior weight of two. Use training-template leave-one-out predictions to fit separate Platt calibrations for the two models; validation and both holdouts are excluded from fitting. Calibration is an estimate on eight training families, not a reliability guarantee.

Training-only revision, before any held-out inference: a non-positive fitted slope falls back to the smoothed global training success rate (successes+1)/(observations+2). It must not reverse category confidence. Preserve the unconstrained slopes as diagnostics. With one training family per category, holdout categories can disappear entirely and OOF probabilities can reflect only changing global rates. A constant fallback supports no claim of useful task-type discrimination. The first unconstrained fit is preserved separately.

Choose the cheaper model among those within ten predicted percentage points of the highest calibrated success probability. Unknown categories use the global backoff. This fixed routing rule has no validation grid. Retry transient execution timeouts; other execution errors follow the selected model's ordinary feedback-repair path. There is no semantic verifier or discretionary model switch in this baseline.

The router stores its source hash, training-profile hash and dataset hash. Evaluation must preserve its hash in the fitted-model manifest. Its measured accuracy/effort remains pending until the development collection and final experiment phases. The cost of profile-based routing training is reported separately from deployment effort.
