# Controller acceptance gating revision

On 13 October 2026, Controller.choose was extended with accept_threshold, wrong_penalty, and abstain_cost using the supplied action-selection rule. Defaults are zero and preserve the original behaviour. The existing experiment and demo callers still use these defaults; nonzero settings must be explicitly passed to choose.

When predicted acceptance success is below the threshold and recovery is eligible, both accept and stop are excluded. The controller must choose repair, escalation, verification, or retry. With no eligible recovery it may still accept below the threshold. This is a recovery trigger, not a strict minimum-confidence guarantee. The unused terminal set from the supplied snippet was omitted; retry is a non-terminal tool operation.

The original benchmark controller source is preserved at results/main/frozen_source/controller.py. Saved trajectories, fitted coefficients, summaries still describe the original measured study. No new inference experiment or performance improvement is claimed. The benchmark source audit will detect the development changes to controller.py and test_study.py; do not rewrite the original provenance hashes to conceal them. Retain the original study and use a separate output directory and fresh held-out tasks for revised experiments.

Validation must select the settings before fresh testing. A penalty for wrong acceptance changes the deployment objective and should be supported by the evaluation metric. Forced recovery can consume the remaining budget without increasing correctness; the numerical success predictor has not been calibrated by this revision.

Verification: 14 study tests and 12 pilot tests pass. Seven added selection tests cover default compatibility, forced recovery, threshold equality, exhausted-recovery fallback, terminal penalties, model-only filtering, no-effort behaviour, and retry eligibility. These are controller/infrastructure checks, not evidence of improved model accuracy.

Thresholds are now plumbed through the experiment and demo CLIs, with pairwise validation selection and exploratory comparisons. See REVISED_STUDY.md for the current workflow.
