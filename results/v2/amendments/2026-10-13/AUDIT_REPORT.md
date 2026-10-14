# Pre-inference amendment audit — 13 October 2026

V1 remains the primary dissertation finding. This audit makes no model calls and is separate from fresh held-out model evaluation. Rules were not changed in response to evaluation-family results. The existing design/evaluation allocation is retrospective; it is not a new unseen detector holdout.

Both amended constraints, including each monthly date endpoint independently, change the independent-oracle answer on 10/10 seeds when omitted. All 480 reference/equivalent/wrong authoring instances pass; all three verifiers leave every reference unflagged. Reference audit counts use one wording per template/seed: 240 instances and 24 distinct SQL strings. The unittest checks additionally cover both wordings.

| Partition | Verifier | Wrong queries detected | Oracle-matching queries flagged | Distinct SQL |
| --- | --- | --- | --- | ---: |
| design | original | 120/170 (70.6%) | 20/370 (5.4%) | 54 |
| design | contract_v2 | 140/170 (82.4%) | 50/370 (13.5%) | 54 |
| design | contract_v3 | 150/170 (88.2%) | 10/370 (2.7%) | 54 |
| evaluation | original | 100/298 (33.6%) | 0/162 (0.0%) | 46 |
| evaluation | contract_v2 | 100/298 (33.6%) | 0/162 (0.0%) | 46 |
| evaluation | contract_v3 | 150/298 (50.3%) | 0/162 (0.0%) | 46 |

The ten contract_v3 design false alarms are omission-of-status mutants whose outputs happen to match the oracle on their current database instances. Adding cancelled rows reveals a concern. They remain false positives under the registered output-correctness metric; this discrepancy does not justify relabelling them or claiming the probes prove SQL correctness. Every canonical reference remains unflagged.

| Family | Partition | Distinct SQL | Wrong cases | Original recall | Revised recall | Revised false alarms |
| --- | --- | ---: | ---: | --- | --- | --- |
| average_expression | design | 1 | 0 | — | — | 0/10 (0.0%) |
| conditional_count | evaluation | 2 | 0 | — | — | 0/20 (0.0%) |
| count_expression | design | 6 | 0 | — | — | 0/60 (0.0%) |
| count_nullable_column | evaluation | 4 | 38 | 10/38 (26.3%) | 10/38 (26.3%) | 0/2 (0.0%) |
| distinct_subquery | design | 1 | 0 | — | — | 0/10 (0.0%) |
| group_order | evaluation | 2 | 0 | — | — | 0/20 (0.0%) |
| join_predicate | design | 2 | 20 | 0/20 (0.0%) | 0/20 (0.0%) | — |
| missing_distinct | evaluation | 1 | 10 | 10/10 (100.0%) | 10/10 (100.0%) | — |
| null_function | design | 4 | 0 | — | — | 0/40 (0.0%) |
| numeric_bias | evaluation | 12 | 120 | 0/120 (0.0%) | 50/120 (41.7%) | — |
| omitted_discount | design | 2 | 20 | 0/20 (0.0%) | 20/20 (100.0%) | — |
| omitted_null_handling | evaluation | 2 | 20 | 20/20 (100.0%) | 20/20 (100.0%) | — |
| omitted_status_filter | design | 11 | 100 | 90/100 (90.0%) | 100/100 (100.0%) | 10/10 (100.0%) |
| ratio_scale | evaluation | 2 | 20 | 0/20 (0.0%) | 0/20 (0.0%) | — |
| reference | design | 12 | 0 | — | — | 0/120 (0.0%) |
| reversed_null_filter | evaluation | 1 | 10 | 0/10 (0.0%) | 0/10 (0.0%) | — |
| status_predicate | design | 12 | 0 | — | — | 0/120 (0.0%) |
| subquery_wrapper | evaluation | 12 | 0 | — | — | 0/120 (0.0%) |
| truncated_date_window | design | 3 | 30 | 30/30 (100.0%) | 30/30 (100.0%) | — |
| wrong_aggregation | evaluation | 8 | 80 | 60/80 (75.0%) | 60/80 (75.0%) | — |

New targeted design examples are reported separately in mutation_audit.json. They use old training/validation contracts only; no test or inventory query failures informed their design. Oracle-matching mutants are kept in the correct-output denominator even when the intervention flags them.

| Removed probe | Design detections after removal | Evaluation detections after removal |
| --- | --- | --- |
| date_boundary_records | 150/170 (88.2%) | 150/298 (50.3%) |
| net_count_boundary_records | 150/170 (88.2%) | 150/298 (50.3%) |
| cancelled_null_discount | 140/170 (82.4%) | 150/298 (50.3%) |
| discount_irrelevance | 150/170 (88.2%) | 150/298 (50.3%) |

Reference false alarms by split: train 0/80, validation 0/40, test fixtures 0/80, schema-shift fixtures 0/40 for each verifier. These are authored reference-query fixtures, not model outputs. The audit has 120 development template/seed instances, 1,000 registered variants and 100 distinct SQL strings, with additional design examples separate. Repeated seeds, wordings, SQL forms and probes are correlated; these are not independent query-family replications.

Generation caching is opt-in in Session/experiment and explicitly enabled only by the prospective workflow. Tests establish exact-body keys, identical parsed SQL, live database execution, recorded-usage replay, default-off request compatibility, consistency sample replay, corruption detection and provider-outage propagation. Twenty-request live sampling is implemented and tested with mocked providers; the real live sample, revised collection, fitted-controller freeze and held-out run are pending. Replayed charged effort must not be presented as live inference latency.

Checks: 12 pilot tests, 118 study tests (including RevisionTests), and 24 separate revision tests passed. All 22 files under results/main match their pre-amendment SHA256 values. No model was invoked in this amendment phase.
