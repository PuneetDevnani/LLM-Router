# Dataset freeze — 13 October 2026

Code commit: c328a3843dfcc91695024d668bf433cf25f9e194

Corpus: 480 author-validated instances; 8 old training, 4 old validation, 8 fresh order-test and 4 inventory schema-shift templates. Ten seeds and two wordings per template. Primary experiments use wording 0; validation has 40 tasks.

- Dataset SHA256: 2786a9b8bba31ad386885d1d97d6cafc5f9b5705bafb09493550a3d2161ca456
- Corpus source SHA256: 5375e19f964b4b28691b8cfe6fdbdae71fc6700129bddaa5626a7cbf8cc61993
- Fresh test-template SHA256: 1b016fd37f8f519d482579aa32e8fa7cd3c685475869b9b0e8fd5752c7d7d8f2
- Inventory-template SHA256: 7d0e74a33b9963af2cfe9bd8e51da2adce8b7e715d927a3719e424f7bc414e05
- Prospective protocol SHA256, recorded before inference: f2734a609b8aa113b3a6ff478de256db509159c93f7f911ba11a35cf60d7d8d9
- Task-routing rule SHA256, recorded before fitting: b56882be27b49ea1271595879149d2587bc5eb5245455d4d33908d9347f5aeb6

Reference, equivalent and wrong SQL authoring fixtures passed for all 480 instances. Both verifiers accepted all 120 training/validation reference-query states. No fresh held-out model-generated query or model score was inspected. V1 remains primary and unchanged.

This is a dataset freeze, not the final fitted-controller FREEZE. Collection, controller training, validation selection and the final held-out runner remain pending. Controller hashes and selected operating points do not yet exist.
