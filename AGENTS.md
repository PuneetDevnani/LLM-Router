# AGENTS.md
Project:  SQL-agent verification/recovery controller (Python 3.11, Ollama, numpy).
## Hard rules
- NEVER edit or regenerate results/main_v1_frozen/** (frozen published evidence). Never edit study/data.py templates/splits for existing templates, or `correct()` in pilot/environment.py.
- The old test templates (january, region_net, large_net) have been seen. They may be used for training/validation of v2 only, never as the v2 test set.
- Never tune prompts, checks, thresholds or hyperparameters using v2 test outcomes. Test is run exactly once after freezing.
- Gold answers, template IDs, split labels and DB seeds must not enter model prompts or controller features (keep the leakage test passing).
- Every new behaviour defaults to old behaviour so v1 results stay reproducible.
- Write outputs to results/v2/ only. Append-only JSONL; keep resumability.
- After every change run: python -m unittest discover -s pilot -p 'test*.py' && python -m unittest study.test_study
- Add tests for every new function. Commit per phase with a message describing what and why.
- Report negative results as-is. Do not pursue "better numbers" by any route other than the protocol below.
