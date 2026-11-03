# Separate grading stability audit

All 600 additional executions agreed with the original primary verdicts. Twenty saved model outputs were each graded 30 more times using the unchanged macOS sandbox and frozen scorer. Eleven outputs passed every repeat; nine failed every repeat. No answer, delivery or reason disagreement was observed.

This is a post hoc check of grader stability, not another model benchmark or candidate selection step. It generated no new model responses, made no cloud calls and replaced no primary scores. The rental was confirmed absent before the audit started. The audit took 12.083 seconds locally.

The fixed sample covers HumanEval/32, /38, /50 and /53 across all five direct arms. Case 32 uses a seeded random generator and serves as a control. The other three fixtures use unseeded randomness. Thirty repeats do not prove stability for every possible random draw or for cases outside this sample. A future protocol should fix evaluator randomness before inference.

| Historical file | SHA256 |
| :--- | :--- |
| `receipt.json` | `e61ac333805c3ba57f219786571910c7f0b0f28ce314384ea546bff1487a935c` |
| `grading_stability.py` | `af516c5c44e7f90ba467fa013d71247b4ef4682dd554a2b24c9a9da9fecb9e39` |
| `test_grading_stability.py` | `25708233c15f3ba2c0c05bbc1dfc11d23a5f7d2060d90a2fc7cf810c424c98d7` |

The receipt preserves every repeat, reason, duration, primary comparison and the hashes of all 87 tracked source and evidence files. These files remained unchanged throughout the audit. The helper and its tests are historical source copied without edits.

## Reproduce separately

First verify the [cloud release](https://github.com/ruvnet/ruvllm/releases/tag/cloud-evidence-20261103) using the [public verifier](../../../../../scripts/verify_cloud_publication.py). Trust the repository commit and archive digest independently. Extract the verified archive into a new directory. Do not add files inside its immutable tree.

Copy the bundle's `source/project/` into another new directory. Copy this directory's `grading_stability.py` into that project's `research/vast_cross_model/analysis/`. The copied project already contains the original scorer, protocol, fixtures and evidence verifier. Do not replace its scorer with the current public repository's portability revision.

On macOS, using Python 3.12 with `jsonschema==4.26.0`, run:

```sh
python -B NEW_PROJECT/research/vast_cross_model/analysis/grading_stability.py \
  --work BUNDLE/work \
  --primary BUNDLE/results \
  --inputs BUNDLE/inputs \
  --output NEW_AUDIT.json
```

This executes saved generated Python inside the original restricted Seatbelt sandbox. It hashes the historical native executable without running it. Keep the new receipt separate: timings and random verdicts may differ, and neither receipt replaces the primary scores.

Acceptance: the receipt contains 600 repeats over 20 outputs, reports `primary_scores_replaced: false`, and preserves its full source and evidence hash inventory.
