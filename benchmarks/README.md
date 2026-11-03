# Completed local benchmark evidence

This directory preserves 462 saved generations from two separate experiments on Apple M4 Pro. It contains exact copied raw outputs, cases, protocols and summaries, with upstream paths and SHA256 digests in `manifest.json`. No GPU, network access, model download or credential is needed to verify it.

```sh
python3 scripts/verify_benchmarks.py
python3 -B -m unittest discover -s benchmarks/tests -v
```

Both commands run from the repository root and use Python 3.11 or newer and only the standard library. The verifier is read only. The tests use temporary copies. An independently trusted manifest digest can be supplied with `--manifest-sha256 HEX`.

| Completed local comparison | Original | Candidate | Paired gains / regressions | Exact two sided McNemar p |
| :--- | ---: | ---: | ---: | ---: |
| 4B synthetic code and tool release, saved passes | 29/35 | 32/35 | 3 / 0 | 0.25 |
| 4B HumanEval chat protocol, saved passes | 123/164 | 128/164 | 13 / 8 | 0.38331031799316406 |
| 0.8B math, direct, delivered correct | 13/16 | 12/16 | 2 / 3 | 1.0 |
| 0.8B math, thinking, delivered correct | 1/16 | 3/16 | 2 / 0 | 0.5 |

The 4B experiment uses a 4 bit Qwen3.5 4B base with the selected coding adapter, MLX 0.32.4, greedy chat generation and a 768 output token cap. The original and selected checkpoint use identical settings. Selection used a separate development suite; selection metadata is included, but development raw generations are outside this export. The release fixtures contain 27 tool cases and 8 code cases. HumanEval includes all 164 original tasks and tests under the pinned upstream revision and MIT license. Its chat prompts, cap, Python sandbox and runtime differ from leaderboard protocols. The 13 gains and 8 losses do not establish a statistically conclusive general coding improvement. Pretraining contamination is unknown.

| 4B local arm | Saved passed | Reached stop | Length stop | Output tokens | Summed elapsed seconds |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Synthetic original | 29/35 | 35/35 | 0 | 2,484 | 70.865 |
| Synthetic selected | 32/35 | 35/35 | 0 | 2,465 | 76.225 |
| HumanEval original | 123/164 | 157/164 | 7 | 38,940 | 547.506 |
| HumanEval selected | 128/164 | 158/164 | 6 | 37,442 | 607.380 |

The 0.8B experiment uses Qwen3.5 0.8B, BF16 MLX, a published sparse intervention and a 2,048 output token cap. The finite development search selected the full strength edit before release inference. The 16 new synthetic math questions share template families with development. Each arm has one seeded sample per question, so power and generalization are limited. This is not MATH 500 or a coding benchmark. The primary direct comparison failed its quality gate because delivered correctness decreased. No production checkpoint was authorized by this experiment.

| 0.8B local arm | Answer correct | Reached EOS | Delivered correct | Length stop | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Original direct | 13/16 | 13/16 | 13/16 | 3 | 13,458 |
| Edited direct | 12/16 | 14/16 | 12/16 | 2 | 10,713 |
| Original thinking | 3/16 | 1/16 | 1/16 | 15 | 32,295 |
| Edited thinking | 5/16 | 3/16 | 3/16 | 13 | 31,167 |

Direct mode used 20.40% fewer output tokens after editing, while delivering one fewer correct answer. Timing and token observations describe these recorded runs and are not production cost or latency forecasts. The 4B and 0.8B results use different models, precisions, corpora and inference protocols and should not be ranked against each other.

The verifier checks the exact public file inventory, rejects symlinks and extra files, validates source and response hashes, requires every expected case and arm, and recomputes pass, completion, truncation and token totals. It recomputes paired gains, losses and exact p values from saved verdicts. For all 64 math generations it independently parses the final boxed integer, checks the expected arithmetic answer, and derives EOS completion from saved token IDs. Delivered correctness additionally requires a completed answer and a closed thinking block when thinking is enabled.

The 4B code and tool verdicts are historical saved verdicts. This verification does not execute generated code, rerun HumanEval tests, reparse tools with the tokenizer or reproduce inference. Saved output token counts are checked against aggregates; the 4B tokenizer is not replayed. This distinction is explicit in the JSON result as `generated_code_reexecuted: false`. The frozen 0.8B rollback receipt records a byte exact 1,746,942,600 byte checkpoint restoration. This verifier checks receipt consistency only; checkpoint bytes are not distributed or rehashed.

Two additional exact historical receipts support implementation and integration observations:

| Receipt | Recorded observation | Verification boundary |
| :--- | :--- | :--- |
| [Native application](local-08b/native-release-alpha-100.json) | 130,188 BF16 changes in 6.691888 seconds, rounded to 6.692 | Original source, candidate, patch, policy, native executable and delta fields match the frozen experiment |
| [Claude Code coding fixture](local-4b/coding-integration.json) | Read, Edit and Bash fixture passed in 20.826 seconds | Selected adapter digest, 32 matched tensors, completed tool names, exit status and independent test result are checked |

These are observations from the original processes, not timings of a new standalone build. Their full original bytes and source hashes are preserved. The native receipt retains a historical workstation output path as inert evidence. Its signed plan reference is a digest only; the referenced signature is not authenticated by this curated verifier. The coding receipt records one repair fixture, not repository wide coding performance. Neither process is reexecuted by this verifier.

`local-4b/paired-verdicts.json` is an explicitly derived projection of only `release_paired` and `external_paired` from the upstream report. Its source file digest and copied fields are recorded. All other upstream files are copied without changing their bytes. Historical workstation or fixture paths are inert evidence, not runnable commands or access grants. Plan hashes can reference development or governance artifacts outside this curated export; this directory does not claim to reproduce the full original signing or activation lifecycle.

The public scope excludes unfinished cloud results, provider receipts, account metadata, private signing keys, model checkpoints and training data. These files do not establish state of the art, recursive self improvement, safety qualification or production authorization. File hashes and self consistency do not prove remote hardware identity or protect against replacement of the entire unsigned repository and manifest. Obtain the repository commit or manifest SHA256 through an independent trusted channel before relying on authenticity.

Acceptance: `python3 scripts/verify_benchmarks.py` must report `verified: true`, `generations_verified: 462`, `math_text_independently_rescored: true`, `generated_code_reexecuted: false` and `production_authorized: false`.
