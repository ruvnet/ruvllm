**The targeted 0.8B edit partially restored thinking completion, but did not improve the strongest unedited baseline. The separate 4B adapter improved a small tool fixture set; its coding gain remains inconclusive.**

This experiment completed 1,851 generations across seven arms on 3 November 2026, producing 2,552,202 output tokens. It compared three models from two Qwen families, with model specific interventions kept separate. All planned arms are reported. No candidate was selected, trained, promoted, or activated using these results. State of the art performance and recursive self improvement are not established.

The [machine readable summary](summary.json), [individual scores](scored.json), [generated report](REPORT.md), and [timing analysis](timing.json) preserve the measurements behind this account. The [publication manifest](publication.json) binds the public artifacts. The [closeout record](closeout.json) records resource cleanup and the cost estimate, including any distinction between estimates and reconciled charges.

The main quality measure is **delivered correct**: a response must satisfy its task checks and finish successfully. Code must pass its executable tests; tool calls must satisfy the requested schema, arguments, and action scope; math must deliver the correct final boxed integer. Thinking responses must also close their thinking block. A correct answer found inside unfinished reasoning is recorded separately and does not count as delivered.

| Direct mode model | HumanEval delivered correct | Tool fixtures delivered correct | Procedural math delivered correct |
| :--- | ---: | ---: | ---: |
| Qwen3.5 0.8B original | 50/164, 30.49% | 15/27, 55.56% | 114/128, 89.06% |
| Qwen3.5 0.8B edited | 49/164, 29.88% | 11/27, 40.74% | 113/128, 88.28% |
| Qwen3.5 4B original | 133/164, 81.10% | 24/27, 88.89% | 127/128, 99.22% |
| Qwen3.5 4B LoRA | 135/164, 82.32% | 27/27, 100.00% | 127/128, 99.22% |
| Qwen3 Coder 30B original | 151/164, 92.07% | 12/27, 44.44% | 127/128, 99.22% |

The 4B adapter gained seven HumanEval cases and lost five, for a net gain of two. Its paired p value is 0.7744140625, so this run does not provide convincing evidence of a general coding improvement. The three additional tool successes are useful evidence about these 27 fixtures, not proof of reliable tool use across repositories or production workloads. Coder 30B has the strongest HumanEval result in this comparison. Its weaker fixture result should not be generalized to all tool use.

The targeted 0.8B edit has a different outcome. In direct mode it loses one HumanEval case, one math case, and four tool cases on aggregate. Math completion improves from 123/128 to 127/128 while delivered correctness falls from 114/128 to 113/128. Finishing more responses is therefore insufficient evidence of better answers.

| 0.8B thinking mode on the same 128 math cases | Correct answer found | Completed | Delivered correct | Truncated |
| :--- | ---: | ---: | ---: | ---: |
| Original | 36/128 | 0/128 | 0/128 | 128/128 |
| Edited | 56/128 | 29/128 | 29/128 | 99/128 |

The edit rescues 29 completed correct answers under thinking mode. The original model produces correct boxed answers inside unfinished reasoning on 36 cases; the edited model still has 27 such correct but undelivered answers. This separation prevents an answer hidden inside a truncated response from being counted as successful delivery.

**The essential baseline is the original model with thinking disabled.** That profile was chosen from prior development evidence before this evaluation, and it delivers 114/128 correct math answers under the same 8,192 token allowance. Edited thinking delivers only 29/128. The result supports partial repair of a completion failure in one mode. It does not establish a better general model or justify activating the edit for coding tasks.

The following comparisons use paired delivered correctness on the same cases. A gain means the original failed and the candidate passed; a regression means the reverse. Percentage point changes are absolute accuracy differences.

| Intervention and panel | Gains | Regressions | Change in percentage points | Exact two sided McNemar p |
| :--- | ---: | ---: | ---: | ---: |
| 0.8B edit, HumanEval direct | 6 | 7 | decrease 0.610 | 1 |
| 0.8B edit, tools direct | 0 | 4 | decrease 14.815 | 0.125 |
| 0.8B edit, math direct | 7 | 8 | decrease 0.781 | 1 |
| 0.8B edit, math thinking | 29 | 0 | increase 22.656 | 0.000000003725290298461914 |
| 4B LoRA, HumanEval | 7 | 5 | increase 1.220 | 0.7744140625 |
| 4B LoRA, tools | 3 | 0 | increase 11.111 | 0.25 |
| 4B LoRA, math | 0 | 0 | unchanged | 1 |

These p values are exploratory and are not adjusted for multiple comparisons. They depend on assumptions about independent task outcomes. Related synthetic task families limit that interpretation, and even a very small p value does not establish broad generalization. The comparisons were independently recomputed from the saved primary scores and match the published summary exactly.

Runtime measurements distinguish inference calls from the broader completion phase. Every row in a batch repeats the same duration, so the analysis counts that duration once per batch. Completion phase time additionally includes tokenization, output writes, final model hashing, and protocol verification. Model loading is separate.

| Arm | Output tokens | Inference API seconds | Completion phase seconds | Model load seconds | Aggregate output tokens per inference second |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.8B original direct | 163,619 | 1188.553 | 1191.276 | 16.214 | 137.66 |
| 0.8B edited direct | 142,919 | 964.690 | 967.302 | 14.764 | 148.15 |
| 4B original direct | 108,247 | 611.094 | 619.389 | 96.045 | 177.14 |
| 4B LoRA direct | 106,717 | 559.689 | 567.165 | 15.524 | 190.67 |
| Coder 30B direct | 78,689 | 522.079 | 567.609 | 25.817 | 150.72 |
| 0.8B original thinking | 1,048,576 | 2353.914 | 2356.348 | 14.022 | 445.46 |
| 0.8B edited thinking | 903,435 | 2367.517 | 2369.981 | 15.294 | 381.60 |

Direct editing used 12.65% fewer output tokens and 18.83% less inference time while losing delivered correctness. Thinking editing used 13.84% fewer tokens but took 0.58% longer. In a synchronous batch, the longest generation can keep the entire batch running after other responses finish. Token reduction therefore does not imply lower elapsed time.

The 4B adapter used 1.41% fewer tokens and 8.41% less inference time in this run. These are descriptive observations from one sequential execution, not repeated performance estimates. Model preparation and cache effects also differ between loads; the 96.045 second original 4B load versus the 15.524 second adapter load is not evidence that the adapter intrinsically loads faster. Aggregate throughput is not individual request latency, and these measurements should not be presented as production cost forecasts.

The frozen experiment used the following settings. Exact model revisions, weight shard hashes, and intervention inputs are recorded in the [model pins](../../models.json), [input pins](../../assets.json), and [protocol](../../protocol.json).

| Item | Recorded protocol |
| :--- | :--- |
| Compute | One A100 SXM4 80GB GPU; tensor parallel size 1 |
| Runtime | vLLM 0.21.0, Transformers 5.12.1, PyTorch 2.11.0 |
| Precision | BF16 for all five variants |
| Models and variants | Original and edited Qwen3.5 0.8B; original and LoRA merged Qwen3.5 4B; unchanged Qwen3 Coder 30B A3B Instruct |
| 0.8B intervention | Published targeted edit, applied through the native ruvLLM implementation; 130,188 BF16 values across two allowed tensors |
| 4B intervention | Previously selected LoRA, merged in FP32 and then rounded to BF16; separate from the 0.8B edit |
| Direct arms | Five variants, each evaluated on 164 HumanEval, 27 tool, and 128 procedural math cases |
| Thinking arms | Original and edited 0.8B, each evaluated on the same 128 math cases |
| Output limits | 2,048 tokens for code and tools; 8,192 for math |
| Sampling | Temperature 0, top p 1, top k disabled, repetition penalty 1, seed 2026110307 |
| Scheduling | Batch size 8; maximum 8 sequences and 8,192 batched tokens; context limit 12,288 |
| Runtime isolation | Fresh process per arm, prefix caching disabled, eager execution, GPU memory utilization 0.90 |
| Primary grading | Saved outputs evaluated locally; code runs inside the macOS Seatbelt sandbox; strict tool checks and exact integer math checks; no model judge |
| Resource bounds | Shared original deadline; maximum 20,700 seconds and 9,295,872 output tokens in the protocol |

The native intervention was applied only to its pinned 0.8B checkpoint. It was not transferred blindly to other architectures. Native rollback restored the original checkpoint exactly, verified by its full digest and byte comparison. The 4B merge separately checked that all untouched tensors remained unchanged. These checks establish artifact integrity and reversibility; they do not grant production deployment authority.

The final protocol SHA256 is:

```text
cf83995f29b5c3f7c28871d87d1295839bd94452dec0b8c352b990ba52a26cb6
```

The container was pinned to this Linux AMD64 manifest:

```text
docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
```

Three preparation corrections are retained in the evidence rather than hidden. They occurred before any scored generation:

1. The initial freeze was replaced before GPU generation to include both official Coder stop tokens, 151645 and 151643. The [unused initial freeze](../../archive/unused-freeze-01/abandonment.json) records that change.
2. Runtime inspection showed that Transformers 5.12.1 returns a dictionary by default from `apply_chat_template`. The worker expected a flat token ID list. The correction explicitly requests `return_dict=False` and validates that contract. This was a discovered compatibility issue, not a recorded scored inference failure. The [original runtime attempt](../../archive/unused-runtime-attempt-01/abandonment.json) and the [unused intermediate correction](../../archive/unused-tokenizer-only-freeze/abandonment.json) remain archived.
3. The original unscored preflight actually stopped before model construction because the Qwen3.5 tokenizer EOS token 248046 was missing from the configured stop set. The final correction includes both 248046 and 248044 for all four Qwen3.5 variants. The observed tokenizer and configuration evidence and original failure log are preserved in the [runtime evidence](../../archive/unused-runtime-attempt-01/eos-runtime-evidence.json).

The corrected 0.8B and Coder preflights then passed. Questions, arms, scoring definitions, sampling settings, token limits, and the resource deadline were preserved through these corrections. There was no retry or replacement of a scored arm. All seven final arms completed under the final protocol.

A separate [post hoc grading stability audit](grading-stability/README.md) examined a limitation in randomized HumanEval test fixtures. It reevaluated 20 saved outputs, covering four cases across the five direct arms, 30 times each. All 600 grader calls completed with zero disagreements with the primary verdicts and no variation in repeated verdicts. `HumanEval/32` was the locally seeded control; `HumanEval/38`, `HumanEval/50`, and `HumanEval/53` retained their unseeded test draws.

That audit generated no new model responses, changed no primary scores, and selected no model. It provides evidence of stability for those saved outputs. It does not make unseeded tests deterministic, validate every HumanEval implementation, or remove the distinction between primary evaluation and later auditing. Separate processes and reviewing agents operated under the same operator; this is not external certification.

The largest remaining uncertainty is transfer to real coding and tool workflows. HumanEval is a repeated public benchmark with unknown pretraining contamination. The 27 tool fixtures are small and synthetic, and their task families can resemble training and development examples. The 128 math cases are procedural arithmetic rather than MATH500 or advanced reasoning. This is one GPU and one generation seed. The earlier Mac results used a different runtime and, for the 4B candidate, a different deployment precision; they should not be pooled with these cloud scores.

A useful next evaluation would freeze unrelated repository tasks and tool schemas, preserve protected capability checks, repeat across seeds and execution order, and retain the strongest unedited inference profile. The current evidence supports keeping the 0.8B edit as a repair experiment and the 4B adapter as an experimental specialist. It does not justify automatic activation or a claim of recursive improvement.

The final download was verified before teardown, including all seven completion receipts, the final run state and all 114 continuation inventory hashes. Rental absence was confirmed at 09:26:10 UTC. Complete cleanup was verified at 09:29:49 UTC, including two further absence observations and removal of the guard job, its service account, and its secret binding. These are operator recorded consistency checks, not provider signed attestation or model quality evidence.

The [closeout record](closeout.json) estimates **USD 3.745339 for GPU and storage**, using 10,673.908 seconds of elapsed rental time and the recorded USD 1.263194 hourly quote. Actual transfer charges, guard charges, and the final invoice remain unreconciled. This figure is therefore a partial cost estimate, not total invoiced spend.

The run reserved USD 30 within the authorized USD 200 budget. Shared reservations, including prior work, totaled USD 34.458369. Reservations and transfer allowances are planning amounts, not invoices or enforced provider billing caps. No cost is inferred from token throughput here.

The [cloud evidence release](https://github.com/ruvnet/ruvllm/releases/tag/cloud-evidence-20261103) carries the reproducibility archive. Its exact size is 38,962,746 bytes and its SHA256 is:

```text
337d1b0ab3f9efd94baf14e30078b73e112c54a0203aec6c3d62ed91f9d48603
```

The archive preserves raw generations, model and runtime bindings, preparation and rollback receipts, correction history, and the frozen evaluation inputs. Digest checks bind the recorded artifacts; they are not hardware attestation. The [publication manifest](publication.json) supplies the public artifact inventory.

Acceptance test: from the repository root, independently reaggregate the primary score records and recompute every paired exact probability. This reads saved scores and executes no generated model code.

```sh
python3 - <<'PY'
import json
import math
from pathlib import Path

folder = Path('research/vast_cross_model/results/cross-model-20261103')
summary = json.loads((folder / 'summary.json').read_text())
scored = json.loads((folder / 'scored.json').read_text())
protocol = json.loads((folder.parents[1] / 'protocol.json').read_text())
assert sum(len(rows) for rows in scored.values()) == summary['generations'] == 1851
assert sum(row['output_tokens'] for rows in scored.values() for row in rows) == 2552202
for arm, rows in scored.items():
    for panel, expected in summary['arms'][arm]['panels'].items():
        selected = [row for row in rows if row['panel'] == panel]
        assert len(selected) == expected['count']
        for key in ('answer_correct', 'completed', 'delivered_correct', 'truncated', 'output_tokens'):
            assert sum(row[key] for row in selected) == expected[key]
for baseline, candidate in protocol['comparisons']:
    for panel, expected in summary['paired'][candidate].items():
        left = [row for row in scored[baseline] if row['panel'] == panel]
        right = [row for row in scored[candidate] if row['panel'] == panel]
        assert [row['id'] for row in left] == [row['id'] for row in right]
        gains = sum(not a['delivered_correct'] and b['delivered_correct'] for a, b in zip(left, right))
        losses = sum(a['delivered_correct'] and not b['delivered_correct'] for a, b in zip(left, right))
        n = gains + losses
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.0
        assert (gains, losses, p) == (expected['gains'], expected['regressions'], expected['exact_mcnemar_two_sided_p'])
print('Verified 1851 records, 2552202 tokens, and all seven paired comparisons.')
PY
```
