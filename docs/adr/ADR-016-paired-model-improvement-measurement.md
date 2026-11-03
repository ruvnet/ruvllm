# ADR 016: Preregistered paired measurement of adapter improvement

Status: accepted for implementation and local contract validation, 2026-11-03.
Fresh GPU inference and the resulting quality comparison remain unrun in the
current cloud environment.

## Context and question

Existing evidence includes small tool panels, arithmetic anchors and saved
code-evaluation results. Reanalysis can establish what those artifacts say but
cannot create fresh inference evidence or attribute an adapter's quality to
the new MetaHarness integration. The user asks to measure model improvement.
We need a reproducible paired experiment with explicit endpoints, complete raw
outputs and a result that can be negative or inconclusive.

The question is: does the included `weights/coding-4b` adapter improve delivered
exact tool-call accuracy over the pinned Qwen3.5-4B base model on a new 192-case
panel without decreasing its arithmetic-anchor accuracy or completion rate?
This is an adapter comparison, not a causal experiment on MetaHarness itself.

## Frozen inputs and treatment

The model is `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The bundle binds known weight-shard
SHA256 values from the existing cross-model model manifest. The included PEFT
adapter is separately bound by configuration and tensor-file hashes. It has
rank 8, alpha 16, scale 2 and targets `down_proj` at layers 16–31. Loading must
validate the expected 32 low-rank tensors.

Both arms use the same BF16 backbone wrapped in the same `PeftModel`, preserving
the adapter tensors at their trained precision. The control
arm disables the loaded adapter with `disable_adapter()`; the treatment enables
it. This avoids comparing different loading wrappers, a merged checkpoint or
a quantized base against an unquantized treatment. Both run sequentially on
the same host with identical pinned torch, Transformers, PEFT and safetensors
versions, greedy decoding, disabled thinking, seed 20261103 and at most 512 new
tokens. The worker uses model-specific EOS handling. Runtime/device bindings
must match before any comparison is accepted.

Preparation creates 192 tool cases across eight equally represented families
and 64 arithmetic cases. Generated prompts are deterministic and are checked
for exact-text collisions against available repository training, development,
release and historical cross-model prompts. Shared schemas and task semantics
remain. We claim exact prompt disjointness only, not semantic disjointness,
pretraining decontamination or representative coverage of all tool use.

The protocol, cases, expected answers, generation settings and preparation,
worker and scoring source hashes are frozen before either arm runs. An observed
output cannot justify changing the panel, token limit, scorer or decision rule
within that run. A revision requires a new bundle/run identity and disclosure.
No best-of-seeds, best-of-panels or checkpoint selection is performed against
this panel. Any tuning makes the tuned-on panel development data and requires
a separately frozen confirmation panel.

## Outcomes and preregistered analysis

The primary endpoint is delivered exact tool-call correctness on all 192 tool
cases. Each case has one expected function name and typed arguments; malformed
calls, extra calls, wrong argument types or values, missing answers and
token-truncated answers are incorrect. Tool calls are never executed. The
64-case arithmetic panel is a retained-capability anchor scored by its final
boxed integer. Model-written programs are not executed or scored as code
benchmarks in this experiment.

The analysis retains the pairing by case identity. Report both arms' counts
and rates, their percentage-point difference, base-only and adapter-only
correct pairs, and the exact two-sided McNemar p-value. For b base-only and c
adapter-only successes, use twice the lower binomial tail at min(b,c) under
Binomial(b+c, 0.5), capped at one; no discordant pairs gives p = 1. This avoids
an asymptotic approximation on small discordant counts. The tool panel is the
single primary significance test. Arithmetic is an explicit guardrail, not a
second route to declare improvement.

The default research qualification rule requires all of:

1. Complete, correctly bound paired evidence for every frozen case.
2. A positive delivered tool-accuracy difference and primary p < 0.05.
3. No decrease in delivered arithmetic correctness.
4. No decrease in completion rate.

Failure of the primary test is inconclusive or negative according to the
observed direction; it must not be described as confirmed improvement. The
arithmetic rule is an observed no-regression check, not a statistical proof of
noninferiority. No multiple-testing correction is needed for the one declared
primary test; exploratory family results must remain descriptive. A small
templated panel can miss real gains or harms. Cases share construction
patterns, so the case-level paired test also does not establish independence
of broader task families or population-level generalization.

## Evidence and execution boundary

`prepare.py` writes a new frozen bundle. `worker.py` validates local model and
adapter files, verifies source/runtime bindings and writes raw per-case output
plus runtime metadata. Completion is recorded only after all cases finish.
`measurement.py` rejects missing/duplicate/unexpected cases, wrong identities,
wrong protocol/prompt/source/model/adapter/runtime hashes, mismatched arm
bindings or incomplete evidence. Test fixtures cannot qualify through the CLI.
The report is derived from the saved response bytes and expected answers, not
from a claimed provider success flag.

The inference worker uses local files with remote code disabled. Downloading
the public pinned snapshot and installing pinned packages are separate host
preparation operations. The worker does not receive provider keys, resolve
ruOS secrets, execute generated code or initiate paid resources. Runtime records
contain bounded measurement metadata, not environment dumps or credentials.
The run deadline bounds inference work and partial runs fail closed.

The scheduled GPU loop from ADR 015 can orchestrate an authenticated operator
host, but it does not provide a new provider API. A configured host must run
both arms and the scorer and inspect the resulting evidence before returning
its structured completion bound to the scheduler job/run/idempotency key.
Existing experiment config/plan paths keep their existing meaning. Local mock
tests, spool requests and accepted submissions cannot substitute for real
inference artifacts. Completion flags remain a trusted-host contract, not
cryptographic attestation of quality.

## Alternatives, limits and acceptance

Reusing the small historical tool panel would increase exposure to selection
and cannot supply fresh outputs. A different loading path per arm confounds
the adapter effect with runtime changes. Executing model-generated code needs
a separate governed sandbox and benchmark protocol; it is outside this panel.
Renting a GPU without an available authenticated host and configured spend
enforcement would not make the experiment reproducible. We instead complete
the local harness and freeze a runnable panel while reporting live execution
as pending.

Acceptance covers deterministic preparation, prompt collision checks, scorer
correctness, exact paired statistics, integrity/coverage rejection, worker
binding and loading/generation contracts using fixtures. These checks rent no
GPU and are reported separately from model results. Live acceptance requires
both complete real CUDA arms and an independently reproducible scored report.
No fresh numerical improvement is asserted until those artifacts exist.

Even a qualifying result supports only this checkpoint/adapter on this panel
under these settings. A single seed and arm order do not establish latency
improvement; runtime measurements are descriptive. Broader task suites,
repeated seeds or counterbalanced order require separately declared follow-up
experiments. Research qualification supplies evidence for review and never
authorizes automatic activation or changes existing model governance gates.

See the [measurement runbook](../../experiments/model-improvement/README.md).
