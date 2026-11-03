# ADR 013: Training-data export and measured development selection

Status: accepted for data export, plans and development replay, 2026-11-03.

## Context

Direct checkpoint edits and LoRA training are separate interventions. RuVLLM's
published adapter targets Qwen3.5 4B, rank 8, selected MLP projections. MetaHarness
Weight-EFT defaults to other targets and emits a `ruvllm microlora` command that
this standalone repository does not implement. Its `runTraining` can return
`trained` without launching a trainer. Treating that status as evidence would
fabricate completion. Its SFT/DPO export and data guards are independently useful.

## Decision: data seam

Expose `exportArchive` around the actual pinned Weight-EFT exporter. Require a
caller-provided holdout list; always enable the upstream reward-hacking monitor
and reject rather than truncate overlength trajectories. Enforce trajectory count,
serialized byte, message, boolean outcome and token-budget limits at this boundary.
Preserve tool calls and upstream drop accounting in standard SFT/DPO data.
Record archive and holdout hashes alongside the report. An empty holdout list is
explicitly permitted for development fixtures; the caller must supply all actual
evaluation IDs for real training.

SFT can include successful frontier trajectories. DPO pairs must be same-model,
same-instance cheap-tier success/failure pairs, as implemented upstream. Instance
disjointness alone does not detect semantic near-duplicates; operators must audit
dataset lineage. The deterministic hacking monitor is a heuristic, not proof that
a gold-resolved claim is genuine. `resolved` remains a caller assertion; do not
silently convert model self-reports into trusted outcomes.

`trainingPlan` uses upstream plan construction, but wraps it with
`executionStatus: not_executed` and `runnerCompatibility: adapter-required`.
Plans contain informational commands only and are never sent to a shell. Do not
copy their strings into shell execution with untrusted identifiers or paths.
Implementing a separately validated configurable trainer, optional DPO objective,
model downloads and paid GPU execution is outside this first integration.

## Decision: development selection

Use the actual Darwin contrastive engine with a single domain component naming
a candidate. The immutable envelope binds the source, policy, executable, plan
and gate. An invariant proposal returns no edits. The domain proposal cites a
measured observation and selects a candidate from a bounded known set; arbitrary
IDs and unmeasured scores are rejected. The probe reads only supplied evidence
and echoes the exact candidate document digest. There are no model or provider calls.

Baseline and candidate scores must be finite fractions, IDs unique, and evidence
digests well-formed. Regressions retain the baseline. The result says
`evidenceMode: score-replay` and `independentConfirmation: false`. The native
runner invokes this only after native evaluation and Flywheel qualification.
The standalone library/JSON CLI is also useful to inspect development records,
but digest shape checks do not authenticate the caller's measurements.

## Rejected alternatives

Calling upstream `runTraining` and trusting its status is rejected because no
trainer executes. Mapping `ewcLambda` into a system prompt is rejected because it
does not protect weights. Automatically adapting the native base tensors from
LoRA output would mix two distinct experiments and invalidate historical evidence.
Activation-lens integration is deferred until activation capture and a fitted,
model-bound lens exist; synthetic lens tests cannot establish real interpretability.

## Acceptance and consequences

Tests exercise real Weight-EFT exports: holdout exclusion, same-model DPO pairs,
reward-hacked success rejection and malformed inputs. Tests exercise actual Darwin
selection, baseline retention, immutable envelopes and invalid scores. Training
plans must always state they were not executed. The JSON adapter CLI consumes
bounded stdin and emits data to stdout; it has no filesystem, provider or trainer
execution API. Frozen model-quality evidence is not changed by these features.

The benefit is a usable experiment/data boundary today. Actual quality improvement
still needs trained weights or effective direct proposals and paired held-out
inference. Training outputs must carry checkpoint and dataset identities before
they can be evaluated by [ADR 012](ADR-012-metaharness-bounded-experiment-runner.md).
