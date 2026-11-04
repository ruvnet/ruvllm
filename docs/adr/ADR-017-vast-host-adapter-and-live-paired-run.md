# ADR 017: Vast.ai host adapter and the first live paired run

Status: accepted, 2026-11-04. Implements the host side that ADR 015 left to the
operator and executes the ADR 016 measurement on a rented GPU.

## Context

ADR 015 defines a durable scheduler that calls an operator-pinned host adapter;
ADR 016 freezes a paired base/adapter panel. Neither had been run against a
real GPU. The operator authorized a Vast.ai rental with a total spending
ceiling of $5 for the paired measurement.

## Decision

`integrations/metaharness/src/vast-host-adapter.mjs` implements `submit`,
`inspect` and `cleanup` against the Vast REST API with `fetch`, so the API key
travels only in an `Authorization` header and never in argv, files or
structured output. An owner-only host config (`ruvllm-vast-host/v1`) maps the
job's opaque `secretRef` to a key file and pins the container image digest,
repository URL and exact commit, bundle name, disk, per-arm deadline and local
paths. The adapter is self-contained so the loader's single-file SHA256 pin
covers all of its code.

Spending ceiling, in four independent layers:

1. Scheduler admission: `budgetUsd` and per-attempt `maxCostUsd` reservations
   (ADR 015), never refunded.
2. Submit refuses an offer unless `dph_total × timeoutSeconds / 3600`, with
   storage included, fits `maxCostUsd`. It never rents a different offer or
   reuses an existing shared instance.
3. Inspect computes accrued cost from the instance's hourly price and start time
   and fails the attempt (forcing cleanup) once it passes the ceiling. The job
   deadline also forces cleanup.
4. Every rental arms a `systemd-run --user` timer that destroys the instance at
   `timeoutSeconds + guardGraceSeconds` even if no further scheduler tick runs.
   If the timer cannot be armed, the new rental is destroyed and submit fails.

Idempotency: each attempt's instance is labelled `ruvllm-<idempotencyKey>`;
submit first looks for that label. Retained outputs: when the remote status
reports `done`, inspect pulls `/workspace/out` and `/workspace/run` over SSH
(per-rental `known_hosts`), re-runs the frozen `measurement.py` locally on the
retrieved raw responses, and reports `completionVerified` only if the local
report is identical to the remote one. A slow pull returns `running` and
retries; it never reports completion without local evidence. Cleanup deletes
the instance and returns `released: true` only after the instance list no
longer contains it, then disarms the guard timer.

The remote side (`experiments/model-improvement/vast/remote-run.sh`) receives no
credential. It checks out the pinned commit, installs the protocol's pinned
packages into the image's Python, downloads the public pinned snapshot, and
runs `run-pair.py`.

## Freeze revisions before any scored output

Running against real files exposed two latent worker defects. Both were found
before any scored generation, and each produced a new freeze with byte-identical
cases:

- v1: the shard-name check rejected the pinned model's actual filenames
  (`model.safetensors-0000N-of-00002.safetensors`), so v1 could never load.
- v2: `PeftModel.from_pretrained(low_cpu_mem_usage=True)` materializes LoRA
  tensors in the BF16 base dtype before the FP32 upcast, rounding the trained
  adapter (max abs diff 3.05e-5). The worker's `verify_adapter` check rejected
  the load. v3 loads eagerly and is bit-exact. v2's single attempt failed at
  package installation, before inference ($0.057).

An unscored preflight, using a separately frozen `preflight-unscored` bundle, ran
on the attempt-1 instance to confirm these fixes and measure throughput
(~15.6 tokens/s with Transformers' torch fallback for the linear-attention
layers). Its outputs are not part of any result.

## Consequences

The Vast path now works end to end, with receipts for submission, progress,
verification, cleanup and the guard. Any earlier evaluation that loaded this
adapter through PEFT with `low_cpu_mem_usage=True` measured a BF16-rounded
adapter, not the trained FP32 one.
Limits: the guard timer runs on the control host, so a powered-off control
host leaves only the provider's own billing. Completion flags are a trusted-host
contract, not attestation (ADR 015). Throughput uses the unaccelerated
linear-attention path; latency is descriptive only.
