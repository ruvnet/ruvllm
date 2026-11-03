# ADR 012: MetaHarness bounded weight experiment runner

Status: accepted for the CPU research integration, 2026-11-03.

## Context and outcome

RuVLLM already enforces tensor allowlists, protected ranges, delta bounds and
exact rollback. The native `evaluate` operation validates receipts; it does not
perform inference. Repeating this lifecycle manually makes it easy to compare
different baselines, lose failed candidates, or mistake an efficiency gain for
an accuracy improvement. We need a repeatable experiment controller that keeps
those native guarantees and explicitly records the origin of evaluation claims.

The actors are a trusted local operator, a proposer (possibly Claude), and an
external evaluator. The operator alone selects the executable, roots, policies,
timeouts and maximum candidate count. A proposer supplies a plan and cannot
change these controls. An evaluator supplies measurements bound to checkpoints,
patches, runtime, protocol and suite identities. No actor receives production
activation authority from this integration.

## Decision

Implement an ESM Node package under `integrations/metaharness`, using published,
exactly pinned packages and an npm integrity lockfile. At initial integration:
`@metaharness/flywheel@0.1.13`, `@metaharness/darwin@0.12.0`, and
`@metaharness/weight-eft@0.1.2`. The inspected MetaHarness source reference is
`2f415bf47a7beb36f6a43c24e85a509acedf9ebb`; published package integrity is pinned
separately and is not asserted to reproduce that Git commit.

Retain the original Rust source and provenance unchanged. Invoke the pinned
native executable with JSON through pipes, no shell, a restricted child
environment, bounded output and a timeout. Rehash its bytes for every operation.

The owner-only `ruvllm-experiment-runtime/v1` file declares `binaryPath`,
`binarySha256`, `readRoots`, `writeRoot`, `policySha256s`, `maxCandidates` (1–16),
and `timeoutMs` (100–300000). The executable and runtime config must be outside
the private output root. Absolute canonical paths and regular files are required;
symlink components and control-file duplicate keys/nonfinite numbers are rejected.

The `ruvllm-experiment-plan/v1` declares `source`, `policy`, `evidenceMode`
(`fixture` or `external`), and candidates with unique IDs, patch paths and optional
evaluation paths. These declarations select data, not executable code. Copy the
approved policy and each bounded input into a new private run directory with
exclusive creation. An input snapshot prevents later data edits from silently
changing the recorded experiment.

## State machine and failure paths

1. Validate runtime, plan, file scope, executable digest, approved policy and limits.
2. Inspect the unchanged source and record its full checkpoint hash.
3. For each candidate, apply the native patch to a fresh output; bind the receipt.
4. Validate supplied evaluation against the patch receipt. All evaluated
   candidates must share baseline, suite and protocol bindings. Missing evaluation
   leaves a mechanically valid candidate unqualified.
5. Apply Flywheel's frozen gate to correctness and completion, additionally
   requiring strict accuracy improvement. Resource cost is unmetered, recorded
   explicitly; zero in the gate is a neutral placeholder, never a free-compute claim.
6. Export and verify native RVF evidence. Restore to a fresh checkpoint and check
   its hash and complete bytes against the source. Rollback runs even after a
   post-patch evaluation rejection. Failure to restore aborts the run.
7. Verify the source is unchanged. Select among research-qualified candidates
   through Darwin's bounded contrastive adapter; keep the immutable experiment
   envelope outside the mutation surface. Persist the report with exclusive creation.

An invalid candidate is recorded as rejected. Structural/preflight failures abort
before candidate execution. Interrupted operations may leave partial artifacts;
retrying uses a new run directory. There is no destructive cleanup or implicit
resumption of a half-finished run. An operator must reconcile partial evidence.

## Evidence and qualification boundaries

Reports bind plan, config, policy, executable, source, patch, candidate,
evaluation and gate identities. Candidate receipts include RVF and restoration
results. `productionActivation`, `activationAuthorized`, and `inferencePerformed`
remain false. Caller-supplied external receipts are not authenticated independent
measurements. Fixture outcomes are synthetic and cannot support model improvement.

Darwin selection is replay over supplied measured scores, not another inference
run or unseen-task confirmation. Scalar accuracy cannot detect every per-task
regression. Statistical confirmation, retained-capability suites, evaluator
identity verification, actual cost metering and production admission remain
deployment responsibilities. Do not use research qualification as deployment approval.

## Alternatives and consequences

Embedding the entire MetaHarness monorepo adds unrelated tooling and lifecycle
dependencies. Reimplementing its gate loses upstream compatibility. Automatically
launching arbitrary evaluator commands expands the proposer trust boundary.
Live SONA/EWC settings are absent from this extracted native repository and
cannot be supplied through prompt text. We choose a narrow file/JSON seam instead.

The cost is full-file hashing, candidate storage and restored copies per candidate.
Those costs deliberately remain in the benchmark. Same-user filesystem races,
malicious operator configuration and a dishonest evaluator are outside the
local bridge's security boundary; separate accounts/sandboxing and evaluator
authentication are needed for stronger isolation.

## Acceptance and rollback

Run the native lifecycle and adversarial engine tests, adapter tests, plugin
wire and installer tests, and the reproducible CPU benchmark. Verify missing
evaluation cannot qualify, policy/binary/source/evaluation mismatches fail,
source bytes never change, and restoration is exact. Preserve failed candidates.
Remove the optional integration/plugin or stop loading its MCP server to roll
back the integration; existing native commands and historical evidence are unchanged.

See [ADR 013](ADR-013-training-and-development-selection.md),
[ADR 014](ADR-014-claude-ruflo-integration.md), and the
[integration guide](../../integrations/metaharness/README.md).
