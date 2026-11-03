# ADR 015: Scheduled GPU loops with host-owned secret resolution

Status: accepted for local scheduling and adapter contract validation, 2026-11-03.
Live ruOS/Vast.ai E2E is explicitly excluded from this implementation's acceptance.

## Context and objective

The MetaHarness integration validates native checkpoint experiments locally.
Real inference and training need a GPU runtime. The operator has ruOS secrets
and wants Vast.ai work to advance on a loop schedule without waiting for this
chat's missing ruOS connector tools. We can implement persistence, scheduling,
budget admission and host contracts without inventing a ruOS secrets endpoint
or claiming a live provider job executed.

Actors are the trusted operator, local scheduler, authenticated host adapter and
GPU worker. The host retains ruOS authentication and resolves opaque secret
references. No API key, token or raw secret crosses the scheduler/spool/MCP
interface. A model can request one tick or status, but cannot choose credentials,
provider offers, code modules, cost limits or experiment paths through tool args.

## Architecture and contracts

`gpu-loop.mjs` implements a persistent single-writer state machine;
`gpu-loop-cli.mjs` exposes `tick`, `status` and bounded `run` operations.
An owner-only operator configuration pins model repository/revision, task,
opaque `secretRef`, an existing instance or offer ID, per-attempt cost reservation,
job timeout, polling interval and maximum attempts. State binds the exact config
digest. Changing config while preserving its state is rejected, so revisions
require deliberate reconciliation or a new state root.

`gpu-host.mjs` verifies the owner-only module entrypoint's SHA256 before import.
It requires structured `submit`, `inspect` and `cleanup` functions. This is a
local integration interface, not an assertion that those are native ruOS API
methods. Its code and transitive imports are trusted operator code; the hash
does not pin the whole module dependency tree. Entrypoint and parent-directory
replacement by the same account remain outside the threat boundary.

`submit` receives the immutable job and persisted idempotency key. The host must
resolve `secretRef` using its supported ruOS mechanism, submit or locate the
corresponding Vast job, enforce the requested budget/deadline and return its
run identity. `inspect` returns structured status. `cleanup` must confirm release
of the exact allocation created or reserved for that job. Existing shared
instances must not be blindly destroyed. The local scheduler neither retrieves
secret values nor parses stdout as successful execution evidence.

The bundled `gpu-spool-adapter.mjs` provides a concrete filesystem handoff for a
ruOS-side scheduled consumer. It writes exclusive, idempotent request and cleanup
request files and reads bounded owner-only status/cleanup replies. A request is
only a request; it performs no provider operation. Pending work remains pending
until a host replies. No bridge callback or vendor HTTP endpoint is guessed.

## State transitions and durability

1. A tick validates config and acquires an exclusive private lock directory.
2. A due queued job reserves its full cost ceiling before any host submission.
   The key, attempt count and `submitting` intent are fsynced before effects.
3. A known run becomes `running`; later ticks inspect it. An ambiguous submit
   or recovered `submitting` state becomes `blocked`. Never blindly resubmit an
   operation whose provider outcome is uncertain.
4. A deadline, failed inspection or reported terminal result moves to `cleanup`.
   Deadline and cleanup work may advance ahead of ordinary polling intervals.
5. Cleanup remains pending until the host explicitly returns `released: true`.
   No next job starts while cleanup remains unresolved. A known failed job can
   retry only after cleanup, within attempt and reservation limits.
6. Successful terminal state requires complete verified completion fields bound
   to the current job, model repository/revision, run and idempotency key. Raw
   completion/provider responses and exception strings are not persisted.

State uses atomic rename, fsync and a conservative reservation ledger. Provider
reported costs are validated against ceilings, but reservations are not refunded
on failure or low reported usage. Reservation limits govern admission, not the
provider's billing system: actual hard spend and timeout enforcement belongs
to the host/provider. Zero total budget disarms submission.

Locks are never automatically declared stale. A crash may leave a lock and
uncertain provider work. The operator must check the host/provider before
removing a lock or reconciling state; automatic expiry could create duplicate
rentals. Timed-out adapters receive an AbortSignal, but trusted adapters must
honor cancellation. No timer can forcibly prevent in-process trusted code from
performing a late remote effect, which is why ambiguous submissions block.

## Schedule, security and failure semantics

The CLI can run a bounded number of ticks at configured intervals or serve as
an external schedule's one-shot command. MCP exposes `gpu_loop_status` and
`gpu_loop_tick` with empty argument objects and operator environment bindings.
The Claude skill supports hosts with a recurring `/loop` facility; availability
of that facility is host-dependent and untested here. No OS cron, provider
schedule, or autonomous process is installed/enabled implicitly.

Paths, config, state, adapter and spool files are owner-controlled and bounded;
duplicate JSON keys, invalid states, malformed outcomes, wrong identities,
symlinks, budget overruns and unapproved module bytes fail closed. Fixed error
codes keep host exception strings out of model-visible output. A real host
must never return credential values in structured fields either.

Completion flags are a contract check, not cryptographic attestation. The host
must derive them from its trusted executor's actual structured response, not
from agent-written JSON. A spool is protected by filesystem ownership; it does
not authenticate a remote principal or prove model quality. Existing native
evidence checks and external activation authority remain unchanged.

## Alternatives and acceptance

Directly implementing an undocumented ruOS secrets API risks leaking credentials
and depending on invented capabilities. Automatically renting an unspecified
GPU has no reviewed spending ceiling. In-memory intervals lose submission intent
on restart. Auto-breaking locks can duplicate jobs. We choose explicit operator
configuration, durable scheduling and host-owned effects instead.

Acceptance uses injected local host mocks and real disk/CLI/MCP exchanges. Test
restart recovery, duplicate prevention, exclusive ticks, due times, reservations,
deadlines, cleanup failures, replayed/wrong-model completions, secret-safe errors,
adapter pinning and spool identity/permission checks. These are contract tests;
no GPU is rented and no ruOS secret is resolved. Live provisioning, actual
inference/training quality and provider cleanup verification remain unrun as
requested. See the [loop runbook](../../integrations/metaharness/GPU-LOOP.md).
