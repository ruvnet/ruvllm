# Scheduled GPU job loop

Run a durable GPU job queue from the CLI, Claude Code or a Ruflo MCP host.
The scheduler can be developed and contract-tested without live ruOS/Vast.ai
access. A host adapter owns real provider calls and uses its existing ruOS
authentication to resolve **secret references**, never export credential values.

This release implements the schedule, durable state, reservations, host loader,
file handoff and MCP tools. It does not include a live ruOS connector, invent a
secrets-service endpoint, or claim provider E2E validation. No GPU rental or model
download is started by installing the package. A zero-budget sample is disarmed.

## Operator configuration

Install the existing [integration](README.md), then prepare a private directory
and copy [gpu-loop.example.json](examples/gpu-loop.example.json) outside your
output directories. Keep the config owner-only (`chmod 600`); create its
`stateRoot` with mode 700. Use canonical absolute paths without symlinks.

Edit the example's paths, model pin, resource ID and `secretRef`. The example
secret alias and instance ID are placeholders, not discovered ruOS resources.
Keep `budgetUsd: 0` until a spending ceiling is explicitly chosen. Merely reading
status does not submit work. A zero-budget tick records a budget-limit failure
and never calls the provider. Use a new state directory for a new reviewed
configuration; state binds the config's exact byte digest.

Configuration fields:

| Field | Meaning |
| --- | --- |
| `intervalSeconds` | Polling cadence, 1–86400 seconds; active deadlines and pending cleanup can become due sooner |
| `maxAttempts` | Maximum attempts per job, 1–10; retries require confirmed cleanup |
| `budgetUsd` | Total conservative admission budget in exact microdollars; zero disarms submissions |
| `jobs` | 1–128 fixed operator-approved jobs |
| `model` | Exact `repo` and 40-character model revision |
| `task` | `evaluate` or `train`; the host must implement that workload |
| `secretRef` | Opaque alias resolved only by the authenticated host |
| `resource` | Exactly one string `instanceId` or `offerId`; no automatic offer selection |
| `maxCostUsd` | Reserved cost ceiling for each attempt; never refunded automatically |
| `timeoutSeconds` | Per-attempt deadline, 1–86400 seconds |
| `experiment` | Optional absolute `configPath`/`planPath` for the host to execute using the native experiment workflow |

The schedule advances a finite queue. It does not reset succeeded jobs or start
unlimited training cycles. Another experiment cycle needs a new reviewed config
and state root. Reservations limit admitted work; they cannot impose a hard cap
on Vast billing. The host must separately enforce actual spend and worker runtime.

## Host adapter

The host module must be a canonical, owner-only `.mjs` file pinned by SHA256.
Its entire transitive dependency tree is trusted operator code. No provider effects
or secret logging may occur at import: all effects belong in the methods below.
This interface is supplied by RuVLLM; these names are not claimed ruOS API methods.

```js
export async function submit({ job, idempotencyKey, signal }) {
  // Resolve job.secretRef at the authenticated host. Submit OR find the exact
  // existing job by idempotencyKey. Enforce timeout/cost before starting work.
  // Return only a bounded provider run identifier, never credential material.
  return { runId };
}
export async function inspect({ job, runId, idempotencyKey, signal }) {
  // Read the real executor's structured evidence. Do not trust worker stdout.
  return { status: 'running' }; // Or failed/completed with costUsd and completion.
}
export async function cleanup({ job, runId, signal }) {
  // Stop/release only this job's allocation. Do not destroy a shared instance.
  // Confirm absence/release through the provider before returning true.
  return { released: false };
}
```

The snippet is an interface illustration, not a runnable provider implementation.
Use the host's actual supported ruOS secrets and Vast execution routes. Honor
AbortSignal and implement idempotent submit/cleanup. A thrown/late/ambiguous submit
blocks automatic resubmission, since a provider may already have allocated a GPU.

A completed result needs this structured contract, derived by the trusted host:

```json
{
  "status": "completed",
  "costUsd": 0.25,
  "completion": {
    "completionVersion": 1,
    "completionVerified": true,
    "status": "ok",
    "exitCode": 0,
    "assertionsVerified": true,
    "evidenceComplete": true,
    "truncated": false,
    "jobId": "EXACT_JOB_ID",
    "modelRepo": "Qwen/Qwen3.5-4B",
    "modelRevision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "runId": "EXACT_CURRENT_RUN_ID",
    "idempotencyKey": "EXACT_CURRENT_ATTEMPT_KEY"
  }
}
```

Boolean flags are structural checks, not cryptographic attestation. Bind evidence
to the actual run and retain raw measurements at the host. An old attempt's reply,
wrong model, missing assertions or unverified completion cannot mark a job
successful. Terminal success also requires explicit resource-release confirmation.

## Concrete file handoff

For a ruOS-side scheduled consumer, use the bundled
`src/gpu-spool-adapter.mjs`. Set `RUVLLM_GPU_SPOOL_ROOT` to an existing private
canonical directory. This adapter performs **no provider calls**; it hands work
to a host sharing that directory, or a securely synchronized copy of the files.
Transfer and remote authentication are separate host responsibilities.

For each persisted attempt key, files are:

| File | Writer and schema |
| --- | --- |
| `KEY.request.json` | Scheduler: `ruvllm-gpu-spool-request/v1`, `runId`, `job`, credential-handling notice |
| `KEY.status.json` | Host: `ruvllm-gpu-spool-status/v1`, matching `runId`, status, optional cost/completion |
| `KEY.cleanup-request.json` | Scheduler: `ruvllm-gpu-spool-cleanup-request/v1`, `runId`, `jobId` |
| `KEY.cleanup.json` | Host: `ruvllm-gpu-spool-cleanup/v1`, matching `runId`, boolean `released` |

All files must be owner-only, regular, nonsymlink files of at most 64 KiB. The
host should publish replies atomically. Repeated identical requests are idempotent;
conflicting contents under the same key are rejected. No status reply means
pending/running, not success. Missing cleanup acknowledgement keeps cleanup
pending. The request's `secretRef` is an alias; no credential belongs in any file.

Example local bindings (operator paths, not secrets):

```sh
export RUVLLM_GPU_LOOP_CONFIG=/absolute/private/gpu-loop.json
export RUVLLM_GPU_SPOOL_ROOT=/absolute/private/gpu-spool
export RUVLLM_GPU_HOST_MODULE="$PWD/integrations/metaharness/src/gpu-spool-adapter.mjs"
chmod 600 "$RUVLLM_GPU_HOST_MODULE"
export RUVLLM_GPU_HOST_SHA256="$(shasum -a 256 "$RUVLLM_GPU_HOST_MODULE" | awk '{print $1}')"
```

Prepare the private spool/state directories and config before ticking. The
source module's chmod is needed after a normal Git checkout because host modules
are deliberately required to be owner-only. A custom trusted module can replace
the spool adapter; recalculate and review its pin after any code change.

## CLI and loop schedule

```sh
node integrations/metaharness/src/gpu-loop-cli.mjs status --config "$RUVLLM_GPU_LOOP_CONFIG"
node integrations/metaharness/src/gpu-loop-cli.mjs tick \
  --config "$RUVLLM_GPU_LOOP_CONFIG" \
  --adapter "$RUVLLM_GPU_HOST_MODULE" --adapter-sha256 "$RUVLLM_GPU_HOST_SHA256"
node integrations/metaharness/src/gpu-loop-cli.mjs run \
  --config "$RUVLLM_GPU_LOOP_CONFIG" \
  --adapter "$RUVLLM_GPU_HOST_MODULE" --adapter-sha256 "$RUVLLM_GPU_HOST_SHA256" \
  --max-ticks 60
```

`run` defaults to one tick, caps the count at 10000, and waits until the next due
poll/deadline/cleanup step. SIGINT/SIGTERM stops between ticks; it does not erase
running work. State survives restarts. Alternatively, have the operator's existing
scheduler invoke `tick`; overlapping processes refuse the exclusive lock. A coarse
external interval can delay cleanup, so the host must always enforce its own
deadline. No cron entry or background process is installed automatically.

## Claude Code and Ruflo

Load [ruflo-metaharness](../../plugins/ruflo-metaharness/README.md) with the four
bindings above in its process environment. `gpu_loop_status` and `gpu_loop_tick`
take `{}` only. Configuration, host code and budget cannot come from tool args.
The `/ruflo-metaharness:gpu-loop` command checks status and advances a tick within
the operator's configured scope. Ruflo can use the same MCP server and bindings.

If the Claude host supports recurring `/loop`, it can repeatedly invoke this command:

```text
/loop 1m /ruflo-metaharness:gpu-loop
```

That host-specific facility was not E2E-tested here. The durable scheduler still
enforces its own due time, budget, lock and submission state across host restarts.
Existing native experiment tools remain separate; no schedule grants activation.

## Reconciliation and validation

`blocked` means the submit outcome is uncertain. `cleanup` means resource release
has not yet been confirmed. Do not delete state or launch a replacement rental
to make either status disappear. Use the saved key/run ID to inspect the actual
host/provider, reconcile its resources, and only then repair state or prepare a
new reviewed cycle. A crash may leave `tick.lock`; it is never auto-broken.

Run the local scheduler/host/spool tests and all plugin wire tests through the
existing `npm test`/Node commands. Test adapters and host files are explicitly
mock contracts; no secrets, provider, model inference or GPU E2E is exercised.
The [ADR](../../docs/adr/ADR-015-scheduled-gpu-loop-and-host-secret-boundary.md)
records the state machine, failure paths and threat boundary.

Validation on 2026-11-03: **60 local tests passed**, with no failures or skips:
17 scheduler tests, 8 host/CLI tests, 3 spool tests (including the scheduled
handoff), 3 new GPU MCP wire tests, and the original 29 integration/plugin tests.
The long-interval deadline test uses a 60-second cadence and a one-second job
deadline and confirms cleanup becomes due without waiting the full cadence.
Repository integrity and unchanged native-source provenance checks also passed.
Live ruOS/Vast.ai, actual Claude `/loop`, secrets resolution and GPU inference/
training remain deliberately unrun. Existing CI test globs include these tests.
