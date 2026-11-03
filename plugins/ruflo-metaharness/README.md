# RuFlo MetaHarness weight experiments

This project-scoped Claude Code plugin exposes one MCP tool, `experiment_run`, around the [RuVLLM MetaHarness runner](../../integrations/metaharness/README.md). It uses the same standard stdio MCP interface with a Ruflo-managed Claude Code session or a different model provider. It neither installs nor requires a Ruflo CLI runtime.

## Install

Prepare an input directory containing your frozen plan, approved native policy, patches and checkpoint. On Linux or macOS with Python 3.12+, Node 22+ and the pinned Rust toolchain:

```sh
bash scripts/install-metaharness.sh \
  --config-dir /absolute/private/operator-profile \
  --read-root /absolute/experiment-inputs \
  --policy /absolute/experiment-inputs/policy.json \
  --output-root /absolute/private/experiment-outputs
```

Repeat `--read-root` or `--policy` to allow additional explicit inputs. The installer runs `npm ci --ignore-scripts`, builds the locked native CLI, verifies source provenance, and executes the tiny native lifecycle smoke test. It creates a mode-600 `runtime.json` with the local binary and approved policy hashes plus a mode-700 `launch-claude.sh`. The config and binary must remain outside output scope. Existing profile files are never overwritten and pre-existing config/output directories must already be owner-only. No global Claude settings, global npm installation, credentials, model downloads or training jobs are involved. `--configure-only` creates another operator profile using the already installed dependencies and binary.

Run the generated launcher, or configure the environment explicitly:

```sh
export RUVLLM_EXPERIMENT_CONFIG=/absolute/private/operator-profile/runtime.json
claude --plugin-dir /absolute/path/to/ruvllm/plugins/ruflo-metaharness
```

The slash command `/ruflo-metaharness:weight-experiment /absolute/path/to/plan.json` runs the scoped skill. All execution remains subject to the host's tool permissions. Plugin permission entries do not grant authority over models, filesystem roots, policy hashes, credentials or deployment.

For a Ruflo or another standard MCP host, register this server in that host's **project-local** MCP configuration using its supported configuration mechanism:

```json
{
  "mcpServers": {
    "ruvllm-experiments": {
      "command": "node",
      "args": ["/absolute/path/to/ruvllm/plugins/ruflo-metaharness/scripts/server.mjs"],
      "env": { "RUVLLM_EXPERIMENT_CONFIG": "/absolute/private/operator-profile/runtime.json" }
    }
  }
}
```

This documents protocol compatibility; no installed Ruflo or Claude client acceptance is implied. Do not copy API tokens into this configuration.

## Contract and limits

`experiment_run` accepts exactly `{"planPath":"/absolute/input/plan.json"}`. The operator config is read only from `RUVLLM_EXPERIMENT_CONFIG`; tools cannot replace it. The engine checks path scope, SHA256 pins, candidate bounds and replay/output rules. The MCP bridge limits requests to 64 KiB, results to 1 MiB and its serialized queue to 16 requests. There is no arbitrary command or activation tool. A missing runtime refuses execution while discovery still works.

Version 0.2 adds `gpu_loop_status` and `gpu_loop_tick`, each accepting `{}` only.
The operator sets `RUVLLM_GPU_LOOP_CONFIG`, `RUVLLM_GPU_HOST_MODULE` and
`RUVLLM_GPU_HOST_SHA256`; the bundled file adapter also reads
`RUVLLM_GPU_SPOOL_ROOT`. A tick may submit GPU work through the pinned host
adapter within the configured budget. It never returns ruOS secret values. The
host owns actual Vast.ai operations and provider-side cost/deadline enforcement.
The `/ruflo-metaharness:gpu-loop` command can be called by a host loop schedule.
See the [GPU loop runbook](../../integrations/metaharness/GPU-LOOP.md). These
interfaces have local contract tests; live ruOS/Vast E2E was intentionally not run.

The runner actually patches checkpoints, creates restoration outputs, checks original/restored digests and packages native RVF evidence. A fixture proves those mechanical contracts using synthetic evaluations. External evaluation files remain caller-supplied claims; hashes establish binding, not evaluator independence. The plugin does not run inference or verify a model has improved. The `experiment-reviewer` agent documents the review boundary.

A timeout, interrupted connection or failed response can leave artifacts. Reconcile the private output directory before retrying. Never erase failed evidence merely to make a repeated plan execute. Files owned by a hostile process with the same account, parent-directory replacement and untrusted host configuration require a stronger operating-system boundary.

## Validate

```sh
node --test plugins/ruflo-metaharness/test/*.test.mjs
claude plugin validate --strict plugins/ruflo-metaharness
```

The portable wire tests use the real engine/native binary and temporary fixtures. The installer test exercises configuration, private permissions, source verification, native smoke and overwrite refusal. The Claude validation command additionally requires an installed Claude Code client and is separate from portable checks.
