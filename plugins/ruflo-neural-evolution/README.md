# RuFlo Neural Evolution

Run bounded native weight experiments from Claude Code, including when Claude Code uses a model router. Six MCP tools inspect tensors, apply approved patches, validate evaluation receipts, export RVF artifacts, restore original bytes and verify integrity. They never activate a model.

## Compatibility

Plugin convention target: `@claude-flow/cli` v3.6 major and minor. The bridge itself has no RuFlo CLI dependency. It requires Node.js 22 or newer, a supported Unix platform and Claude Code plugin support. Reference native source: RuVector `f5c3444968ede44e43744460192198962ae94da2`. No WASM runtime or provider API is assumed.

## Configure and load

1. Build and verify the [standalone native engine](../../native/README.md) included in this repository. From the repository root:

```sh
(cd native && cargo test --locked -p ruvllm-weight-evolution --lib && cargo build --locked --release --bin weight_evolution)
python3 native/verify_source.py
python3 native/smoke.py
shasum -a 256 native/target/release/weight_evolution
```

Use the digest of your own verified build in the configuration. A local build is not expected to match the historical experiment executable byte for byte.

2. Create a private output directory and an owner only configuration file outside that directory. Populate exact paths and approved policy hashes:

```json
{
  "schema": "ruflo-neural-runtime/v1",
  "binaryPath": "/absolute/path/to/ruvllm/native/target/release/weight_evolution",
  "binarySha256": "<64 lowercase hexadecimal characters>",
  "readRoots": ["/absolute/model/root", "/absolute/experiment/input/root"],
  "writeRoot": "/absolute/private/experiment/output",
  "policySha256s": ["<approved policy file SHA256>"],
  "timeoutMs": 120000
}
```

3. Set the configuration path and load the plugin for this invocation:

```sh
chmod 700 /absolute/private/experiment/output
chmod 600 /absolute/path/to/runtime.json
export RUFLO_NEURAL_CONFIG=/absolute/path/to/runtime.json
claude --plugin-dir /absolute/path/to/ruvllm/plugins/ruflo-neural-evolution
```

Missing configuration refuses every tool call. The bridge rechecks executable and policy bytes on each call. It adds the output root to allowed read roots so generated artifacts can be inspected and restored. A policy hash approved by the operator cannot be replaced by a model supplied policy. An empty approved policy list disables patch, rollback and export while retaining configured inspection and verification.

Do not place credentials in the configuration or artifacts. Read roots should cover only the checkpoint and required experiment input files. The operator controls those directories; this bridge does not replace operating system isolation against a hostile local account.

Canonical paths and opened handle checks do not protect against a malicious process with the operator's own filesystem authority replacing parent directories or the executable between validation and launch. The pinned executable and configuration must remain outside plugin output scope. Use an independent operating system account or sandbox when that stronger threat boundary is required.

## Tools

| Tool | Path arguments | Result |
| --- | --- | --- |
| `weight_inspect` | `source`, `tensorList` | Native tensor shape, precision and value statistics; no activation statistics |
| `weight_patch` | `source`, `output`, `patch`, `policy`, `receiptOutput` | New candidate checkpoint and saved native receipt |
| `weight_evaluate` | `evaluation`, `receipt` | Receipt contract validation; no inference or automatic qualification |
| `weight_export` | `patch`, `policy`, `receipt`, `output`; optional `evaluation`, `governanceEvidence` | Complete RVF container with immutable evidence |
| `weight_rollback` | `source`, `output`, `patch`, `policy`, `receipt`, `receiptOutput` | New checkpoint restoring original bytes and saved receipt |
| `weight_verify` | `artifact`, `expectedDigestFile` | RVF digest and container integrity verification |

`tensorList` contains a JSON array of 1 through 256 unique tensor names. `expectedDigestFile` contains exactly `{"sha256":"<expected artifact digest>"}`. Every argument is an absolute path within the configured roots. Outputs must not already exist. All writes preserve the source checkpoint.

The `neural-experiment` skill describes the evidence workflow. The `/ruflo-neural-evolution:neural-inspect` command guides inspection. The `neural-reviewer` agent reviews artifacts without requesting activation.

## Namespace coordination

The reserved intent namespace is `ruflo-neural-evolution-<intent>`, following ruflo-agentdb ADR 0001 namespace conventions. No shared memory namespace is currently written. MCP tools belong to the plugin's `neural` server and remain available regardless of the model provider selected by Claude Code Router.

## Verification

From the source repository:

```sh
bash plugins/ruflo-neural-evolution/scripts/smoke.sh
claude plugin validate --strict plugins/ruflo-neural-evolution
RUFLO_NEURAL_TEST_BINARY=/absolute/path/to/ruvllm/native/target/release/weight_evolution \
  RUFLO_NEURAL_TEST_BINARY_SHA256=<verified-digest> \
  node --test plugins/ruflo-neural-evolution/test/native-wire.test.mjs
```

The native test creates a tiny BF16 checkpoint locally and proves patching, complete byte restoration and RVF integrity through the real MCP wire. Its evaluation outcomes are synthetic fixtures, not evidence of model improvement. Generated test artifacts remain temporary. No cloud cost, provider request, global installation or production activation is involved.

For an additional installed Claude Code acceptance check, first finish any model benchmarks and start the local CCR and MLX coding model using the [repository instructions](../../README.md). Then run `python3 plugins/ruflo-neural-evolution/scripts/accept-claude.py` with `--project-root`, `--binary` and `--binary-sha256`. It loads this plugin with `--plugin-dir`, provides only this MCP bridge through `--strict-mcp-config`, disables built in tools and allows only `mcp__neural__weight_inspect`. The fixture's expected digest is withheld from the prompt and must appear in an actual completed tool result. The receipt records Claude, router, model adapter and native executable bindings. This command uses the configured local coding model and is separate from the CPU wire test.

Historical installed client receipts describe the original plugin snapshot and native build. Portable documentation edits and a new executable require a new acceptance receipt before claiming the same installed client result for this checkout.

Requests are bounded to 64 KiB, native stdout to 1 MiB and stderr to 64 KiB. One operation runs at a time. Native processes have a maximum configured duration of 300 seconds. A cancelled or timed out mutation can leave a completed artifact; reconcile output and receipts before retrying.

The MCP request envelope accepts optional `_meta`, including a progress token, as defined by the [MCP request schema](https://modelcontextprotocol.io/specification/2025-06-18/schema#jsonrpcrequest). Metadata stays inside the 64 KiB request limit and is ignored by the native operation. It cannot add tool arguments, change configured paths or grant permission. The installed client acceptance receipt binds the plugin file hashes and sanitized event log hashes; preserve failed attempts separately when rerunning after an interoperability fix.

## Architecture Decisions

[ADR 0001: Native neural experiment plugin contract](docs/adrs/0001-ruflo-neural-evolution-contract.md) is Proposed. [Claude Code's plugin reference](https://code.claude.com/docs/en/plugins-reference) defines the local manifest and MCP wiring format. Load the checked out plugin with `--plugin-dir`; no marketplace or global installation is required.
