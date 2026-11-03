# ADR 014: Project-local Claude Code and Ruflo experiment module

Status: accepted for portable MCP integration, 2026-11-03.

## Context and trust model

Claude Code and Ruflo hosts need a small tool surface for requesting the bounded
runner. The model is an untrusted proposer. Its plan paths must not grant the
ability to replace the operator config, select commands, install packages, change
global Claude settings or activate a model. Existing native tools and their plugin
remain compatible; the experiment controller is an additional optional plugin.

Assets are checkpoint integrity, approved policies, operator configuration,
executable identity, private artifact paths and evidence accuracy. Entry points
are MCP JSON-RPC stdin, plan/control files and the operator installer. Threats
include path traversal, symlink substitution, output overwrite, malicious package
lifecycle scripts, oversized requests, undeclared arguments, fabricated completion
and authority escalation through a tool description.

## Decision

Ship `plugins/ruflo-metaharness` with manifest, stdio MCP server, scoped skill,
slash command and reviewer agent. The server exposes `experiment_run` with one
argument: an absolute `planPath`. Read the runtime path only from the operator's
`RUVLLM_EXPERIMENT_CONFIG` environment binding. Delegate all lifecycle and root/
digest/policy checks to the same engine used by the CLI. Discovery works without
configuration; execution refuses when configuration is missing.

Bound request/result sizes and the serialized request queue. Reject unknown tools
and arguments. Keep stdout reserved for JSON-RPC. Never give the model arbitrary
shell, provider, training, signing-key or deployment arguments. Returned artifacts
are local receipts, not an authenticated host completion wrapper or proof of
independent execution. Disconnects and timeouts require artifact reconciliation.

The skill guides an inspect/propose/run/review workflow. The reviewer checks
source/candidate/evaluation identities, rejected attempts and exact restoration;
it cannot authorize activation. Ruflo uses the same project-local MCP server
configuration as other MCP clients, independently of the model router. Protocol
compatibility does not claim an installed Claude or Ruflo acceptance run.

## Installer and repeatability

The installer lives at `scripts/install-metaharness.sh`. It uses the pinned Rust
toolchain from `native/`, frozen npm dependencies with lifecycle scripts disabled,
native source verification and the CPU smoke fixture. It generates an owner-only
operator configuration and launcher from explicitly supplied roots and policy
files. It refuses to overwrite existing configuration or accept unsafe path
permissions. It writes no global Claude/Ruflo settings, provider credentials,
Git configuration or npm globals. Existing snapshots/toolchains can be reused.

Full installation prepares dependencies and builds; configure-only mode creates
another private profile using a prepared installation. Requirements and exact
options are in the [plugin guide](../../plugins/ruflo-metaharness/README.md).
The operator retains control of approved policies and the runtime profile.

## Validation and optimization

Acceptance is an actual MCP initialize/list/call exchange through the stdio server
using the real native binary. Test missing configuration, unknown/extra arguments,
and the allowed lifecycle. Installer tests check generated profile permissions,
native readiness and overwrite refusal. The existing plugin's wire tests remain
part of regression validation. CI runs these checks on Linux after the native build.

The server imports the engine once and reuses the Node process across calls.
This avoids repeated module/process startup while preserving all per-experiment
native checks, binary hashing and restoration checks. The CPU benchmark compares
fresh processes against a reused process on the identical fixture workload,
alternates measurement order, retains raw samples and records code/lock/binary
digests. It supports orchestration-overhead claims only. It proves neither faster
large-model editing nor better model outputs.

## Alternatives, limitations and rollback

Global installation would alter unrelated projects and requires wider authority.
A network MCP server adds authentication/deployment concerns absent from local
stdio. Adding these tools to the historical plugin would change its published
surface and receipts. We choose a separate local plugin with a shared runner.

Same-account parent-directory or executable replacement remains outside the
threat boundary; stronger isolation needs an OS sandbox or separate account.
Npm audit covers known advisories in the new locked JavaScript tree, not the
entire repository or unknown package vulnerabilities. No new Rust dependencies
or model weights are installed by the module. Actual Claude/Ruflo client usage
requires the operator's installed host and permissions; portable MCP tests cannot
substitute for that final client-specific acceptance.

Rollback: stop loading the plugin and remove its generated private profile if no
longer needed. Preserve experiment outputs for audit. Native source, historical
benchmarks, base checkpoint bytes and global host settings remain unchanged.
