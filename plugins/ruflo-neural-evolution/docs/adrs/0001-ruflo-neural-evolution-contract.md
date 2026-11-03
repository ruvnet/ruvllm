# ADR 0001: Native neural experiment plugin contract

Status: Proposed

## Decision

Expose six fixed native operations through a local stdio MCP server: `weight_inspect`, `weight_patch`, `weight_evaluate`, `weight_export`, `weight_rollback` and `weight_verify`. The executable and approved policy bytes are pinned by SHA256 in an owner only operator configuration. Tool arguments contain paths, never shell commands, executable names or permission grants.

Core Memory retains activation authority. The plugin creates offline candidate files and artifacts only. Its evaluator operation validates an existing native evaluation receipt; it does not execute an inference benchmark, authenticate reported metrics or invoke Core Memory promotion.

## Compatibility and pinning

Coordinate plugin conventions with `@claude-flow/cli` v3.6 major and minor. The bridge has no RuFlo CLI runtime dependency. It requires Node.js 22 or newer on a supported Unix platform, Claude Code plugin support, and the compiled RuVector native weight evolution binary. The reference native revision is `f5c3444968ede44e43744460192198962ae94da2`; runtime availability and exact executable digest are checked for every call. No installed WASM runtime is assumed.

## Namespace coordination

The plugin claims the `ruflo-neural-evolution-<intent>` namespace. Follow the ruflo-agentdb ADR 0001 namespace convention for any future shared memory records. No `pattern` or `patterns` memory namespace is written here. Claude Code MCP tools are scoped under the plugin's `neural` server.

## Boundaries

Configuration must pin an executable outside the output directory, approved policy hashes, canonical read roots and one private output root. Configuration itself must also be outside that output root. Output paths must be new and remain inside the configured root. The bridge does not overwrite source files. Final input handles must be regular files. The operator owns root directories and must prevent concurrent hostile changes to their directory entries; this process is not an operating system sandbox.

Requests are limited to 64 KiB. Native stdout is limited to 1 MiB and stderr to 64 KiB. The configured native timeout is between 100 milliseconds and 300 seconds, defaulting to 120 seconds. Exactly one native operation runs at a time; competing requests receive `BUSY`. Cancellation terminates the child process. A timeout or failed receipt write requires reconciliation because an output file may already exist.

The child inherits no API keys, model provider configuration or loader injection variables. It receives only a fixed system PATH, locale, timezone and the private output root as temporary directory. The command is always the pinned native executable with JSON stdin and no shell.

## Verification

The structural smoke contract checks manifest version and keywords, skill and command frontmatter, agent frontmatter, compatibility pin, namespace, Proposed ADR, explicit tool permissions, MCP wiring and entrypoint syntax. Native tests use real BF16 fixture bytes, patch, receipt validation, exact restoration, RVF export and verification over the actual MCP wire. Security checks reject unconfigured tools, traversal, symlink escape, outside writes, existing outputs, unapproved policies, executable drift and unknown tools.

Loading this plugin with `--plugin-dir` changes no global Claude Code or Ruflo configuration. A successful local test does not publish the plugin or activate a model.
