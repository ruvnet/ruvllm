# MetaHarness integration acceptance evidence

Validation date: 2026-11-03. Base repository commit:
`f2e7c0e6f1ad53130188ecb9937f62786375e604`. Tests ran on the Linux x86-64 cloud
workspace using Python 3.12.14, Node 24.22.1 and Rust 1.97.1. This is current
instance evidence; hosted GitHub Actions and installed-client acceptance are
separate. CI now registers the new tests after the native build with Node 22.

## Requirement-to-evidence trace

| Requirement | Executed evidence | Result |
| --- | --- | --- |
| Detailed architecture and threat decisions | ADRs 012, 013, 014; integration and plugin guides | Contracts, alternatives, failure paths and limitations recorded |
| Real bounded native experiments | Engine tests execute inspect, patch, receipt evaluation, export, verify and rollback | Passed, with source preservation and complete restoration byte comparison |
| Actual MetaHarness package integration | Flywheel gate, Darwin contrastive selection, Weight-EFT exporter/plan adapters | Exact versions and registry integrity pinned |
| Reject unsafe/mismatched inputs | Engine tests for protected edits, forged evaluation identity, mismatched suites, binary/policy drift, duplicate JSON, file limits, root escape, permissions, symlinks and duplicate IDs | Passed; invalid or unevaluated candidates cannot become qualified |
| Public data interface | JSON CLI tests for export, training-plan and malformed/unknown commands | Passed; training always marked unexecuted |
| Claude Code/Ruflo module | Actual MCP initialize/list/call exchanges, missing-config/extra-argument/oversize/duplicate-key checks, repeated native runs | Passed portable wire tests; previous artifacts preserved |
| Project-local installer | Full default installation executed; automated configure-only profile, native smoke, permissions, overwrite and symlink refusal | Passed; no global configuration changed |
| New integration regression suite | `node --test integrations/metaharness/test/*.test.mjs plugins/ruflo-metaharness/test/*.test.mjs` | 29 passed, 0 failed, 0 skipped |
| Existing native behavior | `cargo test --locked --workspace`, `native/verify_source.py`, `native/smoke.py` | 508 tests and 2 doctests passed; provenance and lifecycle verified |
| Existing plugin behavior | Original structural smoke and native-wire suite | 20 structural checks and 4 wire tests passed |
| Original evidence retained | Repository check, benchmark verifier, benchmark unittest suite | Integrity checks and 16 tests passed; historical scorer/protocol unchanged |
| Portable Python regressions | Existing selected CPU/gateway/cloud tests, excluding three diagnosed macOS-only tests | 47 tests and 37 subtests passed; 3 deselected |
| Benchmark and optimization | Fresh/reused process comparison, 7 measured samples plus 2 warmups per mode, 2 candidates per experiment | All 18 experiments passed; p50 238.42 ms versus 114.58 ms |
| Dependency assessment | `npm audit --omit=dev --json`, advisory query on validation date | 0 reported vulnerabilities in the new 3-package dependency tree |

The new engine, CLI and plugin tests have no skipped cases. Three original Python
tests need macOS Seatbelt; an earlier full run diagnosed `sandbox unavailable`
on Linux. They were explicitly deselected for portable regression validation,
not rewritten or weakened. The existing macOS CI job retains them.

The audit lockfile SHA256 is
`8610e044122c744ecfb317bc671989b0e18b126433fb894b6d2b13ee6eb6e79d`.
This audit queries known npm advisories; it is not a complete security
certification of JavaScript, Rust, Python or the operating system. Independent
review found no concrete high/critical issue within the documented trusted-local-
operator boundary. Same-account filesystem replacement and evaluator honesty
remain limitations, not properties established by these tests.

## Performance scope

The retained [raw benchmark summary](metaharness-cpu-20261103.json) binds the
benchmark, engine, fixture, CLI, adapter, lockfile and native executable bytes.
Those hashes were independently recomputed after final implementation. Keeping
the Node process alive reduced observed median orchestration time by 51.94%,
with all native checks retained. The [methodology](metaharness-cpu.md) explains
the tiny synthetic workload, raw samples and limits. No accuracy gain, GPU
throughput, large-checkpoint speedup or model-inference result is established.

## Explicitly unrun capabilities

Claude Code is not installed in the validation environment. Its plugin loader/
strict validator and an actual Claude/Ruflo agent session were not run. The new
module was tested through its real stdio MCP wire and generated launcher/profile.
The operator must perform host-specific acceptance using the documented plugin
directory and their existing host permissions. No provider credentials are
required by the module itself.

No model downloads, paid GPU jobs, training, DPO execution, activation capture,
fitted lenses, production activation or independent evaluator certification were
performed. Training plans remain informational and explicitly unexecuted.
