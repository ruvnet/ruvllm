# Contributing

Use a branch and keep mechanisms, measurements and deployment authority separate.
The native engine is a source extraction with per file provenance. Changes to it
need new provenance and validation; do not silently rewrite historical receipts.

1. Run `python3 scripts/check_repository.py` and `python3 scripts/verify_benchmarks.py`.
2. Run the native tests from `native/` so Rust uses the pinned toolchain.
3. Include a concrete before and after case and tests for the failure mechanism.
4. For model changes, freeze data, model identity, decoding settings, precision,
   budget and selection rules before release evaluation. Publish regressions and
   failed candidates alongside improvements.
5. Label CPU fixtures, local inference and GPU inference as different evidence.
   A valid artifact or passing test never grants model activation authority.

The CI workflow downloads public dependencies and runs CPU tests. It does not
create cloud jobs, use provider credentials, download base model weights or
execute generated benchmark programs on a shared server. The macOS tests exercise
the controlled Seatbelt sandbox with small test fixtures.

Use the caller's own accounts for paid research. Never commit tokens, private
datasets, signing credentials, SSH keys or provider account configuration.
Security reports should follow [SECURITY.md](SECURITY.md).

Acceptance for a code change: the relevant tests pass and a fresh clone verifies
the same public evidence without credentials.
