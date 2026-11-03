# Native source attribution

The weight evolution module, its tests and CLI, and the `rvf-forge-core`, `rvf-types`, `rvf-wire` and `rvf-crypto` source trees originate in [RuVector](https://github.com/ruvnet/ruvector). The local audited source commit is `f5c3444968ede44e43744460192198962ae94da2`, based on `6a879a726b0d1d63e70a1ddb656263d8e6bab8a0`. Exact source paths and hashes are in `upstream-provenance.json`. The source is supplied directly; availability of the local commit from an upstream Git server is not assumed.

The copied RuVector root MIT notice is retained unchanged in `LICENSE`, copyright 2025 rUv. `rvf-types`, `rvf-wire`, and `rvf-crypto` declare `MIT OR Apache-2.0`; this source distribution uses the MIT option with that upstream notice. Each copied Cargo manifest retains its original license metadata and repository attribution.

The standalone Cargo manifest, wrapper, smoke test and documentation are new packaging under the same MIT notice. Registry dependencies are fetched separately by Cargo and remain subject to their own declared licenses. `registry-dependencies.json` records their names, versions, original registry checksums and license expressions after the locked resolution. No registry source or compiled dependency binary is redistributed by this native source directory.

The original benchmark executable has a separate, larger dependency closure. Its historical notice index must not be represented as the exact closure of this smaller standalone package. Generated native executables and `target` artifacts are excluded from Git.
