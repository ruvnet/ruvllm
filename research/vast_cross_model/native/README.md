# Linux native worker preparation

This helper moves the existing native weight engine to a separately measured Linux x86_64 runtime. It provisions no GPU. Linux compilation remains a worker acceptance test. The 0.8B sparse patch is restricted to its exact original checkpoint.

The public base is [RuVector commit 6a879a7](https://github.com/ruvnet/ruvector/commit/6a879a726b0d1d63e70a1ddb656263d8e6bab8a0). The local reviewed changes are exactly `f5c3444968ede44e43744460192198962ae94da2`, tree `f0a37eeeaf7c515ca6724057536e308112d4ea75`, and the patch must match the previous 0.1.1 research bundle. Package construction reads committed Git objects, not local working files. It preserves original workspace manifests and lockfile. Only root Cargo files, public license/readme/configuration, crates, patched dependencies and explicit workspace example members are included. No credentials, model weights, private host state or Mac executable are copied.

Requirements: Python 3.11 or newer, Git on the preparation host, Linux x86_64 on the worker, a C/C++ compiler and binutils, pkg-config, and Rust/Cargo 1.97.1. The workspace declares Rust 1.77, but the selected locked dependencies declare requirements up to 1.85. That is not a tested minimum. Use the same 1.97.1 release as the prior Mac build. Its [official release manifest](https://static.rust-lang.org/dist/channel-rust-1.97.1.toml) is dated 2026-07-16, SHA256 `03569b1886ceb5c05276b50c8431ab111de944cd6140fe1fa7d821dd8e0f29cf`.

`bootstrap-requirements.json` pins that manifest and the three official Rust component URLs and SHA256 values. Its own SHA256 is `37ddb481023ffd0614367fda31ae75d6e1d07592dd7a34086e2f349590640ac9`, checked by the helper before use. `bootstrap` downloads cargo, rustc and the x86_64 standard library to the work root, verifies each digest and its entry in the pinned release manifest, safely extracts bounded regular files and directories, then runs each verified official `install.sh` into a new `/workspace/rust`. It never pipes network content to a shell, changes HOME, or replaces an existing Rust prefix. It records downloaded artifact, installer, log and installed binary hashes. The installer code is trusted through the official artifact digests; these digests must be retained in the trusted worker entrypoint. If `cc`, `c++`, `ar` or `pkg-config` is absent, bootstrap fails and names the Ubuntu packages `build-essential pkg-config ca-certificates` for the infrastructure owner to install.

The native feature selection is `async-runtime,weight-evolution` with defaults disabled. This CPU file editor needs neither CUDA nor Metal. Cargo uses its sparse registry transport and the unchanged lockfile. Dependencies download from their pinned public registry sources on the worker. Its binary digest differs from the Mac digest and is recorded independently. Four compiler jobs bound parallel memory use; the release profile retains the source workspace's optimization and LTO settings.

All installer, compiler, Git and native subprocesses receive an explicit environment allowlist. Cloud lifecycle keys, model registry tokens and provider credentials are excluded. The worker supervisor must also launch the helper with a clean environment. This is credential propagation control; it does not claim isolation from a hostile process with the same operating system identity.

Prepare locally:

```sh
VAST_NATIVE_PYTHON=.venv/bin/python bash scripts/vast_native_prepare.sh package \
  --source-repo /Users/cohen/Projects/ruvector-neural-evolution \
  --bundle dist/governed-neural-evolution-0.1.1 \
  --output runtime/vast-cross-model/native-source.tar.gz
```

Copy only this source archive, its digest receipt, these helper files and the separately approved public benchmark inputs to the worker. Obtain the archive digest from the local receipt over the existing trusted control channel. Hashes inside a replaceable archive are not independent trust anchors.

On the worker, substitute the externally supplied archive SHA256:

```sh
bash scripts/vast_native_prepare.sh extract --archive /workspace/native-source.tar.gz \
  --expected-sha256 VERIFIED_SHA256 --work-root /workspace/ruflo-code-model/runtime/vast-cross-model
bash scripts/vast_native_prepare.sh bootstrap --rust-prefix /workspace/rust \
  --work-root /workspace/ruflo-code-model/runtime/vast-cross-model
bash scripts/vast_native_prepare.sh build --jobs 4 \
  --rust-prefix /workspace/rust --work-root /workspace/ruflo-code-model/runtime/vast-cross-model
```

The build requires exactly 17 focused native unit tests to pass with zero failures and zero ignored tests, then builds the optimized example. It verifies every projected source file before and after Cargo, checks the source manifest against the preparation digest, and writes separate logs, source bindings and binary hashes under `native/`. `artifacts-build.json`, `artifacts-patch.json` and `artifacts-rollback.json` enumerate the small evidence files available at each completed stage; Cargo caches and model checkpoints are excluded from those inventories. No source manifest is changed to make the build pass. The source projection's hash proves its recorded files, not the omitted parts of the public repository.

Put the exact patch and policy in `inputs/patch.json` and `inputs/policy.json` under that work root. The helper accepts only the original whole checkpoint SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, the already published patch, and its frozen policy:

```sh
bash scripts/vast_native_prepare.sh apply --work-root /workspace/ruflo-code-model/runtime/vast-cross-model \
  --source-checkpoint /absolute/path/to/original-checkpoint.safetensors
bash scripts/vast_native_prepare.sh rollback --work-root /workspace/ruflo-code-model/runtime/vast-cross-model \
  --source-checkpoint /absolute/path/to/original-checkpoint.safetensors
```

`apply` produces the same filename under `models/qwen035-08b-edited/` and raw native receipt at `native/patch-receipt.json`. `rollback` writes `models/qwen035-08b-restored/` and `native/rollback-receipt.json`. New outputs are required. Model configuration/tokenizer metadata are the benchmark runner's responsibility. Whole SHA256 and `cmp` must restore the original checkpoint bytes; the original remains unchanged. The helper refuses any other model rather than suggesting that a published sparse edit generalizes to unrelated weights.

Acceptance: Linux build logs show all 17 focused tests passing, the edited checkpoint SHA256 is `bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5`, and rollback's whole SHA256 equals the original with a successful byte comparison. These mechanical checks do not imply improved model accuracy or activation authority.
