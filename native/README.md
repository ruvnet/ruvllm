# Native weight evolution

This standalone Rust package inspects Safetensors checkpoints, applies bounded BF16 or F32 edits, restores original bytes, and packages evidence in RVF. It performs no model inference and has no activation operation.

The engine, its 17 focused tests, the JSON CLI, and four RVF source crates are copied unchanged from RuVector source commit `f5c3444968ede44e43744460192198962ae94da2`. `upstream-provenance.json` lists every copied file and SHA256. The standalone manifest and library wrapper are new packaging. The locked registry versions and checksums match the original lockfile. A new binary is not the historical benchmark binary.

1. Install Rust 1.97.1 and Python 3.12 or newer on macOS or Linux. The pinned toolchain is recorded in `rust-toolchain.toml`; Cargo may download public dependencies on the first build.

2. Select the pinned toolchain by entering the native directory, then test and build:

   ```sh
   cd native
   python3 verify_source.py
   cargo test --locked -p ruvllm-weight-evolution --lib
   cargo build --locked --release --bin weight_evolution
   python3 smoke.py
   cd ..
   ```

   The smoke test creates a four value BF16 fixture in a temporary directory. It checks actual inspection, exact candidate bytes, protected edit rejection, overwrite rejection, exact rollback, RVF export and verification, and stale generation rejection. Evaluation counts in this fixture are synthetic contract data. No model is downloaded and no benchmark answer is generated.

3. Send one JSON request through stdin:

   ```sh
   printf '%s\n' '{"op":"inspect","source":"/absolute/checkpoint.safetensors","tensors":["layer.weight"]}' | native/target/release/weight_evolution
   ```

   Replace the path and tensor name with a checkpoint you own. Inspection returns checkpoint and tensor digests plus weight statistics. Activation statistics are not measured.

The CLI operations are `inspect`, `patch`, `rollback`, `evaluate`, `export`, `verify`, and `check_generation`. The exact tagged request schema is the `Request` enum in `src/weight_evolution.rs`. `evaluate` validates a supplied receipt contract; an external model runner must produce the actual inference measurements. `check_generation` compares supplied identities; it does not discover remote workers or attest hardware.

Patch policy explicitly names allowed tensors, protected index ranges, the maximum changed value count, and absolute or relative delta limits. Relative L2 means the delta norm across the allowed tensors divided by the original norm across that same complete allowed tensor set. Sparse changed value counts alone do not mean a small relative change in selected tensors.

The engine preserves source bytes, rejects existing output paths, verifies full checkpoint hashes, checks original scalar bits, and publishes through an atomic temporary file. It accepts bounded regular files through Unix file handles that reject symlinks and avoid FIFO blocking. The output directory and the operator account remain trusted. These checks do not protect against the same operator maliciously replacing parent directories between validation and use. Unsupported non Unix file access fails closed.

The standalone engine and RVF source use the retained upstream MIT notice in `LICENSE`; the RVF format crates also declare an Apache 2.0 option. The `THIRD_PARTY_NOTICES.md` file identifies each component and its scope. This repository does not relicense model checkpoints, research patches, or the Python environment.

Acceptance: all 17 focused Rust tests pass, and `python3 native/smoke.py` returns `passed: true`, equal source and restored SHA256 values, `fixture_only: true`, and `production_activation: false`.
