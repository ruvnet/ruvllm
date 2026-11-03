//! Bounded offline checkpoint edits. This module never activates a model.
//!
//! Patch validity and RVF integrity are not Core Memory authorization. An
//! independent policy is required for each edit. Original scalar bits and full
//! checkpoint hashes make rollback exact; LoRA reset is not used for rollback.
use half::bf16;
use memmap2::Mmap;
use rvf_forge_core::{ContainerBuilder, SegmentType, VerifyOptions};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const MAX_JSON: u64 = 128 * 1024 * 1024;
const MAX_CONTROL_JSON: u64 = 1024 * 1024;
const MAX_CHECKPOINT: u64 = 64 * 1024 * 1024 * 1024;
const MAX_EDITS: usize = 1_000_000;
const MAX_RVF: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub tensor: String,
    pub index: usize,
    pub old_bits: u32,
    pub new_bits: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub schema: String,
    pub source_sha256: String,
    pub edits: Vec<Edit>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedRange {
    pub tensor: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub allowed_tensors: Vec<String>,
    pub max_changed_values: usize,
    pub max_l2: f64,
    pub max_abs_delta: f64,
    #[serde(default)]
    pub max_relative_l2: Option<f64>,
    #[serde(default)]
    pub protected: Vec<ProtectedRange>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchReceipt {
    pub schema: String,
    pub source_sha256: String,
    pub candidate_sha256: String,
    pub patch_sha256: String,
    pub policy_sha256: String,
    pub changed_values: usize,
    pub l2: f64,
    pub max_abs_delta: f64,
    pub source_l2: f64,
    pub relative_l2: f64,
    pub write_runs: usize,
    pub output: String,
    pub activation_authorized: bool,
}

/// Pins must match for every cache entry and every distributed inference stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationBinding {
    pub model_generation: u64,
    pub model_sha256: String,
    pub patch_sha256: String,
    pub runtime_sha256: String,
    pub topology_sha256: String,
    pub protocol_sha256: String,
    pub toolchain_sha256: String,
}
impl GenerationBinding {
    pub fn validate(&self) -> Result<()> {
        for hash in [
            &self.model_sha256,
            &self.patch_sha256,
            &self.runtime_sha256,
            &self.topology_sha256,
            &self.protocol_sha256,
            &self.toolchain_sha256,
        ] {
            valid_hash(hash)?;
        }
        Ok(())
    }
    pub fn require_match(&self, observed: &Self) -> Result<()> {
        self.validate()?;
        observed.validate()?;
        ensure(self == observed, "stale or mixed model generation")
    }
}

/// An adapter for KV, prompt and recurrent-state caches. A generation change
/// drops every cached value before the new identity can be used. Callers must
/// hold their serving lock across bind and the corresponding backend reload.
pub struct GenerationCache<T> {
    binding: Option<GenerationBinding>,
    entries: BTreeMap<String, T>,
}
impl<T> Default for GenerationCache<T> {
    fn default() -> Self {
        Self {
            binding: None,
            entries: BTreeMap::new(),
        }
    }
}
impl<T> GenerationCache<T> {
    pub fn bind(&mut self, binding: GenerationBinding) -> Result<usize> {
        binding.validate()?;
        let removed = if self.binding.as_ref() != Some(&binding) {
            let count = self.entries.len();
            self.entries.clear();
            count
        } else {
            0
        };
        self.binding = Some(binding);
        Ok(removed)
    }
    pub fn insert(&mut self, binding: &GenerationBinding, key: String, value: T) -> Result<()> {
        self.binding
            .as_ref()
            .ok_or("cache generation is unbound")?
            .require_match(binding)?;
        ensure(key.len() <= 4096, "cache key exceeds resource limit")?;
        ensure(
            self.entries.contains_key(&key) || self.entries.len() < 4096,
            "generation cache is full",
        )?;
        self.entries.insert(key, value);
        Ok(())
    }
    pub fn get(&self, binding: &GenerationBinding, key: &str) -> Result<Option<&T>> {
        self.binding
            .as_ref()
            .ok_or("cache generation is unbound")?
            .require_match(binding)?;
        Ok(self.entries.get(key))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReceipt {
    pub schema: String,
    pub baseline: GenerationBinding,
    pub candidate: GenerationBinding,
    pub suite_sha256: String,
    pub protocol_sha256: String,
    pub baseline_output_sha256: String,
    pub candidate_output_sha256: String,
    pub evaluator_identity: String,
    pub proposer_identity: String,
    pub total: u64,
    pub baseline_correct: u64,
    pub candidate_correct: u64,
    pub baseline_truncated: u64,
    pub candidate_truncated: u64,
}

/// This validates a receipt contract, not the evaluator's honesty or authority.
pub fn validate_evaluation(e: &EvaluationReceipt, r: &PatchReceipt) -> Result<()> {
    ensure(
        r.schema == "ruvllm.weight_receipt.v1" && !r.activation_authorized,
        "invalid patch receipt posture",
    )?;
    ensure(
        e.schema == "ruvllm.weight_evaluation.v1",
        "unsupported evaluation schema",
    )?;
    e.baseline.validate()?;
    e.candidate.validate()?;
    for hash in [
        &e.suite_sha256,
        &e.protocol_sha256,
        &e.baseline_output_sha256,
        &e.candidate_output_sha256,
    ] {
        valid_hash(hash)?;
    }
    ensure(
        e.baseline.model_sha256 == r.source_sha256
            && e.candidate.model_sha256 == r.candidate_sha256
            && e.candidate.patch_sha256 == r.patch_sha256,
        "evaluation is not bound to this patch",
    )?;
    ensure(
        e.baseline.runtime_sha256 == e.candidate.runtime_sha256
            && e.baseline.topology_sha256 == e.candidate.topology_sha256
            && e.baseline.protocol_sha256 == e.protocol_sha256
            && e.candidate.protocol_sha256 == e.protocol_sha256
            && e.baseline.toolchain_sha256 == e.candidate.toolchain_sha256,
        "evaluation runtime or topology mismatch",
    )?;
    ensure(
        !e.evaluator_identity.is_empty()
            && !e.proposer_identity.is_empty()
            && e.evaluator_identity.len() <= 512
            && e.proposer_identity.len() <= 512
            && e.evaluator_identity != e.proposer_identity,
        "independent evaluator identity required",
    )?;
    ensure(
        e.total > 0
            && [
                e.baseline_correct,
                e.candidate_correct,
                e.baseline_truncated,
                e.candidate_truncated,
            ]
            .iter()
            .all(|n| *n <= e.total),
        "invalid evaluation counts",
    )
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    CheckGeneration {
        expected: GenerationBinding,
        observed: Vec<GenerationBinding>,
    },
    Inspect {
        source: PathBuf,
        tensors: Vec<String>,
    },
    Patch {
        source: PathBuf,
        output: PathBuf,
        patch: PathBuf,
        policy: PathBuf,
    },
    Rollback {
        source: PathBuf,
        output: PathBuf,
        patch: PathBuf,
        policy: PathBuf,
        receipt: PathBuf,
    },
    Evaluate {
        evaluation: PathBuf,
        receipt: PathBuf,
    },
    Export {
        patch: PathBuf,
        policy: PathBuf,
        receipt: PathBuf,
        evaluation: Option<PathBuf>,
        governance_evidence: Option<PathBuf>,
        output: PathBuf,
    },
    Verify {
        artifact: PathBuf,
        expected_sha256: String,
    },
}

pub fn execute(request: Request) -> Result<serde_json::Value> {
    match request {
        Request::CheckGeneration { expected, observed } => {
            ensure(
                !observed.is_empty() && observed.len() <= 1024,
                "expected 1..1024 worker bindings",
            )?;
            for binding in &observed {
                expected.require_match(binding)?;
            }
            Ok(
                serde_json::json!({"consistent":true,"workers":observed.len(),"binding":expected,"activation_authorized":false}),
            )
        }
        Request::Inspect { source, tensors } => inspect(&source, &tensors),
        Request::Patch {
            source,
            output,
            patch,
            policy,
        } => Ok(serde_json::to_value(apply(
            &source, &output, &patch, &policy, None,
        )?)?),
        Request::Rollback {
            source,
            output,
            patch,
            policy,
            receipt,
        } => {
            let r = serde_json::from_slice(&read_bounded(&receipt, MAX_CONTROL_JSON)?)?;
            Ok(serde_json::to_value(apply(
                &source,
                &output,
                &patch,
                &policy,
                Some(&r),
            )?)?)
        }
        Request::Evaluate {
            evaluation,
            receipt,
        } => {
            let e = serde_json::from_slice(&read_bounded(&evaluation, MAX_CONTROL_JSON)?)?;
            let r = serde_json::from_slice(&read_bounded(&receipt, MAX_CONTROL_JSON)?)?;
            validate_evaluation(&e, &r)?;
            Ok(
                serde_json::json!({"valid":true,"activation_authorized":false,"candidate_improved":e.candidate_correct>e.baseline_correct && e.candidate_truncated<=e.baseline_truncated}),
            )
        }
        Request::Export {
            patch,
            policy,
            receipt,
            evaluation,
            governance_evidence,
            output,
        } => export(
            &patch,
            &policy,
            &receipt,
            evaluation.as_deref(),
            governance_evidence.as_deref(),
            &output,
        ),
        Request::Verify {
            artifact,
            expected_sha256,
        } => verify(&artifact, &expected_sha256),
    }
}

fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.to_string().into())
    }
}
fn valid_hash(value: &str) -> Result<()> {
    ensure(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "expected lowercase SHA256",
    )
}
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn open_regular(path: &Path, limit: u64) -> Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Inspect the opened handle, not a path that can be swapped between
        // metadata and open. NONBLOCK also prevents a substituted FIFO hanging.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        let metadata = file.metadata()?;
        ensure(metadata.is_file(), "regular nonsymlink file required")?;
        ensure(metadata.len() <= limit, "file exceeds resource limit")?;
        Ok(file)
    }
    #[cfg(not(unix))]
    {
        let _ = (path, limit);
        Err("weight evolution requires Unix nofollow file handles".into())
    }
}
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = open_regular(path, limit)?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure(
        bytes.len() as u64 <= limit,
        "file grew beyond resource limit",
    )?;
    Ok(bytes)
}

struct Stage {
    path: PathBuf,
    file: File,
}
impl Stage {
    fn new(output: &Path) -> Result<Self> {
        ensure(!output.try_exists()?, "output already exists")?;
        // symlink_metadata also detects a dangling symlink, which try_exists does not.
        ensure(
            fs::symlink_metadata(output).is_err(),
            "output entry already exists",
        )?;
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        ensure(parent.is_dir(), "output parent directory missing")?;
        let path = parent.join(format!(".weight-evolution-{}.tmp", uuid::Uuid::new_v4()));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        Ok(Self {
            file: options.open(&path)?,
            path,
        })
    }
    fn publish(&mut self, output: &Path) -> Result<()> {
        self.file.sync_all()?;
        // Linking a same-directory temporary file is atomic and never overwrites.
        fs::hard_link(&self.path, output)?;
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(parent)?.sync_all()?;
        Ok(())
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn copy_and_hash(source: &Path, destination: &mut File) -> Result<String> {
    let mut file = open_regular(source, MAX_CHECKPOINT)?;
    let mut buffer = vec![0; 1024 * 1024];
    let mut hash = Sha256::new();
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        size += n as u64;
        ensure(size <= MAX_CHECKPOINT, "checkpoint grew beyond limit")?;
        hash.update(&buffer[..n]);
        destination.write_all(&buffer[..n])?;
    }
    destination.flush()?;
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone)]
struct TensorLayout {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    len: usize,
}
fn layouts(map: &[u8]) -> Result<BTreeMap<String, TensorLayout>> {
    ensure(map.len() >= 8, "truncated safetensors")?;
    let header = u64::from_le_bytes(map[..8].try_into()?);
    ensure(
        header <= 64 * 1024 * 1024,
        "safetensors header exceeds 64 MiB",
    )?;
    let tensors = SafeTensors::deserialize(map)?;
    let mut result = BTreeMap::new();
    for (name, tensor) in tensors.tensors() {
        let start = tensor.data().as_ptr() as usize - map.as_ptr() as usize;
        result.insert(
            name,
            TensorLayout {
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                start,
                len: tensor.data().len(),
            },
        );
    }
    Ok(result)
}
fn width(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        _ => Err("only BF16 and F32 tensors may be edited".into()),
    }
}
fn scalar(dtype: Dtype, bits: u32) -> Result<f64> {
    let value = match dtype {
        Dtype::BF16 => {
            ensure(bits <= u16::MAX as u32, "BF16 bits overflow")?;
            bf16::from_bits(bits as u16).to_f32()
        }
        Dtype::F32 => f32::from_bits(bits),
        _ => return Err("unsupported scalar dtype".into()),
    };
    ensure(value.is_finite(), "nonfinite edited scalar")?;
    Ok(value as f64)
}

fn validate_policy(patch: &Patch, policy: &Policy) -> Result<()> {
    ensure(
        patch.schema == "ruvllm.weight_patch.v1",
        "unsupported patch schema",
    )?;
    valid_hash(&patch.source_sha256)?;
    ensure(
        !patch.edits.is_empty()
            && patch.edits.len() <= MAX_EDITS
            && patch.edits.len() <= policy.max_changed_values,
        "changed value budget exceeded or empty patch",
    )?;
    ensure(
        policy.max_changed_values <= MAX_EDITS
            && policy.max_l2.is_finite()
            && policy.max_l2 >= 0.0
            && policy.max_abs_delta.is_finite()
            && policy.max_abs_delta >= 0.0,
        "invalid policy limits",
    )?;
    if let Some(limit) = policy.max_relative_l2 {
        ensure(
            limit.is_finite() && limit >= 0.0,
            "invalid relative norm limit",
        )?;
    }
    let allowed: BTreeSet<_> = policy.allowed_tensors.iter().collect();
    ensure(
        !allowed.is_empty()
            && allowed.len() <= 256
            && allowed.len() == policy.allowed_tensors.len()
            && allowed.iter().all(|s| !s.is_empty() && s.len() <= 512),
        "invalid allowed tensor set",
    )?;
    ensure(policy.protected.len() <= 4096, "too many protected ranges")?;
    let mut protected: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for range in &policy.protected {
        ensure(
            range.start < range.end && allowed.contains(&range.tensor),
            "invalid protected range",
        )?;
        protected
            .entry(&range.tensor)
            .or_default()
            .push((range.start, range.end));
    }
    for ranges in protected.values_mut() {
        ranges.sort_unstable();
        ensure(
            ranges.windows(2).all(|w| w[0].1 <= w[1].0),
            "overlapping protected ranges",
        )?;
    }
    let mut seen = BTreeSet::new();
    for edit in &patch.edits {
        ensure(
            allowed.contains(&edit.tensor),
            "tensor outside independent policy",
        )?;
        ensure(
            seen.insert((&edit.tensor, edit.index)),
            "duplicate scalar edit",
        )?;
        ensure(edit.old_bits != edit.new_bits, "no-op scalar edit")?;
        if let Some(ranges) = protected.get(edit.tensor.as_str()) {
            let i = ranges.partition_point(|r| r.0 <= edit.index);
            ensure(
                i == 0 || edit.index >= ranges[i - 1].1,
                "protected scalar edit",
            )?;
        }
    }
    Ok(())
}

pub fn apply(
    source: &Path,
    output: &Path,
    patch_path: &Path,
    policy_path: &Path,
    rollback: Option<&PatchReceipt>,
) -> Result<PatchReceipt> {
    let patch_bytes = read_bounded(patch_path, MAX_JSON)?;
    let policy_bytes = read_bounded(policy_path, MAX_CONTROL_JSON)?;
    let patch: Patch = serde_json::from_slice(&patch_bytes)?;
    let policy: Policy = serde_json::from_slice(&policy_bytes)?;
    validate_policy(&patch, &policy)?;
    let patch_sha256 = sha256(&patch_bytes);
    let policy_sha256 = sha256(&policy_bytes);
    if let Some(receipt) = rollback {
        validate_receipt(receipt, &patch, &patch_sha256, &policy_sha256)?;
    }
    let expected = rollback
        .map(|r| r.candidate_sha256.as_str())
        .unwrap_or(&patch.source_sha256);
    let mut stage = Stage::new(output)?;
    let source_sha256 = copy_and_hash(source, &mut stage.file)?;
    ensure(
        source_sha256 == expected,
        "source checkpoint digest mismatch",
    )?;
    let (mut writes, l2, max_abs_delta, source_l2) = {
        // The mapping is private to this newly created staging file. It is
        // dropped before writes; the original checkpoint is never mapped mutable.
        let map = unsafe { Mmap::map(&stage.file)? };
        let tensors = layouts(&map)?;
        for name in &policy.allowed_tensors {
            ensure(tensors.contains_key(name), "allowed tensor absent")?;
        }
        for range in &policy.protected {
            let t = &tensors[&range.tensor];
            ensure(
                range.end <= t.len / width(t.dtype)?,
                "protected range outside tensor",
            )?;
        }
        let mut writes = Vec::with_capacity(patch.edits.len());
        let mut squared = 0.0f64;
        let mut largest = 0.0f64;
        let mut original_squares = 0.0f64;
        for name in &policy.allowed_tensors {
            let t = &tensors[name];
            let size = width(t.dtype)?;
            for bytes in map[t.start..t.start + t.len].chunks_exact(size) {
                let mut bits = [0; 4];
                bits[..size].copy_from_slice(bytes);
                let value = scalar(t.dtype, u32::from_le_bytes(bits))?;
                original_squares += value * value;
            }
        }
        for edit in &patch.edits {
            let tensor = tensors.get(&edit.tensor).ok_or("tensor missing")?;
            let size = width(tensor.dtype)?;
            ensure(edit.index < tensor.len / size, "scalar index out of range")?;
            let offset = tensor
                .start
                .checked_add(edit.index.checked_mul(size).ok_or("index overflow")?)
                .ok_or("offset overflow")?;
            let old = scalar(tensor.dtype, edit.old_bits)?;
            let new = scalar(tensor.dtype, edit.new_bits)?;
            let delta = (new - old).abs();
            if rollback.is_some() {
                original_squares += old * old - new * new;
            }
            largest = largest.max(delta);
            squared += delta * delta;
            let (before, after) = if rollback.is_some() {
                (edit.new_bits, edit.old_bits)
            } else {
                (edit.old_bits, edit.new_bits)
            };
            ensure(
                map[offset..offset + size] == before.to_le_bytes()[..size],
                "original scalar bits mismatch",
            )?;
            writes.push((offset, size, after));
        }
        ensure(
            squared.is_finite()
                && squared.sqrt() <= policy.max_l2
                && largest <= policy.max_abs_delta,
            "patch norm budget exceeded",
        )?;
        ensure(
            original_squares.is_finite() && original_squares > 0.0,
            "zero or invalid original tensor norm",
        )?;
        (writes, squared.sqrt(), largest, original_squares.sqrt())
    };
    let relative_l2 = l2 / source_l2;
    if let Some(limit) = policy.max_relative_l2 {
        ensure(relative_l2 <= limit, "relative norm budget exceeded")?;
    }
    // Coalesce adjacent scalar replacements: published dense-column edits can
    // otherwise issue hundreds of thousands of seek/write syscalls.
    writes.sort_unstable_by_key(|w| w.0);
    let mut write_runs = 0;
    let mut cursor = 0;
    while cursor < writes.len() {
        let start = writes[cursor].0;
        let mut block = Vec::new();
        while cursor < writes.len() && writes[cursor].0 == start + block.len() {
            let (_, size, after) = writes[cursor];
            block.extend_from_slice(&after.to_le_bytes()[..size]);
            cursor += 1;
        }
        stage.file.seek(SeekFrom::Start(start as u64))?;
        stage.file.write_all(&block)?;
        write_runs += 1;
    }
    stage.file.flush()?;
    stage.file.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = stage.file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let candidate_sha256 = format!("{:x}", hash.finalize());
    if rollback.is_some() {
        ensure(
            candidate_sha256 == patch.source_sha256,
            "rollback did not restore original checkpoint digest",
        )?;
    }
    stage.publish(output)?;
    Ok(PatchReceipt {
        schema: if rollback.is_some() {
            "ruvllm.weight_rollback.v1"
        } else {
            "ruvllm.weight_receipt.v1"
        }
        .into(),
        source_sha256,
        candidate_sha256,
        patch_sha256,
        policy_sha256,
        changed_values: patch.edits.len(),
        l2,
        max_abs_delta,
        source_l2,
        relative_l2,
        write_runs,
        output: output.to_string_lossy().into(),
        activation_authorized: false,
    })
}

fn validate_receipt(
    r: &PatchReceipt,
    patch: &Patch,
    patch_hash: &str,
    policy_hash: &str,
) -> Result<()> {
    ensure(
        r.schema == "ruvllm.weight_receipt.v1" && !r.activation_authorized,
        "invalid patch receipt posture",
    )?;
    valid_hash(&r.candidate_sha256)?;
    ensure(
        r.source_sha256 == patch.source_sha256
            && r.patch_sha256 == patch_hash
            && r.policy_sha256 == policy_hash
            && r.changed_values == patch.edits.len()
            && r.write_runs > 0
            && r.write_runs <= r.changed_values,
        "receipt identity mismatch",
    )?;
    ensure(
        r.l2.is_finite()
            && r.l2 >= 0.0
            && r.max_abs_delta.is_finite()
            && r.max_abs_delta >= 0.0
            && r.source_l2.is_finite()
            && r.source_l2 > 0.0
            && r.relative_l2.is_finite()
            && (r.relative_l2 - r.l2 / r.source_l2).abs() < 1e-12,
        "invalid receipt norms",
    )
}

pub fn inspect(source: &Path, requested: &[String]) -> Result<serde_json::Value> {
    ensure(
        !requested.is_empty() && requested.len() <= 256,
        "inspect selects 1..256 tensors",
    )?;
    // Snapshot before inspection prevents mutable source mappings and binds the
    // reported statistics to the hash actually inspected.
    let target = std::env::temp_dir().join(format!("ruvllm-inspect-{}", uuid::Uuid::new_v4()));
    let mut stage = Stage::new(&target)?;
    let hash = copy_and_hash(source, &mut stage.file)?;
    let map = unsafe { Mmap::map(&stage.file)? };
    let tensors = layouts(&map)?;
    let mut results = Vec::new();
    for name in requested {
        let t = tensors.get(name).ok_or("requested tensor absent")?;
        let size = width(t.dtype)?;
        let data = &map[t.start..t.start + t.len];
        let mut sum = 0.0;
        let mut sq = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut nonfinite = 0usize;
        for bytes in data.chunks_exact(size) {
            let mut bits = [0; 4];
            bits[..size].copy_from_slice(bytes);
            match scalar(t.dtype, u32::from_le_bytes(bits)) {
                Ok(v) => {
                    sum += v;
                    sq += v * v;
                    min = min.min(v);
                    max = max.max(v);
                }
                Err(_) => nonfinite += 1,
            }
        }
        let count = t.len / size;
        results.push(serde_json::json!({"name":name,"dtype":format!("{:?}", t.dtype),"shape":t.shape,"count":count,"nonfinite":nonfinite,
            "mean":if count>nonfinite {Some(sum/(count-nonfinite) as f64)} else {None},"l2":sq.sqrt(),"min":if min.is_finite(){Some(min)}else{None},"max":if max.is_finite(){Some(max)}else{None},"sha256":sha256(data)}));
    }
    Ok(
        serde_json::json!({"schema":"ruvllm.weight_inspection.v1","source_sha256":hash,"tensors":results,"activation_statistics_available":false}),
    )
}

pub fn export(
    patch_path: &Path,
    policy_path: &Path,
    receipt_path: &Path,
    evaluation_path: Option<&Path>,
    governance_evidence_path: Option<&Path>,
    output: &Path,
) -> Result<serde_json::Value> {
    let pbytes = read_bounded(patch_path, MAX_JSON)?;
    let policy_bytes = read_bounded(policy_path, MAX_CONTROL_JSON)?;
    let receipt_bytes = read_bounded(receipt_path, MAX_CONTROL_JSON)?;
    let p: Patch = serde_json::from_slice(&pbytes)?;
    let policy: Policy = serde_json::from_slice(&policy_bytes)?;
    let receipt: PatchReceipt = serde_json::from_slice(&receipt_bytes)?;
    validate_policy(&p, &policy)?;
    validate_receipt(&receipt, &p, &sha256(&pbytes), &sha256(&policy_bytes))?;
    ensure(
        receipt.changed_values <= policy.max_changed_values
            && receipt.l2 <= policy.max_l2
            && receipt.max_abs_delta <= policy.max_abs_delta
            && policy
                .max_relative_l2
                .map_or(true, |bound| receipt.relative_l2 <= bound),
        "receipt exceeds independent policy",
    )?;
    let mut payload_bytes = pbytes.len() + policy_bytes.len() + receipt_bytes.len();
    ensure(
        payload_bytes as u64 <= MAX_RVF - 8192,
        "RVF payloads exceed resource limit",
    )?;
    let mut builder = ContainerBuilder::new()
        .segment(SegmentType::Meta, policy_bytes)
        .segment(SegmentType::Delta, pbytes)
        .segment(SegmentType::Witness, receipt_bytes);
    if let Some(path) = evaluation_path {
        let bytes = read_bounded(path, MAX_CONTROL_JSON)?;
        payload_bytes += bytes.len();
        ensure(
            payload_bytes as u64 <= MAX_RVF - 8192,
            "RVF payloads exceed resource limit",
        )?;
        validate_evaluation(&serde_json::from_slice(&bytes)?, &receipt)?;
        builder = builder.segment(SegmentType::Witness, bytes);
    }
    let governance_evidence_sha256 = if let Some(path) = governance_evidence_path {
        let bytes = read_bounded(path, MAX_CONTROL_JSON)?;
        let _: serde_json::Value = serde_json::from_slice(&bytes)?;
        let hash = sha256(&bytes);
        let witness = GovernanceWitness {
            schema: "ruvllm.governance_evidence.v1".into(),
            sha256: hash.clone(),
            json: String::from_utf8(bytes)?,
            authenticated: false,
        };
        let payload = serde_json::to_vec(&witness)?;
        payload_bytes += payload.len();
        ensure(
            payload_bytes as u64 <= MAX_RVF - 8192,
            "RVF payloads exceed resource limit",
        )?;
        builder = builder.segment(SegmentType::Witness, payload);
        Some(hash)
    } else {
        None
    };
    let artifact = builder.build()?;
    ensure(
        artifact.bytes.len() as u64 <= MAX_RVF,
        "RVF exceeds resource limit",
    )?;
    let hash = sha256(&artifact.bytes);
    let report = rvf_forge_core::verify_bytes(
        &artifact.bytes,
        &VerifyOptions {
            expected_identity: Some(hash.clone()),
            ..Default::default()
        },
    )?
    .into_result()?;
    let mut stage = Stage::new(output)?;
    stage.file.write_all(&artifact.bytes)?;
    stage.publish(output)?;
    Ok(
        serde_json::json!({"schema":"ruvllm.weight_export.v1","sha256":hash,"bytes":artifact.bytes.len(),"segments":report.segment_count,"integrity_verified":true,"activation_authorized":false,"governance_evidence_sha256":governance_evidence_sha256,"governance_evidence_authenticated":false,"output":output}),
    )
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernanceWitness {
    schema: String,
    sha256: String,
    json: String,
    authenticated: bool,
}

pub fn verify(path: &Path, expected: &str) -> Result<serde_json::Value> {
    valid_hash(expected)?;
    let bytes = read_bounded(path, MAX_RVF)?;
    ensure(
        bytes.len() >= 4096
            && &bytes[bytes.len() - 4096..bytes.len() - 4096 + 4] == [0x30, 0x4d, 0x56, 0x52],
        "complete RVF root manifest missing",
    )?;
    let report = rvf_forge_core::verify_bytes(
        &bytes,
        &VerifyOptions {
            expected_identity: Some(expected.into()),
            ..Default::default()
        },
    )?
    .into_result()?;
    let inspection = rvf_forge_core::inspect_bytes(&bytes)?;
    let mut governance_hashes = Vec::new();
    for segment in inspection
        .segments
        .iter()
        .filter(|s| s.segment_type == SegmentType::Witness as u8)
    {
        ensure(
            segment.payload_length <= 3 * MAX_CONTROL_JSON,
            "witness exceeds resource limit",
        )?;
        let start = usize::try_from(segment.offset)?
            .checked_add(64)
            .ok_or("witness offset overflow")?;
        let end = start
            .checked_add(usize::try_from(segment.payload_length)?)
            .ok_or("witness length overflow")?;
        let payload = bytes.get(start..end).ok_or("truncated witness")?;
        let value: serde_json::Value = serde_json::from_slice(payload)?;
        if value["schema"] == "ruvllm.governance_evidence.v1" {
            let witness: GovernanceWitness = serde_json::from_value(value)?;
            ensure(
                witness.json.len() as u64 <= MAX_CONTROL_JSON && !witness.authenticated,
                "invalid governance evidence posture",
            )?;
            valid_hash(&witness.sha256)?;
            ensure(
                sha256(witness.json.as_bytes()) == witness.sha256,
                "governance evidence digest mismatch",
            )?;
            let _: serde_json::Value = serde_json::from_str(&witness.json)?;
            governance_hashes.push(witness.sha256);
        }
    }
    Ok(
        serde_json::json!({"integrity_verified":true,"activation_authorized":false,"governance_evidence_sha256":governance_hashes,"governance_evidence_authenticated":false,"report":report}),
    )
}

#[cfg(test)]
mod tests;
