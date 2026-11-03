use super::*;
use tempfile::TempDir;

fn checkpoint(dtype: &str, values: &[u32]) -> Vec<u8> {
    let size = if dtype == "BF16" { 2 } else { 4 };
    let mut header = serde_json::to_vec(&serde_json::json!({"layer.weight":{"dtype":dtype,"shape":[values.len()],"data_offsets":[0,values.len()*size]}})).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    for value in values {
        bytes.extend(&value.to_le_bytes()[..size]);
    }
    bytes
}
struct Fixture {
    dir: TempDir,
    original: Vec<u8>,
    patch: Patch,
    policy: Policy,
}
impl Fixture {
    fn new() -> Self {
        let original = checkpoint("BF16", &[0x3f80, 0x4000, 0x4040, 0x4080]);
        let patch = Patch {
            schema: "ruvllm.weight_patch.v1".into(),
            source_sha256: sha256(&original),
            edits: vec![Edit {
                tensor: "layer.weight".into(),
                index: 1,
                old_bits: 0x4000,
                new_bits: 0x4020,
            }],
        };
        let policy = Policy {
            allowed_tensors: vec!["layer.weight".into()],
            max_changed_values: 1,
            max_l2: 1.0,
            max_abs_delta: 1.0,
            max_relative_l2: Some(1.0),
            protected: vec![],
        };
        let f = Self {
            dir: tempfile::tempdir().unwrap(),
            original,
            patch,
            policy,
        };
        f.write();
        f
    }
    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
    fn write(&self) {
        fs::write(self.path("base"), &self.original).unwrap();
        fs::write(
            self.path("patch.json"),
            serde_json::to_vec(&self.patch).unwrap(),
        )
        .unwrap();
        fs::write(
            self.path("policy.json"),
            serde_json::to_vec(&self.policy).unwrap(),
        )
        .unwrap();
    }
    fn apply(&self) -> Result<PatchReceipt> {
        apply(
            &self.path("base"),
            &self.path("candidate"),
            &self.path("patch.json"),
            &self.path("policy.json"),
            None,
        )
    }
    fn reject(&self) {
        self.write();
        assert!(self.apply().is_err());
        assert!(!self.path("candidate").exists());
        assert_eq!(fs::read(self.path("base")).unwrap(), self.original);
        assert!(fs::read_dir(self.dir.path()).unwrap().all(|x| !x
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".weight-evolution")));
    }
}

#[test]
fn bf16_patch_preserves_layout_and_rolls_back_exactly() {
    let f = Fixture::new();
    let r = f.apply().unwrap();
    assert_eq!(r.changed_values, 1);
    assert_eq!(r.l2, 0.5);
    assert!((r.source_l2 - 30f64.sqrt()).abs() < 1e-12);
    assert!(!r.activation_authorized);
    let actual = fs::read(f.path("candidate")).unwrap();
    assert_eq!(actual.len(), f.original.len());
    assert_eq!(
        actual
            .iter()
            .zip(&f.original)
            .filter(|(a, b)| a != b)
            .count(),
        1
    );
    let rollback = apply(
        &f.path("candidate"),
        &f.path("restored"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        Some(&r),
    )
    .unwrap();
    assert_eq!(rollback.candidate_sha256, f.patch.source_sha256);
    assert_eq!(fs::read(f.path("restored")).unwrap(), f.original);
    assert_eq!(fs::read(f.path("base")).unwrap(), f.original);
}
#[test]
fn f32_patch_roundtrip() {
    let mut f = Fixture::new();
    f.original = checkpoint("F32", &[1f32.to_bits(), 2f32.to_bits()]);
    f.patch.source_sha256 = sha256(&f.original);
    f.patch.edits[0].old_bits = 2f32.to_bits();
    f.patch.edits[0].new_bits = 2.25f32.to_bits();
    f.write();
    let r = f.apply().unwrap();
    assert_eq!(r.l2, 0.25);
    apply(
        &f.path("candidate"),
        &f.path("restored"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        Some(&r),
    )
    .unwrap();
    assert_eq!(fs::read(f.path("restored")).unwrap(), f.original);
}

#[test]
fn adjacent_edits_are_coalesced_into_one_write() {
    let mut f = Fixture::new();
    f.patch.edits = vec![(0, 0x3f80), (1, 0x4000), (2, 0x4040)]
        .into_iter()
        .map(|(index, bits)| Edit {
            tensor: "layer.weight".into(),
            index,
            old_bits: bits,
            new_bits: bits + 0x20,
        })
        .collect();
    f.policy.max_changed_values = 3;
    f.write();
    let receipt = f.apply().unwrap();
    assert_eq!(receipt.changed_values, 3);
    assert_eq!(receipt.write_runs, 1);
    apply(
        &f.path("candidate"),
        &f.path("restored"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        Some(&receipt),
    )
    .unwrap();
    assert_eq!(fs::read(f.path("restored")).unwrap(), f.original);
}
#[test]
fn bounds_and_protection_fail_without_publishing() {
    let mut f = Fixture::new();
    f.policy.max_changed_values = 0;
    f.reject();
    let mut f = Fixture::new();
    f.policy.max_l2 = 0.1;
    f.reject();
    let mut f = Fixture::new();
    f.policy.max_abs_delta = 0.1;
    f.reject();
    let mut f = Fixture::new();
    f.policy.max_relative_l2 = Some(0.01);
    f.reject();
    let mut f = Fixture::new();
    f.policy.allowed_tensors = vec!["other".into()];
    f.reject();
    let mut f = Fixture::new();
    f.policy.protected.push(ProtectedRange {
        tensor: "layer.weight".into(),
        start: 1,
        end: 2,
    });
    f.reject();
    let mut f = Fixture::new();
    f.policy.protected.push(ProtectedRange {
        tensor: "layer.weight".into(),
        start: 5,
        end: 6,
    });
    f.reject();
}
#[test]
fn malformed_edits_fail_without_publishing() {
    let mut f = Fixture::new();
    f.patch.edits[0].index = 4;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits[0].old_bits = 0x3f80;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits[0].new_bits = 0x7f80;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits[0].new_bits = 0x7fc0;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits[0].new_bits = 0x10000;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits.push(f.patch.edits[0].clone());
    f.policy.max_changed_values = 2;
    f.reject();
    let mut f = Fixture::new();
    f.patch.edits[0].new_bits = f.patch.edits[0].old_bits;
    f.reject();
    let mut f = Fixture::new();
    f.patch.source_sha256 = "0".repeat(64);
    f.reject();
}
#[test]
fn malformed_header_and_truncated_tensor_rejected() {
    let mut f = Fixture::new();
    f.original[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    f.patch.source_sha256 = sha256(&f.original);
    f.reject();
    let mut f = Fixture::new();
    f.original.pop();
    f.patch.source_sha256 = sha256(&f.original);
    f.reject();
}
#[test]
fn overwrite_and_corrupt_rollback_rejected() {
    let f = Fixture::new();
    let r = f.apply().unwrap();
    let candidate = fs::read(f.path("candidate")).unwrap();
    assert!(f.apply().is_err());
    assert_eq!(fs::read(f.path("candidate")).unwrap(), candidate);
    let mut corrupted = candidate;
    let n = corrupted.len();
    corrupted[n - 1] ^= 1;
    fs::write(f.path("candidate"), corrupted).unwrap();
    assert!(apply(
        &f.path("candidate"),
        &f.path("restored"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        Some(&r)
    )
    .is_err());
    assert!(!f.path("restored").exists());
}
#[test]
fn receipt_tampering_rejected() {
    let f = Fixture::new();
    let mut r = f.apply().unwrap();
    r.patch_sha256 = "0".repeat(64);
    assert!(apply(
        &f.path("candidate"),
        &f.path("restored"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        Some(&r)
    )
    .is_err());
    assert!(!f.path("restored").exists());
}
#[cfg(unix)]
#[test]
fn symlinks_rejected() {
    let f = Fixture::new();
    std::os::unix::fs::symlink(f.path("missing"), f.path("candidate")).unwrap();
    assert!(f.apply().is_err());
    assert!(read_bounded(&f.path("candidate"), 100).is_err());
    std::os::unix::fs::symlink(f.path("base"), f.path("base-link")).unwrap();
    assert!(read_bounded(&f.path("base-link"), 1024).is_err());
}

#[cfg(unix)]
#[test]
fn fifo_and_directory_inputs_fail_without_blocking() {
    use std::os::unix::ffi::OsStrExt;
    let f = Fixture::new();
    let path = f.path("pipe");
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let start = std::time::Instant::now();
    assert!(read_bounded(&path, 1024).is_err());
    assert!(start.elapsed() < std::time::Duration::from_secs(1));
    assert!(read_bounded(f.dir.path(), 1024).is_err());
    assert!(apply(
        &path,
        &f.path("candidate"),
        &f.path("patch.json"),
        &f.path("policy.json"),
        None
    )
    .is_err());
    assert!(!f.path("candidate").exists());
}
#[test]
fn inspection_reports_tensor_stats_without_activation_claim() {
    let f = Fixture::new();
    let value = inspect(&f.path("base"), &["layer.weight".into()]).unwrap();
    assert_eq!(value["source_sha256"], f.patch.source_sha256);
    assert_eq!(value["tensors"][0]["mean"], 2.5);
    assert_eq!(value["activation_statistics_available"], false);
}
#[test]
fn rvf_is_complete_deterministic_and_detects_tampering() {
    let f = Fixture::new();
    let r = f.apply().unwrap();
    fs::write(f.path("receipt.json"), serde_json::to_vec(&r).unwrap()).unwrap();
    let result = export(
        &f.path("patch.json"),
        &f.path("policy.json"),
        &f.path("receipt.json"),
        None,
        None,
        &f.path("one.rvf"),
    )
    .unwrap();
    export(
        &f.path("patch.json"),
        &f.path("policy.json"),
        &f.path("receipt.json"),
        None,
        None,
        &f.path("two.rvf"),
    )
    .unwrap();
    let mut bytes = fs::read(f.path("one.rvf")).unwrap();
    assert_eq!(bytes, fs::read(f.path("two.rvf")).unwrap());
    assert_eq!(
        &bytes[bytes.len() - 4096..bytes.len() - 4092],
        &[0x30, 0x4d, 0x56, 0x52]
    );
    assert!(verify(&f.path("one.rvf"), result["sha256"].as_str().unwrap()).is_ok());
    bytes[65] ^= 1;
    fs::write(f.path("one.rvf"), &bytes).unwrap();
    assert!(verify(&f.path("one.rvf"), result["sha256"].as_str().unwrap()).is_err());
    assert!(verify(&f.path("one.rvf"), &sha256(&bytes)).is_err());
}
fn binding(x: char) -> GenerationBinding {
    GenerationBinding {
        model_generation: 0,
        model_sha256: x.to_string().repeat(64),
        patch_sha256: "1".repeat(64),
        runtime_sha256: "2".repeat(64),
        topology_sha256: "3".repeat(64),
        protocol_sha256: "5".repeat(64),
        toolchain_sha256: "9".repeat(64),
    }
}
#[test]
fn generation_cache_invalidates_and_stale_workers_fail() {
    let a = binding('a');
    let b = binding('b');
    let mut cache = GenerationCache::default();
    assert!(cache.get(&a, "prompt").is_err());
    cache.bind(a.clone()).unwrap();
    cache.insert(&a, "prompt".into(), vec![1, 2]).unwrap();
    assert_eq!(cache.bind(a.clone()).unwrap(), 0);
    assert!(cache.get(&a, "prompt").unwrap().is_some());
    assert_eq!(cache.bind(b.clone()).unwrap(), 1);
    assert!(cache.get(&a, "prompt").is_err());
    assert!(cache.get(&b, "prompt").unwrap().is_none());
    assert!(execute(Request::CheckGeneration {
        expected: a.clone(),
        observed: vec![a, b]
    })
    .is_err());
}
#[test]
fn evaluation_is_bound_independent_and_does_not_authorize() {
    let f = Fixture::new();
    let r = f.apply().unwrap();
    let mut e = EvaluationReceipt {
        schema: "ruvllm.weight_evaluation.v1".into(),
        baseline: binding('a'),
        candidate: binding('b'),
        suite_sha256: "4".repeat(64),
        protocol_sha256: "5".repeat(64),
        baseline_output_sha256: "6".repeat(64),
        candidate_output_sha256: "7".repeat(64),
        evaluator_identity: "evaluator".into(),
        proposer_identity: "proposer".into(),
        total: 10,
        baseline_correct: 5,
        candidate_correct: 6,
        baseline_truncated: 2,
        candidate_truncated: 0,
    };
    e.baseline.model_sha256 = r.source_sha256.clone();
    e.candidate.model_sha256 = r.candidate_sha256.clone();
    e.candidate.patch_sha256 = r.patch_sha256.clone();
    assert!(validate_evaluation(&e, &r).is_ok());
    e.candidate.runtime_sha256 = "8".repeat(64);
    assert!(validate_evaluation(&e, &r).is_err());
    e.candidate.runtime_sha256 = e.baseline.runtime_sha256.clone();
    e.evaluator_identity = e.proposer_identity.clone();
    assert!(validate_evaluation(&e, &r).is_err());
}

#[test]
fn identical_weights_new_generation_invalidates_cache() {
    let old = binding('a');
    let mut new = old.clone();
    new.model_generation = 1;
    let mut cache = GenerationCache::default();
    cache.bind(old.clone()).unwrap();
    cache.insert(&old, "kv".into(), vec![1.0f32]).unwrap();
    assert_eq!(cache.bind(new.clone()).unwrap(), 1);
    assert!(cache.get(&old, "kv").is_err());
    assert!(cache.get(&new, "kv").unwrap().is_none());
    new.protocol_sha256 = "6".repeat(64);
    assert!(old.require_match(&new).is_err());
}

#[test]
fn export_embeds_governance_without_claiming_authenticity() {
    let f = Fixture::new();
    let r = f.apply().unwrap();
    fs::write(f.path("receipt.json"), serde_json::to_vec(&r).unwrap()).unwrap();
    let governance = b"{\n\"synthetic_unit_test\": true\n}";
    fs::write(f.path("governance.json"), governance).unwrap();
    let result = export(
        &f.path("patch.json"),
        &f.path("policy.json"),
        &f.path("receipt.json"),
        None,
        Some(&f.path("governance.json")),
        &f.path("governance.rvf"),
    )
    .unwrap();
    assert_eq!(result["governance_evidence_sha256"], sha256(governance));
    assert_eq!(result["governance_evidence_authenticated"], false);
    let read = verify(
        &f.path("governance.rvf"),
        result["sha256"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(read["governance_evidence_sha256"][0], sha256(governance));
}

#[test]
fn export_rejects_receipt_exceeding_policy() {
    let f = Fixture::new();
    let mut r = f.apply().unwrap();
    r.l2 = 2.0;
    r.relative_l2 = r.l2 / r.source_l2;
    fs::write(f.path("receipt.json"), serde_json::to_vec(&r).unwrap()).unwrap();
    assert!(export(
        &f.path("patch.json"),
        &f.path("policy.json"),
        &f.path("receipt.json"),
        None,
        None,
        &f.path("bad.rvf")
    )
    .is_err());
    assert!(!f.path("bad.rvf").exists());
}
