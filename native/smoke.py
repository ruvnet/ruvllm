#!/usr/bin/env python3
"""Exercise the compiled JSON CLI on a tiny synthetic BF16 checkpoint only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(binary):
    require(binary.is_file() and not binary.is_symlink(), "Build a regular native executable first")
    checks = []
    environment = {key: value for key, value in os.environ.items()
                   if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}}
    with tempfile.TemporaryDirectory(prefix="ruvllm-native-smoke-") as temporary:
        root = Path(temporary)

        def write(name, value):
            path = root / name
            with path.open("x") as output:
                json.dump(value, output, sort_keys=True, allow_nan=False)
                output.write("\n")
            return str(path)

        def call(request, expected_success=True):
            result = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                                    capture_output=True, timeout=30, env=environment, check=False)
            require(len(result.stdout) + len(result.stderr) < 1024 * 1024, "Unexpected CLI output size")
            require((result.returncode == 0) is expected_success, "Unexpected CLI exit status for " + request["op"])
            value = json.loads(result.stdout if expected_success else result.stderr)
            if not expected_success:
                require(value.get("ok") is False, "Failure must be structured JSON")
            return value

        header = json.dumps({"layer.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}},
                            separators=(",", ":")).encode()
        header += b" " * ((-len(header)) % 8)
        original = struct.pack("<Q", len(header)) + header + struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
        source = root / "original.safetensors"
        source.write_bytes(original)
        source_sha = digest(original)
        inspection = call({"op": "inspect", "source": str(source), "tensors": ["layer.weight"]})
        require(inspection["source_sha256"] == source_sha and inspection["tensors"][0]["count"] == 4
                and inspection["tensors"][0]["mean"] == 2.5, "Inspection does not match fixture bytes")
        checks.append("inspect")
        patch = write("patch.json", {"schema": "ruvllm.weight_patch.v1", "source_sha256": source_sha,
                                    "edits": [{"tensor": "layer.weight", "index": 1,
                                               "old_bits": 0x4000, "new_bits": 0x4020}]})
        policy_value = {"allowed_tensors": ["layer.weight"], "max_changed_values": 1,
                        "max_l2": 0.5, "max_abs_delta": 0.5, "max_relative_l2": 1.0, "protected": []}
        policy = write("policy.json", policy_value)
        candidate = root / "candidate.safetensors"
        request = {"op": "patch", "source": str(source), "output": str(candidate), "patch": patch, "policy": policy}
        receipt = call(request)
        expected = original[:-8] + struct.pack("<4H", 0x3F80, 0x4020, 0x4040, 0x4080)
        require(candidate.read_bytes() == expected and receipt["candidate_sha256"] == digest(expected)
                and receipt["changed_values"] == 1 and receipt["l2"] == 0.5
                and receipt["activation_authorized"] is False, "Patch does not match exact expected bytes")
        receipt_path = write("receipt.json", receipt)
        checks.append("patch")
        call(request, False)
        require(candidate.read_bytes() == expected, "Rejected overwrite changed the candidate")
        checks.append("overwrite_rejected")
        protected = write("protected.json", {**policy_value, "protected": [{"tensor": "layer.weight", "start": 1, "end": 2}]})
        denied = root / "denied.safetensors"
        call({**request, "policy": protected, "output": str(denied)}, False)
        require(not denied.exists(), "Protected edit created output")
        checks.append("protected_edit_rejected")
        restored = root / "restored.safetensors"
        rollback = call({"op": "rollback", "source": str(candidate), "output": str(restored),
                         "patch": patch, "policy": policy, "receipt": receipt_path})
        require(restored.read_bytes() == original and source.read_bytes() == original
                and rollback["candidate_sha256"] == source_sha, "Rollback did not restore original bytes")
        checks.append("byte_exact_rollback")
        shared = {"runtime_sha256": "1" * 64, "topology_sha256": "2" * 64,
                  "protocol_sha256": "3" * 64, "toolchain_sha256": "4" * 64}
        baseline = {**shared, "model_generation": 0, "model_sha256": source_sha, "patch_sha256": "0" * 64}
        changed = {**shared, "model_generation": 1, "model_sha256": digest(expected), "patch_sha256": receipt["patch_sha256"]}
        evaluation = write("synthetic-evaluation.json", {
            "schema": "ruvllm.weight_evaluation.v1", "baseline": baseline, "candidate": changed,
            "suite_sha256": "5" * 64, "protocol_sha256": shared["protocol_sha256"],
            "baseline_output_sha256": "6" * 64, "candidate_output_sha256": "7" * 64,
            "evaluator_identity": "synthetic-evaluator", "proposer_identity": "synthetic-proposer",
            "total": 2, "baseline_correct": 0, "candidate_correct": 1,
            "baseline_truncated": 1, "candidate_truncated": 0})
        valid = call({"op": "evaluate", "evaluation": evaluation, "receipt": receipt_path})
        require(valid["valid"] is True and valid["activation_authorized"] is False, "Evaluation contract differs")
        checks.append("synthetic_evaluation_contract")
        artifact = root / "fixture.rvf"
        export = call({"op": "export", "patch": patch, "policy": policy, "receipt": receipt_path,
                       "evaluation": evaluation, "output": str(artifact)})
        require(export["sha256"] == digest(artifact.read_bytes()) and export["activation_authorized"] is False,
                "RVF export differs")
        verified = call({"op": "verify", "artifact": str(artifact), "expected_sha256": export["sha256"]})
        require(verified["integrity_verified"] is True and verified["activation_authorized"] is False,
                "RVF integrity validation differs")
        checks.extend(["rvf_export", "rvf_verify"])
        require(call({"op": "check_generation", "expected": changed, "observed": [changed]})["consistent"] is True,
                "Matching generation rejected")
        call({"op": "check_generation", "expected": changed, "observed": [baseline]}, False)
        checks.append("stale_generation_rejected")
        call({"op": "activate"}, False)
        checks.append("unknown_operation_rejected")
        return {"schema": "ruvllm-native-cli-smoke/v1", "passed": True, "fixture_only": True,
                "binary_sha256": digest(binary.read_bytes()), "checks": checks,
                "source_sha256": source_sha, "restored_sha256": rollback["candidate_sha256"],
                "candidate_sha256": receipt["candidate_sha256"], "rvf_sha256": export["sha256"],
                "production_activation": False, "inference_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parent / "target/release/weight_evolution")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.binary.absolute()), indent=2, sort_keys=True))
