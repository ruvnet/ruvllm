"""Verify an extracted release using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory):
    root = Path(directory).resolve()
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
             and p != root / "SHA256SUMS.json" and "__pycache__" not in p.parts}
    if files != set(hashes):
        raise ValueError("Release file inventory differs from the manifest")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Release integrity failure: {name}")
    meta = json.loads((root / "release.json").read_text())
    if not meta["qualified_local_preview"] or meta["sota_verified"] or not meta["integration_passed"]:
        raise ValueError("Release does not qualify as a measured local preview")
    if sha256(root / "mlx/adapters.safetensors") != meta["mlx_adapter_sha256"]:
        raise ValueError("MLX weight mismatch")
    if sha256(root / "adapter_model.safetensors") != meta["peft_adapter_sha256"]:
        raise ValueError("PEFT weight mismatch")
    for folder in (root / "evidence").iterdir():
        if not folder.is_dir() or not (folder / "responses.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate evaluation task IDs")
        groups = {}
        for row in rows:
            group = groups.setdefault(row["kind"], {"passed": 0, "total": 0})
            group["passed"] += int(row["passed"])
            group["total"] += 1
        actual = {"total": len(rows), "passed": sum(r["passed"] for r in rows),
                  "truncated": sum(r["finish_reason"] == "length" for r in rows),
                  "output_tokens": sum(r["output_tokens"] for r in rows), "groups": groups}
        summary = json.loads((folder / "summary.json").read_text())
        if any(actual[key] != summary[key] for key in actual):
            raise ValueError(f"Summary mismatch: {folder.name}")
        if summary.get("response_sha256") and sha256(folder / "responses.jsonl") != summary["response_sha256"]:
            raise ValueError(f"Response receipt mismatch: {folder.name}")
    external = json.loads((root / "evidence/external.json").read_text())
    if not external["candidate_qualified"] or external["selection_use"] or external["selected"]["passed"] < external["baseline"]["passed"]:
        raise ValueError("External release gate failed")
    for name in ("read-integration.json", "coding-integration.json"):
        receipt = json.loads((root / "evidence" / name).read_text())
        if not receipt["passed"] or receipt["adapter_sha256"] != meta["mlx_adapter_sha256"] or receipt["permission_bypass"] or not receipt.get("backend_binding_verified") or receipt.get("matched_adapter_tensors") != 32:
            raise ValueError("Integration does not match the released weights")
    loaded = json.loads((root / "evidence/backend-loaded.json").read_text())
    if not loaded["binding_verified"] or loaded["matched_adapter_tensors"] != 32 or loaded["adapter_sha256"] != meta["mlx_adapter_sha256"] or loaded["adapter_config_sha256"] != sha256(root / "mlx/adapter_config.json"):
        raise ValueError("Loaded tensor receipt does not match the released adapter")
    cloud = json.loads((root / "evidence/cloud/receipts.json").read_text())
    if any(j["stage"] != "COMPLETED" for j in cloud) or sum(j["reserved_max_usd"] for j in cloud) > 200:
        raise ValueError("Cloud completion or budget gate failed")
    return {"verified": True, "files": len(hashes), "adapter_sha256": meta["mlx_adapter_sha256"],
            "external_passed": external["selected"]["passed"], "external_total": external["selected"]["total"],
            "sota_verified": False, "public_upload_performed": meta["public_upload_performed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=str(Path(__file__).resolve().parent))
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
