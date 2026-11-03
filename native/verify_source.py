#!/usr/bin/env python3
"""Verify copied source hashes and the standalone lock against local provenance."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tomllib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), "Expected a regular source file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    provenance = json.loads((root / "upstream-provenance.json").read_text())
    require(provenance["source_commit"] == "f5c3444968ede44e43744460192198962ae94da2", "Unexpected upstream source")
    paths = set()
    for record in provenance["files"]:
        relative = PurePosixPath(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and str(relative) == record["path"] and record["path"] not in paths, "Invalid source path")
        path = root
        for component in relative.parts:
            path = path / component
            require(not path.is_symlink(), "Symlink in copied source path")
        require(record["modified"] is False and path.stat().st_size == record["bytes"]
                and sha(path) == record["sha256"], "Copied source changed: " + str(relative))
        paths.add(str(relative))
    for folder in ("vendor", "src/weight_evolution", "examples"):
        actual = {path.relative_to(root).as_posix() for path in (root / folder).rglob("*") if path.is_file()}
        require(actual == {path for path in paths if path.startswith(folder + "/")}, "Copied source inventory differs")
    registry = json.loads((root / "registry-dependencies.json").read_text())
    require(sha(root / "Cargo.lock") == registry["cargo_lock_sha256"]
            and registry["historical_cargo_lock_sha256"] == provenance["source_cargo_lock_sha256"]
            and registry["all_versions_and_checksums_match_historical_lock"] is True, "Cargo lock provenance differs")
    lock = tomllib.loads((root / "Cargo.lock").read_text())
    observed = {(p["name"], p["version"]): (p["source"], p["checksum"])
                for p in lock["package"] if "source" in p}
    expected = {(p["name"], p["version"]): (p["source"], p["checksum"]) for p in registry["packages"]}
    require(observed == expected and len(expected) == len(registry["packages"]), "Registry dependency set differs")
    return {"verified": True, "copied_source_files": len(paths), "registry_dependencies": len(expected),
            "source_commit": provenance["source_commit"], "cargo_lock_sha256": registry["cargo_lock_sha256"],
            "scope": "Local integrity against bundled provenance; not independent source authentication"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, sort_keys=True))
