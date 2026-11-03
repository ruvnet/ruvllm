"""Check the local archive and verify its extracted contents and evidence."""
import json

from ruflo_model.common import ROOT, digest
from verify_release import verify


def main():
    artifact = json.loads((ROOT / "results/release-artifact.json").read_text())
    if digest(ROOT / artifact["archive"]) != artifact["archive_sha256"]:
        raise ValueError("Archive hash mismatch")
    print(json.dumps(verify(ROOT / artifact["directory"]), indent=2))


if __name__ == "__main__":
    main()
