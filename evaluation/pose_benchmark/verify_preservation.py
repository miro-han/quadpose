"""Verify both generations of historical inputs, using the archived original README."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    for inventory in ("FULL_PACKAGE_MANIFEST.json", "evaluation/pose_benchmark/provenance/before.json"):
        entries = json.loads((ROOT / inventory).read_text(encoding="utf-8-sig"))
        failed = []
        for entry in entries:
            path = ROOT / entry["path"]
            if entry["path"] == "README.md":
                path = ROOT / "evaluation/pose_benchmark/provenance/README.before.md"
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                failed.append(entry["path"])
                continue
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    failed.append(entry["path"])
        print(f"{inventory}: {len(entries) - len(failed)}/{len(entries)} verified")
        if failed:
            raise SystemExit("Missing or changed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
