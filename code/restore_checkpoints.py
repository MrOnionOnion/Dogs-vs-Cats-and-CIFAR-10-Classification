"""Restore large checkpoints from GitHub-sized parts and verify their hashes."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoint_parts" / "manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["checkpoints"]:
        destination = ROOT / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = entry["sha256"]
        if destination.exists():
            current = hashlib.sha256(destination.read_bytes()).hexdigest()
            if current == expected:
                print(f"OK {entry['path']} (already restored)")
                continue
        temporary = destination.with_suffix(destination.suffix + ".restoring")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                for part in entry["parts"]:
                    chunk = (ROOT / "checkpoint_parts" / part).read_bytes()
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            if size != entry["size"] or digest.hexdigest() != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {entry['path']}")
            temporary.replace(destination)
            print(f"Restored {entry['path']} ({size} bytes, SHA-256 verified)")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
