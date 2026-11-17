"""Verify the distributed code, weights and submission without ML dependencies."""

import hashlib
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checksum path outside the repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        count += 1
    print(f"{count} files verified")
