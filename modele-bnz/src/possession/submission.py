"""Deterministic CSV rendering and exact official validation."""

from pathlib import Path
import hashlib
import numpy as np
from possession import LABELS
from possession._official import possession_metrics as official


def write_submission(labels, path, expected=199319):
    labels = np.asarray(labels)
    if labels.shape != (expected,) or not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError("Invalid submission shape or label code")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        for index, value in enumerate(labels):
            handle.write(f"{index},{LABELS[int(value)]}\n")
    error = official.validate_submission(official.read_label_csv(path), expected)
    if error:
        raise ValueError(error)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest
