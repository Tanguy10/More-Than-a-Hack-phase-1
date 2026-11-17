"""Generate a possession CSV from tactical tracking."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from possession.io.tactical import load_tactical
from possession.pipeline import infer
from possession.submission import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Participant bundle containing tactical/"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Directory containing manifest.json and the two models",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination frame,label CSV")
    args = parser.parse_args()
    started = time.monotonic()
    sequence = load_tactical(args.data)
    prediction = infer(sequence, args.weights, lambda row: print(json.dumps(row), flush=True))
    digest = write_submission(prediction, args.output)
    metadata = dict(
        csv_sha256=digest,
        rows=len(prediction),
        weights_manifest_sha256=hashlib.sha256(
            (args.weights / "manifest.json").read_bytes()
        ).hexdigest(),
        runtime=dict(
            python=sys.version.split()[0],
            torch=str(torch.__version__),
            numpy=np.__version__,
            platform=platform.system(),
            device="cpu",
            precision="float64",
            threads=4,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    args.output.with_suffix(".generation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dict(csv_valid=True, rows=len(prediction), sha256=digest)))


if __name__ == "__main__":
    main()
