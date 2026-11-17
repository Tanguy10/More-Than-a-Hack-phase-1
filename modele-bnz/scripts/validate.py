"""Check the submission format with the official validator."""

import argparse
import hashlib
import json
from pathlib import Path

from possession._official import possession_metrics as official


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--rows", type=int, default=199319)
    args = parser.parse_args()
    reason = official.validate_submission(official.read_label_csv(args.submission), args.rows)
    if reason:
        parser.exit(1, reason + "\n")
    print(
        json.dumps(
            dict(
                valid=True,
                rows=args.rows,
                sha256=hashlib.sha256(args.submission.read_bytes()).hexdigest(),
            )
        )
    )
