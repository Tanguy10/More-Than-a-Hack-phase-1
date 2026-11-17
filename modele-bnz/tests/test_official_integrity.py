import hashlib
import json
from pathlib import Path

from possession._official import possession_metrics


def test_official_code_matches_recorded_provenance():
    root = Path(possession_metrics.__file__).parent
    provenance = json.loads((root / "provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
