"""Locate external datasets, which must never be committed to Git.

Default: Git-ignored data/external/, locally linked to /Data/tanguy.azema/external.
Override with MTAH_EXTERNAL_DIR."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def external_dir():
    return os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(ROOT, "data", "external")
