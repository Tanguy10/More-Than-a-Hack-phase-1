import hashlib
import json

import numpy as np
import pytest
import torch

from possession import pipeline
from possession.model import PossessionTCN
from possession.relational import RelationalPossession
from possession.schema import MatchSequence


@pytest.fixture
def bundle(tmp_path):
    torch.manual_seed(37)
    models, files = [], {}
    for architecture, cls in (
        ("set-temporal", PossessionTCN),
        ("relational-temporal", RelationalPossession),
    ):
        model = cls(channels=16, blocks=2, set_width=16, set_output=16, dropout=0)
        path = tmp_path / (architecture + ".pt")
        torch.save(
            dict(architecture=architecture, config=model.config, state_dict=model.state_dict()),
            path,
        )
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        models.append(dict(file=path.name, architecture=architecture, training_matches=["train"]))
    manifest = dict(
        format="football-possession-pair",
        models=models,
        files=files,
        decoder=dict(temperature=6, change_penalty=2, boundary_weight=0.5),
        reference=dict(
            device="cpu", precision="float64", threads=4, batch_size=1, evidence_round_decimals=9
        ),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def sequence(match_id="synthetic"):
    n = 70
    rng = np.random.default_rng(3)
    return MatchSequence(
        np.arange(n),
        np.ones(n, "i1"),
        rng.random((n, 4, 2)).astype("f4") * 30,
        np.array([0, 0, 1, 1]),
        np.ones((n, 4), "f4"),
        rng.random((n, 2)).astype("f4") * 30,
        (105.0, 68.0),
        match_id,
    )


def test_pair_is_deterministic_and_refuses_changed_weights(bundle, monkeypatch):
    monkeypatch.setattr(pipeline, "configure_reference", lambda: None)
    first = pipeline.infer(sequence(), bundle)
    second = pipeline.infer(sequence(), bundle)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (70,) and np.isin(first, (0, 1, 2)).all()
    with (bundle / "set-temporal.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        pipeline.infer(sequence(), bundle)


def test_training_match_cannot_be_presented_as_unseen(bundle, monkeypatch):
    monkeypatch.setattr(pipeline, "configure_reference", lambda: None)
    with pytest.raises(ValueError, match="training"):
        pipeline.infer(sequence("train"), bundle)


def test_incomplete_pair_and_changed_decoder_are_rejected(bundle):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["decoder"]["temperature"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="settings"):
        pipeline.load_bundle(bundle)
    manifest["decoder"]["temperature"] = 6
    manifest["models"] = manifest["models"][:1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="architectures"):
        pipeline.load_bundle(bundle)


def test_manifest_cannot_read_outside_weights_directory(bundle):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["models"][0]
    old = entry["file"]
    entry["file"] = "../outside.pt"
    manifest["files"][entry["file"]] = manifest["files"].pop(old)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="directly"):
        pipeline.load_bundle(bundle)
