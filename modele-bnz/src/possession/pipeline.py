"""Deterministic inference for the selected two-network model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit, logsumexp

from possession.decode import decode
from possession.inference import predict_sequence
from possession.model import PossessionTCN
from possession.relational import RelationalPossession, predict_relational


def configure_reference(threads=4):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "football-possession-pair":
        raise ValueError("Unsupported model bundle")
    expected = ["set-temporal", "relational-temporal"]
    if [entry["architecture"] for entry in manifest["models"]] != expected:
        raise ValueError("The selected pair requires both architectures in order")
    reference = dict(
        device="cpu", precision="float64", threads=4, batch_size=1, evidence_round_decimals=9
    )
    decoder = dict(temperature=6, change_penalty=2, boundary_weight=0.5)
    if manifest["reference"] != reference or manifest["decoder"] != decoder:
        raise ValueError("Reference inference settings differ from the supplied model")
    names = [entry["file"] for entry in manifest["models"]]
    if len(set(names)) != 2 or set(names) != set(manifest["files"]):
        raise ValueError("Model file inventory is inconsistent")
    for name, expected_hash in manifest["files"].items():
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError("Model files must be directly inside the weights directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Model checksum mismatch")
    return manifest


def infer(sequence, directory, progress=None):
    configure_reference()
    directory = Path(directory)
    manifest = load_bundle(directory)
    states, boundaries = [], []
    for entry in manifest["models"]:
        if sequence.match_id in entry["training_matches"]:
            raise ValueError("The input match was included in model training")
        if progress:
            progress(
                dict(
                    stage="network", model=entry["architecture"], device="cpu", precision="float64"
                )
            )
        checkpoint = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != entry["architecture"]:
            raise ValueError("Checkpoint architecture differs from its manifest")
        if entry["architecture"] == "set-temporal":
            model = PossessionTCN(**checkpoint["config"])
            predict = predict_sequence
        else:
            model = RelationalPossession(**checkpoint["config"])
            predict = predict_relational
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.double().eval()
        state, boundary = predict(model, sequence, "cpu", batch_size=1)
        if state.dtype != np.float64 or boundary.dtype != np.float64:
            raise ValueError("Reference inference requires float64 outputs")
        states.append(state)
        boundaries.append(expit(boundary))
    state = (logsumexp(np.stack(states), axis=0) - np.log(2.0)) / 6
    state -= logsumexp(state, axis=-1, keepdims=True)
    boundary = logit(np.clip(np.mean(boundaries, axis=0), 1e-6, 1 - 1e-6))
    state = np.round(state, 9)
    boundary = np.round(boundary, 9)
    return decode(state, sequence.period, change_penalty=2, boundary=boundary, boundary_weight=0.5)
