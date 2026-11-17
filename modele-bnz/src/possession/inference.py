"""Overlapped inference: retain only central outputs, with matching period padding."""

import numpy as np
import torch
from possession.windows import extract_window, collate
from possession.schema import period_runs


@torch.inference_mode()
def predict_sequence(model, sequence, device="cpu", batch_size=4, central=1024):
    model.eval()
    context = max(512, model.context)
    length = central + context * 2
    dtype = next(model.parameters()).dtype
    numpy_dtype = "f8" if dtype == torch.float64 else "f4"
    state = np.zeros((len(sequence.period), 3), numpy_dtype)
    boundary = np.zeros((len(sequence.period), 3, 3), numpy_dtype)
    tasks = [
        (run, start, min(central, run[1] - start))
        for run in period_runs(sequence.period)
        for start in range(run[0], run[1], central)
    ]
    for index in range(0, len(tasks), batch_size):
        part = tasks[index : index + batch_size]
        batch = collate(
            [extract_window(sequence, start - context, length, run) for run, start, n in part]
        )
        batch = {
            k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()
        }
        logp, b = model(**batch)
        logp = logp.cpu().numpy()
        b = b.cpu().numpy()
        for i, (run, start, n) in enumerate(part):
            state[start : start + n] = logp[i, context : context + n]
            boundary[start : start + n] = b[i, context : context + n]
    return state, boundary
