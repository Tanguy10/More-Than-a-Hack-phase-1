"""Full-rate Viterbi. A missing ball never creates an automatic dead-ball state."""

import numpy as np
from numba import njit
from possession_c.schema import period_runs


@njit(cache=True)
def _viterbi(scores, penalty, boundary, boundary_weight):
    n = len(scores)
    classes = scores.shape[1]
    back = np.zeros((n, classes), np.int8)
    previous = scores[0].copy()
    for t in range(1, n):
        current = np.empty(classes, np.float64)
        for k in range(classes):
            best, arg = -1e300, 0
            for j in range(classes):
                value = previous[j]
                if j != k:
                    value -= penalty[j, k]
                    value += boundary_weight * boundary[t, j, k]
                if value > best:
                    best, arg = value, j
            current[k] = best + scores[t, k]
            back[t, k] = arg
        previous = current
    output = np.empty(n, np.int8)
    output[-1] = np.argmax(previous)
    for t in range(n - 1, 0, -1):
        output[t - 1] = back[t, output[t]]
    return output


def decode(
    scores, period, change_penalty=8.0, dead_penalty=None, boundary=None, boundary_weight=0.0
):
    scores = np.asarray(scores, np.float64)
    period = np.asarray(period)
    if (scores.ndim != 2 or scores.shape[0] != len(period) or scores.shape[1] not in (2, 3)
            or not np.isfinite(scores).all()):
        raise ValueError("Decoder requires finite T x 2 or T x 3 state scores")
    classes = scores.shape[1]
    if not np.isfinite(change_penalty) or change_penalty < 0 or not np.isfinite(boundary_weight):
        raise ValueError("Invalid decoder weights")
    penalty = np.full((classes, classes), float(change_penalty))
    if dead_penalty is not None:
        if classes != 3 or not np.isfinite(dead_penalty) or dead_penalty < 0:
            raise ValueError("Dead penalty requires a ternary decoder")
        penalty[2, :2] = penalty[:2, 2] = dead_penalty
    np.fill_diagonal(penalty, 0.0)
    if boundary is None:
        boundary = np.zeros((len(period), classes, classes), np.float32)
    boundary = np.asarray(boundary)
    if boundary.shape != (len(period), classes, classes) or not np.isfinite(boundary).all():
        raise ValueError("Transition scores must match the class axes")
    output = np.full(len(scores), 3, np.int8)
    for start, stop in period_runs(period):
        output[start:stop] = _viterbi(
            scores[start:stop], penalty, boundary[start:stop], boundary_weight
        )
    return output


