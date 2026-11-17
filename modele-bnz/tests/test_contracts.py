import numpy as np
from possession.decode import decode
from possession.submission import write_submission


def test_viterbi_short_possession_and_long_segment():
    truth = np.array([0] * 100 + [1] * 5 + [0] * 100 + [2] * 20 + [0] * 5000)
    scores = np.full((len(truth), 3), -20.0)
    scores[np.arange(len(truth)), truth] = 0
    pred = decode(scores, np.ones(len(truth), np.int8), change_penalty=5.0)
    assert np.array_equal(pred, truth)


def test_weak_opponent_touch_does_not_force_a_new_possession():
    scores = np.full((100, 3), -20.0)
    scores[:, 0] = 0.0
    scores[:, 1] = -3.0
    scores[50, 1] = 0.2
    pred = decode(scores, np.ones(100, np.int8), change_penalty=5.0)
    assert np.all(pred == 0)


def test_future_confirmation_does_not_delay_the_first_control_frame():
    scores = np.full((100, 3), -20.0)
    scores[:50, 0] = 0.0
    scores[:50, 1] = -8.0
    scores[50:55, 0] = -0.05
    scores[50:55, 1] = 0.0
    scores[55:, 0] = -8.0
    scores[55:, 1] = 0.0
    pred = decode(scores, np.ones(100, np.int8), change_penalty=5.0)
    np.testing.assert_array_equal(pred, np.array([0] * 50 + [1] * 50))


def test_no_cross_period_or_forced_delay():
    truth = np.array([0] * 30 + [3] * 5 + [1] * 30)
    period = np.array([1] * 30 + [0] * 5 + [2] * 30)
    scores = np.full((65, 3), -20.0)
    scores[:30, 0] = 0
    scores[35:, 1] = 0
    assert np.array_equal(decode(scores, period), truth)


def test_csv_is_byte_reproducible_and_exactly_valid(tmp_path):
    labels = np.full(199319, 3, np.int8)
    labels[11185:90652] = 0
    labels[110808:190198] = 2
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert write_submission(labels, a) == write_submission(labels, b)
    assert a.read_bytes() == b.read_bytes()
