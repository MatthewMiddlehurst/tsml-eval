"""Tests for the DynamicRocket classifier."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval.estimators.classification.convolution_based._dynamic_rocket import (
    _dynamic_rocket_transform,
    _generate_kernels,
)


def _reference_dtw(a, b):
    cost = np.full((len(a) + 1, len(b) + 1), np.inf)
    cost[0, 0] = 0
    for i in range(len(a)):
        for j in range(len(b)):
            cost[i + 1, j + 1] = abs(a[i] - b[j]) + min(
                cost[i, j], cost[i, j + 1], cost[i + 1, j]
            )
    return cost[-1, -1]


def test_dynamic_rocket_transform():
    """Test the transform against a direct computation of each feature."""
    X, _ = make_example_3d_numpy(n_cases=3, n_timepoints=60, random_state=0)
    X = X[:, 0]
    lengths, weights, biases, dilations = _generate_kernels(np.arange(5), 60)

    Xt = _dynamic_rocket_transform(X, lengths, weights, biases, dilations)

    for k in range(5):
        w = weights[k, : lengths[k]]
        window = lengths[k] * dilations[k]
        for i, x in enumerate(X):
            segments = np.array(
                [x[s : s + window : dilations[k]] for s in range(60 - window)]
            )
            conv = segments @ w + biases[k]
            dtw = [_reference_dtw(s, w) for s in segments]

            np.testing.assert_allclose(Xt[i, k], max(dtw), rtol=1e-5)
            np.testing.assert_allclose(Xt[i, 5 + k], conv.max())
            np.testing.assert_allclose(Xt[i, 10 + k], np.mean(conv > 0))
