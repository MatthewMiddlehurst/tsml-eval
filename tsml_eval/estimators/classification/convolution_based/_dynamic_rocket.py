"""DynamicRocket classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DynamicRocketClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.linear_model import RidgeClassifier
from sklearn.utils import check_random_state


class DynamicRocketClassifier(BaseClassifier):
    """ROCKET with dynamic convolution (DynamicRocket).

    DynamicRocket [1]_ adds dynamic convolution features to ROCKET. Dynamic
    convolution replaces the dot product between a kernel and each dilated
    subsequence of the series with the dynamic time warping (DTW) distance between
    them. Each kernel gives three features: the maximum DTW distance between the
    kernel and the subsequences, and the maximum and proportion of positive values of
    the standard convolution output, as in ROCKET.

    Kernels are generated as in the reference implementation [2]_, which differs from
    ROCKET. The kernel weights are not mean centred, the dilation exponent is drawn
    up to one less than the largest that fits the series, and no kernels are padded.
    The series are not normalised, and a ``RidgeClassifier`` with the default
    regularisation is fitted on the unscaled features.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of random kernels.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means using
        all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to generate the kernels. Kernel ``i`` is
        generated from the seed ``random_state + i``, so ``random_state=0`` gives the
        kernels of the reference implementation.

    Attributes
    ----------
    classifier_ : RidgeClassifier
        The ridge classifier fitted on the transformed training data.

    Notes
    -----
    The DTW distance uses the absolute difference between points and no warping
    window. As in the reference, the convolution is applied at the
    ``n_timepoints - length * dilation`` positions where the kernel and one further
    dilation step fit in the series. Series shorter than 10 are zero padded to length
    10, and dilations below one, which the reference does not handle, are set to one.

    References
    ----------
    .. [1] Buza, Krisztian, and Margit Antal. "ROCKET with dynamic convolution for
       time series classification." International Conference on Machine Learning,
       Optimization, and Data Science, 2024.
    .. [2] https://github.com/kr7/DynamicROCKET

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     DynamicRocketClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = DynamicRocketClassifier(n_kernels=50, random_state=0)
    >>> clf.fit(X, y)
    DynamicRocketClassifier(n_kernels=50, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(self, n_kernels=10000, n_jobs=1, random_state=None):
        self.n_kernels = n_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        X = self._preprocess(X)

        if self.random_state is None or isinstance(
            self.random_state, np.random.RandomState
        ):
            rng = check_random_state(self.random_state)
            seed = rng.randint(np.iinfo(np.int32).max)
        else:
            seed = self.random_state

        self._kernels = _generate_kernels(seed + np.arange(self.n_kernels), X.shape[1])
        self.classifier_ = RidgeClassifier().fit(self._transform(X), y)

        return self

    def _predict(self, X):
        return self.classifier_.predict(self._transform(self._preprocess(X)))

    def _transform(self, X):
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        Xt = _dynamic_rocket_transform(X, *self._kernels)
        set_num_threads(prev_threads)
        return Xt

    @staticmethod
    def _preprocess(X):
        X = X[:, 0].astype(np.float64)
        if X.shape[1] < 10:
            X = np.pad(X, ((0, 0), (0, 10 - X.shape[1])))
        return X

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {"n_kernels": 20}


def _generate_kernels(seeds, n_timepoints):
    """Generate one kernel from each seed as in the DynamicRocket reference."""
    n_kernels = len(seeds)
    lengths = np.zeros(n_kernels, dtype=np.int64)
    weights = np.zeros((n_kernels, 9), dtype=np.float64)
    biases = np.zeros(n_kernels, dtype=np.float64)
    dilations = np.zeros(n_kernels, dtype=np.int64)

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        length = rng.randint(0, 2) * 2 + 7
        weights[i, :length] = rng.normal(0, 1, length)
        biases[i] = rng.uniform(-1, 1)
        max_exponent = np.floor(np.log2((n_timepoints - 1) / (length - 1))) - 1
        dilations[i] = max(int(2 ** rng.uniform(0, max_exponent)), 1)
        lengths[i] = length

    return lengths, weights, biases, dilations


@njit(fastmath=True, cache=True)
def _dtw(x, start, dilation, weights, length, cost):
    """DTW distance between a dilated subsequence and a kernel, in float32."""
    for i in range(length):
        a = np.float32(x[start + i * dilation])
        for j in range(length):
            d = np.float32(abs(a - np.float32(weights[j])))
            if i == 0 and j == 0:
                cost[i, j] = d
            elif i == 0:
                cost[i, j] = cost[i, j - 1] + d
            elif j == 0:
                cost[i, j] = cost[i - 1, j] + d
            else:
                cost[i, j] = min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]) + d
    return cost[length - 1, length - 1]


@njit(fastmath=True, parallel=True, cache=True)
def _dynamic_rocket_transform(X, lengths, weights, biases, dilations):
    n_cases, n_timepoints = X.shape
    n_kernels = len(lengths)
    features = np.zeros((n_cases, 3 * n_kernels), dtype=np.float64)

    for i in prange(n_cases):
        cost = np.zeros((9, 9), dtype=np.float32)
        for k in range(n_kernels):
            length = lengths[k]
            dilation = dilations[k]
            n_positions = n_timepoints - length * dilation
            if n_positions < 1:
                continue

            max_dtw = -np.inf
            max_conv = -np.inf
            n_positive = 0
            for s in range(n_positions):
                conv = biases[k]
                for m in range(length):
                    conv += X[i, s + m * dilation] * weights[k, m]
                if conv > max_conv:
                    max_conv = conv
                if conv > 0:
                    n_positive += 1

                dtw = _dtw(X[i], s, dilation, weights[k], length, cost)
                if dtw > max_dtw:
                    max_dtw = dtw

            features[i, k] = max_dtw
            features[i, n_kernels + k] = max_conv
            features[i, 2 * n_kernels + k] = n_positive / n_positions

    return features
