"""ROCKET-XGBoost regressor."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RocketXGBoostRegressor"]

import threading

import numpy as np
from aeon.regression import BaseRegressor
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation import check_n_jobs
from numba import config, get_num_threads, njit, prange, set_num_threads
from sklearn.utils import check_random_state

try:
    # aeon >= 1.6 serialises its own numba thread changes with this lock.
    from aeon.utils._parallel import _NUMBA_PARALLEL_LOCK
except ImportError:
    _NUMBA_PARALLEL_LOCK = threading.Lock()


class RocketXGBoostRegressor(BaseRegressor):
    """ROCKET with five pooling operators and an XGBoost regressor (ROCKET-XGBoost).

    ROCKET-XGBoost [1]_ modifies the ROCKET extrinsic regression baseline of [2]_ in
    two ways. In place of a ridge regression layer, a gradient boosted tree ensemble
    (``XGBRegressor`` with default hyperparameters) is fitted directly on the pooled
    convolution features. To give the tree ensemble richer summary statistics without
    hyperparameter tuning, three pooling operators are added to ROCKET's proportion
    of positive values (PPV) and global max pooling: global average pooling (mean),
    global min pooling (min), and proportion of negative values (PNV), producing five
    features per random kernel.

    With the default ``n_kernels=2000``, the transform extracts 10,000 ``float32``
    features. Input series are not z-normalised by default, matching the default
    extrinsic regression setup of [2]_ on which [1]_ builds.

    Parameters
    ----------
    n_kernels : int, default=2000
        Number of random convolutional kernels. Each kernel produces 5 features.
    normalise : bool, default=False
        Whether to z-normalise each input time series per instance and channel before
        applying the kernels.
    n_jobs : int, default=1
        The number of jobs to run in parallel for kernel application and
        ``XGBRegressor``. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for ROCKET kernel generation and
        ``XGBRegressor``.

    Attributes
    ----------
    regressor_ : XGBRegressor
        The fitted XGBoost regressor.

    Notes
    -----
    Random kernels are generated with aeon's ``Rocket`` transformer (``float32``
    weights and biases), convolved against ``float64`` series and pooled into a
    ``float32`` feature matrix, matching ``models/rocket.py`` in [2]_. The downstream
    tree ensemble uses ``XGBRegressor`` (requires ``xgboost>=2.0``). Series shorter
    than 11 time points are zero padded to length 11 so all candidate kernel lengths
    ``{7, 9, 11}`` have at least one valid convolution position.

    References
    ----------
    .. [1] Bayani, Mani. "Improving Time Series Extrinsic Regression." In *Essays on
       Machine Learning Methods in Economics*, Chapter 2, pp. 49-73. PhD diss.,
       City University of New York, 2022.
    .. [2] Tan, Chang Wei, Christoph Bergmeir, Francois Petitjean, and Geoffrey I.
       Webb. "Time series extrinsic regression." *Data Mining and Knowledge
       Discovery*, 35(3), 1032-1060, 2021.

    Examples
    --------
    >>> from tsml_eval.estimators.regression.convolution_based import (
    ...     RocketXGBoostRegressor,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=20, n_channels=1, regression_target=True, random_state=0
    ... )
    >>> reg = RocketXGBoostRegressor(n_kernels=50, random_state=0)  # doctest: +SKIP
    >>> reg.fit(X, y)  # doctest: +SKIP
    RocketXGBoostRegressor(n_kernels=50, random_state=0)
    >>> y_pred = reg.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": "xgboost>=2.0",
    }

    def __init__(
        self,
        n_kernels=2000,
        normalise=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_kernels < 1:
            raise ValueError(f"n_kernels must be at least 1, got {self.n_kernels}.")

        from xgboost import XGBRegressor

        self._n_jobs = check_n_jobs(self.n_jobs)
        self._numba_jobs = min(self._n_jobs, config.NUMBA_NUM_THREADS)

        if self.random_state is None or isinstance(
            self.random_state, np.random.RandomState
        ):
            rng = check_random_state(self.random_state)
            seed = int(rng.randint(np.iinfo(np.int32).max))
        else:
            seed = int(self.random_state)

        X_prep = self._preprocess(X)
        self._kernels = (
            Rocket(
                n_kernels=self.n_kernels,
                normalise=False,
                n_jobs=self._numba_jobs,
                random_state=seed,
            )
            .fit(X_prep)
            .kernels
        )
        Xt = self._transform(X_prep)

        self.regressor_ = XGBRegressor(n_jobs=self._n_jobs, random_state=seed)
        self.regressor_.fit(Xt, y)
        return self

    def _predict(self, X):
        Xt = self._transform(self._preprocess(X))
        return self.regressor_.predict(Xt)

    def _preprocess(self, X):
        if self.normalise:
            X = Normalizer().fit_transform(X)
        if X.shape[2] < 11:
            X = np.pad(X, ((0, 0), (0, 0), (0, 11 - X.shape[2])))
        return np.asarray(X, dtype=np.float64)

    def _transform(self, X):
        with _NUMBA_PARALLEL_LOCK:
            prev_threads = get_num_threads()
            try:
                set_num_threads(self._numba_jobs)
                return _apply_rocket_xgboost_kernels(X, self._kernels)
            finally:
                set_num_threads(prev_threads)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {"n_kernels": 20}


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate_5(X, weights, length, bias, dilation, padding):
    """Apply one ROCKET kernel to a univariate series and return 5 summary features."""
    n_timepoints = len(X)
    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _pnv = 0
    _max = -np.inf
    _min = np.inf
    _total = 0.0

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = np.float64(bias)
        index = i
        for j in range(length):
            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation

        if _sum > _max:
            _max = _sum
        if _sum < _min:
            _min = _sum
        if _sum > 0:
            _ppv += 1
        elif _sum < 0:
            _pnv += 1
        _total += _sum

    return (
        np.float32(_ppv / output_length),
        np.float32(_max),
        np.float32(_total / output_length),
        np.float32(_min),
        np.float32(_pnv / output_length),
    )


@njit(fastmath=True, cache=True)
def _apply_kernel_multivariate_5(
    X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices
):
    """Apply one ROCKET kernel to a multivariate series and return 5 features."""
    _, n_timepoints = X.shape
    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _pnv = 0
    _max = -np.inf
    _min = np.inf
    _total = 0.0

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = np.float64(bias)
        index = i
        for j in range(length):
            if index > -1 and index < n_timepoints:
                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]
            index = index + dilation

        if _sum > _max:
            _max = _sum
        if _sum < _min:
            _min = _sum
        if _sum > 0:
            _ppv += 1
        elif _sum < 0:
            _pnv += 1
        _total += _sum

    return (
        np.float32(_ppv / output_length),
        np.float32(_max),
        np.float32(_total / output_length),
        np.float32(_min),
        np.float32(_pnv / output_length),
    )


@njit(parallel=True, fastmath=True, cache=True)
def _apply_rocket_xgboost_kernels(X, kernels):
    """Apply all ROCKET kernels to ``X`` and return ``(n_cases, 5 * n_kernels)``."""
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        n_channel_indices,
        channel_indices,
    ) = kernels
    n_cases = len(X)
    n_kernels = len(lengths)

    _X = np.zeros((n_cases, n_kernels * 5), dtype=np.float32)

    for i in prange(n_cases):
        a1 = 0
        a2 = 0
        a3 = 0

        for j in range(n_kernels):
            b1 = a1 + n_channel_indices[j] * lengths[j]
            b2 = a2 + n_channel_indices[j]

            if n_channel_indices[j] == 1:
                ppv, mx, mean_val, mn, pnv = _apply_kernel_univariate_5(
                    X[i][channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )
            else:
                _weights = weights[a1:b1].reshape((n_channel_indices[j], lengths[j]))
                ppv, mx, mean_val, mn, pnv = _apply_kernel_multivariate_5(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    n_channel_indices[j],
                    channel_indices[a2:b2],
                )

            _X[i, a3] = ppv
            _X[i, a3 + 1] = mx
            _X[i, a3 + 2] = mean_val
            _X[i, a3 + 3] = mn
            _X[i, a3 + 4] = pnv

            a1 = b1
            a2 = b2
            a3 += 5

    return _X
