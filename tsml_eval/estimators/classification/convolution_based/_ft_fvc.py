"""Fast transformation-based feature vector concatenation (FT-FVC) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["FTFVCClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._rocket_utils import (
    _check_numba_jobs,
)


@njit(fastmath=True, parallel=True, cache=True)
def _discrete_hilbert_transform(X):
    """Compute the discrete Hilbert transform ``sum_{i != k} s_i / (k - i)``."""
    n_cases, n_timepoints = X.shape
    out = np.zeros((n_cases, n_timepoints), dtype=np.float64)
    inv_diff = np.empty(2 * n_timepoints - 1, dtype=np.float64)
    center = n_timepoints - 1
    for d in range(-(n_timepoints - 1), n_timepoints):
        inv_diff[center + d] = 0.0 if d == 0 else 1.0 / d

    for c in prange(n_cases):
        s = X[c]
        for k in range(n_timepoints):
            acc = 0.0
            offset = center + k
            for i in range(n_timepoints):
                acc += s[i] * inv_diff[offset - i]
            out[c, k] = acc
    return out


class FTFVCClassifier(BaseClassifier):
    """Fast transformation-based feature vector concatenation (FT-FVC) classifier.

    FT-FVC [1]_ standardises each univariate time series and computes three
    time-domain and analytical transformations: the discrete Hilbert transform,
    the first-order difference, and the second-order difference. Separate ROCKET
    transforms with ``n_kernels`` kernels each are applied to the standardised
    raw series and to each of the three transformed representations.

    Rather than concatenating all four feature blocks into a single matrix,
    FT-FVC forms three paired views by concatenating the raw ROCKET features
    with each transformation's ROCKET features in turn, fits an independent
    ridge classifier on each view, and combines test predictions by hard
    majority voting.

    Parameters
    ----------
    n_kernels : int, default=5000
        Number of ROCKET kernels applied to the raw series and to each of the
        three transformed series (``4 * n_kernels`` kernels in total, producing
        ``4 * n_kernels`` features in each of the three paired views).
    n_jobs : int, default=1
        The number of jobs to run in parallel for the ROCKET transforms and
        Hilbert transform. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to seed the four ROCKET transforms.

    Attributes
    ----------
    classifiers_ : list of Pipeline
        The three fitted ``StandardScaler`` and ``RidgeClassifierCV`` pipelines,
        one for each paired view (Hilbert, first difference, second difference).

    Notes
    -----
    Only the supervised ``FT-FVC`` algorithm from [1]_ is implemented; the
    transductive ``Semi-FT-FVC`` variant in the same paper iteratively labels
    test instances during training and is excluded. No reference code was
    published with [1]_. Each view uses ``StandardScaler(with_mean=False)`` and
    ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))``, matching aeon's
    ``RocketClassifier``. Three-way voting ties are broken in the order of
    ``classes_``.

    References
    ----------
    .. [1] He, Changchun, Xin Huo, and Hewei Gao. "FT-FVC: fast
       transformation-based feature vector concatenation for time series
       classification." Applied Intelligence 53, no. 14 (2023): 17778-17795.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     FTFVCClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = FTFVCClassifier(n_kernels=50, random_state=0)
    >>> clf.fit(X, y)
    FTFVCClassifier(n_kernels=50, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(self, n_kernels=5000, n_jobs=1, random_state=None):
        self.n_kernels = n_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _preprocess_views(self, X):
        X = np.nan_to_num(X.astype(np.float64, copy=False), nan=0.0)
        if X.shape[2] < 4:
            X = np.pad(X, ((0, 0), (0, 0), (0, 4 - X.shape[2])))

        X_norm = self._normalizer.transform(X)

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            X_hilbert = _discrete_hilbert_transform(X_norm[:, 0, :])[:, None, :]
        finally:
            set_num_threads(prev_threads)

        X_diff1 = np.diff(X_norm, n=1, axis=-1)
        X_diff2 = np.diff(X_norm, n=2, axis=-1)
        return X_norm, X_hilbert, X_diff1, X_diff2

    def _fit(self, X, y):
        if self.n_kernels < 1:
            raise ValueError(f"n_kernels must be >= 1, got {self.n_kernels}.")

        self._n_jobs = _check_numba_jobs(check_n_jobs(self.n_jobs))
        self._normalizer = Normalizer().fit(X)

        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=4)

        views = self._preprocess_views(X)
        self._rockets = [
            Rocket(
                n_kernels=self.n_kernels,
                normalise=True,
                n_jobs=self._n_jobs,
                random_state=int(seeds[i]),
            )
            for i in range(4)
        ]
        transformed = [
            rocket.fit_transform(view).astype(np.float64)
            for rocket, view in zip(self._rockets, views)
        ]

        _, y_ind = np.unique(y, return_inverse=True)
        v_raw = transformed[0]
        self.classifiers_ = []
        for v_trans in transformed[1:]:
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            clf.fit(np.hstack((v_raw, v_trans)), y_ind)
            self.classifiers_.append(clf)

        return self

    def _predict_proba(self, X):
        n_cases = len(X)
        if self.n_classes_ == 1:
            return np.ones((n_cases, 1), dtype=float)

        views = self._preprocess_views(X)
        transformed = [
            rocket.transform(view).astype(np.float64)
            for rocket, view in zip(self._rockets, views)
        ]
        v_raw = transformed[0]

        votes = np.zeros((n_cases, self.n_classes_), dtype=float)
        rows = np.arange(n_cases)
        for clf, v_trans in zip(self.classifiers_, transformed[1:]):
            preds_ind = clf.predict(np.hstack((v_raw, v_trans)))
            votes[rows, preds_ind] += 1.0

        return votes / len(self.classifiers_)

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

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
        return {"n_kernels": 25}
