"""Shared utilities for the HDC-ROCKET classifiers.

The HDC-ROCKET classifiers bind the pooled convolution outputs of a ROCKET transform
to hyperdimensional vectors that encode the time of each output, using fractional
power encoding (FPE). This module holds the time encoding, the scalers and the shared
fit and predict process, which follow the HDC-ROCKETs reference implementation.
"""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import check_random_state

# the reference beta grid, 2 ** linspace(0, 3, 7) - 1
HDC_BETAS = np.logspace(0, 3, 7, base=2) - 1


def _fpe_poses(n_timepoints, beta, dim, seed, n_columns=None):
    """Return the sinusoidal FPE time encodings of each time point.

    Time points are scaled to ``[0, 1]`` and encoded with the sinusoidal fractional
    power encoding of the reference implementation, using random phases drawn with
    the "sinc" kernel. The vector of a time point ``t`` is ``cos(p * (beta * t + 1))``
    followed by the reversed ``sin(p * (beta * t + 1))``, for the phase vector ``p``.

    Parameters
    ----------
    n_timepoints : int
        Number of time points to encode.
    beta : float
        Bandwidth of the encoding. Larger values make the encodings of nearby time
        points less similar. With ``beta=0`` every time point has the same encoding.
    dim : int
        Dimensionality of the hyperdimensional vectors.
    seed : int or None
        Seed for the random phases.
    n_columns : int or None, default=None
        Return only the first ``n_columns`` dimensions of each vector. All
        ``2 * (dim // 2)`` dimensions are returned if None.

    Returns
    -------
    poses : np.ndarray
        The encodings, of shape ``(n_columns, n_timepoints)`` with one row per
        dimension.
    """
    rng = np.random.RandomState(seed)
    phases = rng.rand(dim) * 2 * np.pi - np.pi
    if dim % 2 == 1:
        phasor = phases[: (dim - 1) // 2]
    else:
        phasor = np.concatenate(([0.0], phases[: (dim - 2) // 2]))

    half = len(phasor)
    if n_columns is None:
        n_columns = 2 * half
    if n_columns > 2 * half:
        raise ValueError(
            f"The transform needs {n_columns} HDC dimensions but only {2 * half} are "
            f"available, increase the HDC dimension."
        )

    # time points are float32 in the reference, the encoding is computed in float64
    timepoints = np.linspace(0, 1, n_timepoints, dtype=np.float32)
    exponent = np.float64(beta) * timepoints.astype(np.float64) + 1
    columns = np.arange(n_columns)
    cos_columns = columns[columns < half]
    sin_columns = columns[columns >= half]

    poses = np.empty((n_columns, n_timepoints), dtype=np.float32)
    poses[cos_columns] = np.cos(np.outer(phasor[cos_columns], exponent))
    poses[sin_columns] = np.sin(np.outer(phasor[2 * half - 1 - sin_columns], exponent))
    return poses


class _SparseScaler(TransformerMixin, BaseEstimator):
    """The Hydra sparse scaler, applied to the columns from ``start`` onwards.

    Columns are transformed with a signed square root and standardised, with zero
    values kept at zero. Columns before ``start`` are returned unchanged. As in the
    reference, the Hydra features are stored in half precision, and the square root
    and mean are computed in half precision.
    """

    def __init__(self, start=0, exponent=4):
        self.start = start
        self.exponent = exponent

    def fit(self, X, y=None):
        X = self._sqrt(X[:, self.start :])
        epsilon = (X == 0).mean(0) ** self.exponent + 1e-8
        self.mu_ = X.mean(0)
        self.sigma_ = X.std(0) + epsilon
        return self

    def transform(self, X):
        Xs = self._sqrt(X[:, self.start :])
        Xs = ((Xs - self.mu_) * (Xs != 0)) / self.sigma_
        X = np.hstack((X[:, : self.start], Xs))
        X[~np.isfinite(X)] = 0
        return X

    @staticmethod
    def _sqrt(X):
        X = X.astype(np.float16)
        return np.sqrt(np.abs(X)) * np.sign(X)


class _BaseHDCRocketClassifier(BaseClassifier):
    """Base class for the HDC-ROCKET classifiers.

    Fitting follows the "hdc_auto" variant of the reference implementation. Each
    channel is z-normalised using the mean and standard deviation of the training
    data, and series are zero padded to the minimum length of the transform. For each
    candidate ``beta``, the training data is transformed with the time encodings for
    that ``beta``, scaled, and used to fit a ``RidgeClassifierCV``. The ``beta`` with
    the best leave-one-out score of the ridge classifier is kept, along with its
    scaling and ridge classifier.

    Subclasses implement ``_fit_kernels`` to fit the convolution parameters,
    ``_transform_hdc`` to apply the transform with a set of time encodings, and set
    ``_hdc_dim_param``, the name of the parameter holding the HDC dimension.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    _min_length = 9
    _hdc_dim_param = None

    def __init__(self, beta=None, n_jobs=1, random_state=None):
        self.beta = beta
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        self._seed = (
            rng.randint(np.iinfo(np.int32).max)
            if self.random_state is None
            or isinstance(self.random_state, np.random.RandomState)
            else self.random_state
        )

        X = self._remove_nans(X)
        self._channel_mean = np.zeros(X.shape[1])
        self._channel_std = np.ones(X.shape[1])
        for i in range(X.shape[1]):
            self._channel_mean[i] = np.nanmean(X[:, i])
            std = np.nanstd(X[:, i])
            self._channel_std[i] = 1 if std == 0 else std
        X = self._preprocess(X)
        self._n_timepoints = X.shape[2]

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        self._fit_kernels(X)

        betas = HDC_BETAS if self.beta is None else [self.beta]
        self.beta_scores_ = np.zeros(len(betas))
        best_score = -np.inf
        for i, beta in enumerate(betas):
            poses = self._get_poses(beta)
            scaler = self._make_scaler()
            Xt = scaler.fit_transform(self._transform_hdc(X, poses))
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Xt, y)

            self.beta_scores_[i] = ridge.best_score_
            if ridge.best_score_ > best_score:
                best_score = ridge.best_score_
                self.beta_ = beta
                self._poses = poses
                self._scaler = scaler
                self.classifier_ = ridge

        set_num_threads(prev_threads)
        return self

    def _predict(self, X):
        X = self._preprocess(self._remove_nans(X))

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        Xt = self._scaler.transform(self._transform_hdc(X, self._poses))
        set_num_threads(prev_threads)

        return self.classifier_.predict(Xt)

    @staticmethod
    def _remove_nans(X):
        X = np.array(X, dtype=np.float64)
        X[np.isnan(X)] = 0
        return X

    def _preprocess(self, X):
        """Normalise each channel and pad, keeping double precision.

        The transforms cast to single precision, after differencing where used.
        """
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - self._channel_mean[i]) / self._channel_std[i]
        if X.shape[2] < self._min_length:
            X = np.pad(X, ((0, 0), (0, 0), (0, self._min_length - X.shape[2])))
        return X

    def _get_poses(self, beta):
        return _fpe_poses(
            self._n_timepoints,
            beta,
            getattr(self, self._hdc_dim_param),
            self._seed,
            n_columns=self._n_hdc_columns(),
        )

    def _make_scaler(self):
        return make_pipeline(
            MinMaxScaler(feature_range=(-1, 1), clip=True), StandardScaler()
        )

    def _n_hdc_columns(self):
        raise NotImplementedError("abstract method")

    def _fit_kernels(self, X):
        raise NotImplementedError("abstract method")

    def _transform_hdc(self, X, poses):
        raise NotImplementedError("abstract method")
