"""Kernel Grouping with Multiple Transformations and Pooling (KG-MTP) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["KGMTPClassifier"]

from itertools import combinations

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from scipy import fftpack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._rocket_utils import (
    _check_numba_jobs,
    _L2NormalisedRidgeClassifierCV,
    _quantiles,
)

_INDICES_1 = np.arange(6, dtype=np.int32)
_INDICES_2 = np.array(list(combinations(range(6), 2)), dtype=np.int32)
_INDICES_3 = np.array(list(combinations(range(6), 3)), dtype=np.int32)
_INDICES_4 = np.array(list(combinations(range(6), 4)), dtype=np.int32)
_INDICES_5 = np.array(list(combinations(range(6), 5)), dtype=np.int32)


def _fit_dilations(input_length, n_features_budget, max_dilations_per_kernel):
    n_kernels = 62
    n_features_per_kernel = n_features_budget // n_kernels
    true_max_dilations = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max_dilations

    max_exponent = np.log2((input_length - 1) / (6 - 1))
    dilations, n_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations, base=2).astype(np.int32),
        return_counts=True,
    )
    n_features_per_dilation = (n_features_per_dilation * multiplier).astype(np.int32)

    remainder = n_features_per_kernel - int(np.sum(n_features_per_dilation))
    i = 0
    while remainder > 0:
        n_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(n_features_per_dilation)

    return dilations, n_features_per_dilation


@njit(fastmath=True, cache=True)
def _convolve_shift(A, G, padded_length, dilation, padding):
    C_alpha = np.zeros(padded_length, dtype=np.float64)
    C_alpha[:] = A
    C_gamma = np.zeros((6, padded_length), dtype=np.float64)
    C_gamma[3] = G

    start = dilation
    end = padded_length - padding
    for gamma_index in range(3):
        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]
        end += dilation
    for gamma_index in range(4, 6):
        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]
        start += dilation
    return C_alpha, C_gamma


@njit(fastmath=True, cache=True)
def _fit_biases(
    X,
    dilations,
    n_features_per_dilation,
    quantiles,
    ind1,
    ind2,
    ind3,
    ind4,
    ind5,
    seed,
):
    if seed >= 0:
        np.random.seed(seed)

    n_cases, raw_len = X.shape
    n_dilations = len(dilations)
    n_features = 62 * np.sum(n_features_per_dilation)
    biases = np.zeros(n_features, dtype=np.float32)
    f_start = 0

    for d_idx in range(n_dilations):
        dilation = dilations[d_idx]
        padded_len = raw_len + dilation
        padding = (6 * dilation) // 2
        n_feat_d = n_features_per_dilation[d_idx]

        A = np.zeros(padded_len, dtype=np.float64)
        G = np.zeros(padded_len, dtype=np.float64)

        # Group 1: 6 kernels, weights {-1, 5}
        for k in range(6):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            A[:-dilation] = -x
            G[:-dilation] = x + x + x + x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            C = (C_alpha + C_gamma[ind1[k]]).astype(np.float32)
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

        # Group 2: 15 kernels, weights {-1, 2}
        for k in range(15):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            A[:-dilation] = -x
            G[:-dilation] = x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            i0, i1 = ind2[k]
            C = (C_alpha + C_gamma[i0] + C_gamma[i1]).astype(np.float32)
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

        # Group 3: 20 kernels, weights {-1, 1}
        for k in range(20):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            A[:-dilation] = -x
            G[:-dilation] = x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            i0, i1, i2 = ind3[k]
            C = (C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]).astype(np.float32)
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

        # Group 4: 15 kernels, weights {-2, 1}
        for k in range(15):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            A[:-dilation] = -x - x
            G[:-dilation] = x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            i0, i1, i2, i3 = ind4[k]
            C = (
                C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3]
            ).astype(np.float32)
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

        # Group 5: 6 kernels, weights {-5, 1}
        for k in range(6):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            A[:-dilation] = -x - x - x - x - x
            G[:-dilation] = x + x + x + x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            i0, i1, i2, i3, i4 = ind5[k]
            C = (
                C_alpha
                + C_gamma[i0]
                + C_gamma[i1]
                + C_gamma[i2]
                + C_gamma[i3]
                + C_gamma[i4]
            ).astype(np.float32)
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

    return biases


@njit(fastmath=True, cache=True)
def _pool_kernel(
    C,
    k_idx,
    f_start,
    n_feat_d,
    n_features,
    biases,
    C_max,
    C_min,
    C_max_idx,
    C_min_idx,
    out_row,
):
    c_len = C.shape[0]
    for j in range(c_len):
        val = C[j]
        if C_max[j] <= val:
            C_max[j] = val
            C_max_idx[j] = k_idx
        if C_min[j] >= val:
            C_min[j] = val
            C_min_idx[j] = k_idx

    for fc in range(n_feat_d):
        f_idx = f_start + fc
        bias = biases[f_idx]
        ppv = 0
        last_val = 0
        max_stretch = 0.0
        mean_index = 0
        mean = 0.0
        zero_count = 0

        for j in range(c_len):
            if j < c_len - 1 and (
                (C[j] > bias and C[j + 1] < bias) or (C[j] < bias and C[j + 1] > bias)
            ):
                zero_count += 1
            if C[j] > bias:
                ppv += 1
                mean_index += j
                mean += C[j] + bias
            elif C[j] < bias:
                stretch = j - last_val
                if stretch > max_stretch:
                    max_stretch = stretch
                last_val = j
        stretch = c_len - 1 - last_val
        if stretch > max_stretch:
            max_stretch = stretch

        out_row[f_idx] = ppv / c_len
        out_row[f_idx + n_features] = mean / ppv if ppv > 0 else 0.0
        out_row[f_idx + 2 * n_features] = max_stretch
        out_row[f_idx + 3 * n_features] = mean_index / ppv if ppv > 0 else -1.0
        out_row[f_idx + 4 * n_features] = zero_count / c_len


@njit(fastmath=True, parallel=True, cache=True)
def _transform_single(
    X, dilations, n_features_per_dilation, biases, ind1, ind2, ind3, ind4, ind5
):
    n_cases, raw_len = X.shape
    n_kernels = 62
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)

    features = np.zeros((n_cases, 5 * n_features), dtype=np.float32)
    features_hydra = np.zeros((n_cases, 2 * n_kernels * n_dilations), dtype=np.float32)

    for c in prange(n_cases):
        x = X[c]
        f_start = 0
        h_idx = 0

        for d_idx in range(n_dilations):
            dilation = dilations[d_idx]
            padded_len = raw_len + dilation
            padding = (6 * dilation) // 2
            n_feat_d = n_features_per_dilation[d_idx]

            A = np.zeros(padded_len, dtype=np.float64)
            G = np.zeros(padded_len, dtype=np.float64)

            # Group 1 (6 kernels)
            C_max_1 = np.full(padded_len, -np.inf, dtype=np.float64)
            C_min_1 = np.full(padded_len, np.inf, dtype=np.float64)
            C_max_idx_1 = np.zeros(padded_len, dtype=np.int32)
            C_min_idx_1 = np.zeros(padded_len, dtype=np.int32)
            A[:-dilation] = -x
            G[:-dilation] = x + x + x + x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            for k in range(6):
                C = (C_alpha + C_gamma[ind1[k]]).astype(np.float32)
                _pool_kernel(
                    C,
                    k,
                    f_start,
                    n_feat_d,
                    n_features,
                    biases,
                    C_max_1,
                    C_min_1,
                    C_max_idx_1,
                    C_min_idx_1,
                    features[c],
                )
                f_start += n_feat_d

            # Group 2 (15 kernels)
            C_max_2 = np.full(padded_len, -np.inf, dtype=np.float64)
            C_min_2 = np.full(padded_len, np.inf, dtype=np.float64)
            C_max_idx_2 = np.zeros(padded_len, dtype=np.int32)
            C_min_idx_2 = np.zeros(padded_len, dtype=np.int32)
            A[:-dilation] = -x
            G[:-dilation] = x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            for k in range(15):
                i0, i1 = ind2[k]
                C = (C_alpha + C_gamma[i0] + C_gamma[i1]).astype(np.float32)
                _pool_kernel(
                    C,
                    k,
                    f_start,
                    n_feat_d,
                    n_features,
                    biases,
                    C_max_2,
                    C_min_2,
                    C_max_idx_2,
                    C_min_idx_2,
                    features[c],
                )
                f_start += n_feat_d

            # Group 3 (20 kernels)
            C_max_3 = np.full(padded_len, -np.inf, dtype=np.float64)
            C_min_3 = np.full(padded_len, np.inf, dtype=np.float64)
            C_max_idx_3 = np.zeros(padded_len, dtype=np.int32)
            C_min_idx_3 = np.zeros(padded_len, dtype=np.int32)
            A[:-dilation] = -x
            G[:-dilation] = x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            for k in range(20):
                i0, i1, i2 = ind3[k]
                C = (C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]).astype(
                    np.float32
                )
                _pool_kernel(
                    C,
                    k,
                    f_start,
                    n_feat_d,
                    n_features,
                    biases,
                    C_max_3,
                    C_min_3,
                    C_max_idx_3,
                    C_min_idx_3,
                    features[c],
                )
                f_start += n_feat_d

            # Group 4 (15 kernels)
            C_max_4 = np.full(padded_len, -np.inf, dtype=np.float64)
            C_min_4 = np.full(padded_len, np.inf, dtype=np.float64)
            C_max_idx_4 = np.zeros(padded_len, dtype=np.int32)
            C_min_idx_4 = np.zeros(padded_len, dtype=np.int32)
            A[:-dilation] = -x - x
            G[:-dilation] = x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            for k in range(15):
                i0, i1, i2, i3 = ind4[k]
                C = (
                    C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3]
                ).astype(np.float32)
                _pool_kernel(
                    C,
                    k,
                    f_start,
                    n_feat_d,
                    n_features,
                    biases,
                    C_max_4,
                    C_min_4,
                    C_max_idx_4,
                    C_min_idx_4,
                    features[c],
                )
                f_start += n_feat_d

            # Group 5 (6 kernels)
            C_max_5 = np.full(padded_len, -np.inf, dtype=np.float64)
            C_min_5 = np.full(padded_len, np.inf, dtype=np.float64)
            C_max_idx_5 = np.zeros(padded_len, dtype=np.int32)
            C_min_idx_5 = np.zeros(padded_len, dtype=np.int32)
            A[:-dilation] = -x - x - x - x - x
            G[:-dilation] = x + x + x + x + x + x
            C_alpha, C_gamma = _convolve_shift(A, G, padded_len, dilation, padding)
            for k in range(6):
                i0, i1, i2, i3, i4 = ind5[k]
                C = (
                    C_alpha
                    + C_gamma[i0]
                    + C_gamma[i1]
                    + C_gamma[i2]
                    + C_gamma[i3]
                    + C_gamma[i4]
                ).astype(np.float32)
                _pool_kernel(
                    C,
                    k,
                    f_start,
                    n_feat_d,
                    n_features,
                    biases,
                    C_max_5,
                    C_min_5,
                    C_max_idx_5,
                    C_min_idx_5,
                    features[c],
                )
                f_start += n_feat_d

            h_max_1 = np.zeros(6, dtype=np.float32)
            h_min_1 = np.zeros(6, dtype=np.float32)
            h_max_2 = np.zeros(15, dtype=np.float32)
            h_min_2 = np.zeros(15, dtype=np.float32)
            h_max_3 = np.zeros(20, dtype=np.float32)
            h_min_3 = np.zeros(20, dtype=np.float32)
            h_max_4 = np.zeros(15, dtype=np.float32)
            h_min_4 = np.zeros(15, dtype=np.float32)
            h_max_5 = np.zeros(6, dtype=np.float32)
            h_min_5 = np.zeros(6, dtype=np.float32)

            for t in range(padded_len):
                h_max_1[C_max_idx_1[t]] += np.float32(C_max_1[t])
                h_min_1[C_min_idx_1[t]] += 1.0
                h_max_2[C_max_idx_2[t]] += np.float32(C_max_2[t])
                h_min_2[C_min_idx_2[t]] += 1.0
                h_max_3[C_max_idx_3[t]] += np.float32(C_max_3[t])
                h_min_3[C_min_idx_3[t]] += 1.0
                h_max_4[C_max_idx_4[t]] += np.float32(C_max_4[t])
                h_min_4[C_min_idx_4[t]] += 1.0
                h_max_5[C_max_idx_5[t]] += np.float32(C_max_5[t])
                h_min_5[C_min_idx_5[t]] += 1.0

            features_hydra[c, h_idx : h_idx + 6] = h_max_1
            features_hydra[c, h_idx + 6 : h_idx + 21] = h_max_2
            features_hydra[c, h_idx + 21 : h_idx + 41] = h_max_3
            features_hydra[c, h_idx + 41 : h_idx + 56] = h_max_4
            features_hydra[c, h_idx + 56 : h_idx + 62] = h_max_5
            h_idx += n_kernels

            features_hydra[c, h_idx : h_idx + 6] = h_min_1
            features_hydra[c, h_idx + 6 : h_idx + 21] = h_min_2
            features_hydra[c, h_idx + 21 : h_idx + 41] = h_min_3
            features_hydra[c, h_idx + 41 : h_idx + 56] = h_min_4
            features_hydra[c, h_idx + 56 : h_idx + 62] = h_min_5
            h_idx += n_kernels

    return features, features_hydra


class _KGSparseScaler(TransformerMixin, BaseEstimator):
    """Hydra sparse scaler matching ``KG-MTP/utils/tools.py`` in double precision."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        """Fit sparse scaler parameters on ``X``."""
        Xs = np.sqrt(np.maximum(X.astype(np.float64), 0.0))
        epsilon = (Xs == 0).mean(0) ** self.exponent + 1e-8
        self.mu_ = Xs.mean(0)
        ddof = 1 if len(Xs) > 1 else 0
        self.sigma_ = Xs.std(0, ddof=ddof) + epsilon
        return self

    def transform(self, X):
        """Transform ``X`` using the fitted sparse scaler."""
        Xs = np.sqrt(np.maximum(X.astype(np.float64), 0.0))
        if self.mask:
            out = ((Xs - self.mu_) * (Xs != 0)) / self.sigma_
        else:
            out = (Xs - self.mu_) / self.sigma_
        return out.astype(np.float32)


class KGMTPClassifier(BaseClassifier):
    """Kernel Grouping with Multiple Transformations and Pooling (KG-MTP) classifier.

    KG-MTP [1]_ extracts features from three representations of a univariate
    time series: the raw series, its discrete Hilbert transform (``fftpack``),
    and its first-order difference. All 62 non-constant binary-valued kernels of
    length 6 are partitioned by weight ratio into five groups of sizes 6, 15,
    20, 15, and 6.

    For each representation and dilation, KG-MTP computes five MultiRocket-style
    pooling operators (``PPV``, ``MPV``, ``LSPV``, ``MIPV``, and zero-crossing
    rate ``ZCR``) alongside Hydra-style per-group soft maximum and hard minimum
    kernel response counts. The kernel response features are scaled with a
    Hydra sparse scaler, concatenated with the pooling features, standardised,
    and classified with an ``l_2``-normalised ``RidgeClassifierCV``.

    This follows [1]_ and the reference implementation [2]_.

    Parameters
    ----------
    n_features : int, default=50000
        Target number of pooling features across all three representations
        (``(n_features // 3) // 5`` kernel-dilation biases per representation;
        must be at least 930 so each of the 62 kernels receives at least one
        dilation).
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per kernel.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the Numba transforms. ``-1``
        means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used when sampling training series for
        bias quantiles.

    Attributes
    ----------
    classifier_ : Pipeline
        Fitted ``StandardScaler`` and ``_L2NormalisedRidgeClassifierCV``
        pipeline.

    Notes
    -----
    Three bugs in the reference implementation [2]_ are fixed:

    1. ``models/KGMTP.py`` initialises per-group ``C_max`` and ``C_min`` arrays
       to ``-10000`` and ``10000`` and ``C_max_index`` / ``C_min_index`` to
       ``1``, silently corrupting kernel response counts whenever unnormalised
       convolutions exceed ``10000`` in magnitude. ``-np.inf``, ``np.inf``, and
       index ``0`` are used instead, with no change on standard z-normalised UCR
       datasets.
    2. ``utils/tools.py`` computes ``SparseScaler`` mean and sample standard
       deviation in single precision via PyTorch, which can overflow on
       high-amplitude inputs. Statistics are computed in ``float64`` (with
       ``ddof=1`` to match ``torch.Tensor.std``) and cast back to ``float32``,
       removing the PyTorch dependency while producing identical predictions on
       all tested UCR datasets.
    3. ``_fit_biases`` in [2]_ draws random training instances without seeding
       Numba's RNG, and ``transform`` nests ``prange`` loops inside an outer
       ``prange``. Numba's RNG is seeded from ``random_state`` for each
       representation and only the outer instance loop is parallelised.

    References
    ----------
    .. [1] Wang, Panjie, Jiang Wu, Yuan Wei, and Taiyong Li. "Kernel grouping
       for time series classification with multiple transformations and pooling
       operators." Expert Systems with Applications 293 (2025): 128693.
    .. [2] https://github.com/WangPanJie2024/KG-MTP

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     KGMTPClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = KGMTPClassifier(n_features=930, random_state=0)
    >>> clf.fit(X, y)
    KGMTPClassifier(n_features=930, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_features=50000,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=None,
    ):
        self.n_features = n_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    @staticmethod
    def _preprocess_representations(X):
        X_raw = np.nan_to_num(X[:, 0, :].astype(np.float64, copy=False), nan=0.0)
        if X_raw.shape[1] < 7:
            X_raw = np.pad(X_raw, ((0, 0), (0, 7 - X_raw.shape[1])))

        X_hilbert = np.empty(X_raw.shape, dtype=np.float32)
        for i in range(X_raw.shape[0]):
            X_hilbert[i] = fftpack.hilbert(X_raw[i])
        X_hilbert = X_hilbert.astype(np.float64)

        X_diff = np.diff(X_raw, 1)
        return X_raw, X_hilbert, X_diff

    def _fit_single_rep(self, X_rep, n_kernels_budget, seed):
        dilations, n_features_per_dilation = _fit_dilations(
            X_rep.shape[1], n_kernels_budget, self.max_dilations_per_kernel
        )
        quantiles = _quantiles(62 * int(np.sum(n_features_per_dilation)))
        biases = _fit_biases(
            X_rep,
            dilations,
            n_features_per_dilation,
            quantiles,
            _INDICES_1,
            _INDICES_2,
            _INDICES_3,
            _INDICES_4,
            _INDICES_5,
            seed,
        )
        return dilations, n_features_per_dilation, biases

    def _transform_all(self, X, fit_sparse=False):
        reps = self._preprocess_representations(X)
        pool_blocks = []
        hydra_blocks = []

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            for rep, params in zip(reps, self._rep_params):
                dilations, n_feat_per_dil, biases = params
                f_pool, f_hydra = _transform_single(
                    rep,
                    dilations,
                    n_feat_per_dil,
                    biases,
                    _INDICES_1,
                    _INDICES_2,
                    _INDICES_3,
                    _INDICES_4,
                    _INDICES_5,
                )
                pool_blocks.append(np.nan_to_num(f_pool))
                hydra_blocks.append(f_hydra)
        finally:
            set_num_threads(prev_threads)

        X_pool = np.hstack(pool_blocks)
        X_hydra = np.hstack(hydra_blocks)
        if fit_sparse:
            self._sparse_scaler = _KGSparseScaler().fit(X_hydra)
        X_hydra_scaled = self._sparse_scaler.transform(X_hydra)
        return np.hstack((X_pool, X_hydra_scaled))

    def _fit(self, X, y):
        if self.n_features < 930:
            raise ValueError(f"n_features must be >= 930, got {self.n_features}.")
        if self.max_dilations_per_kernel < 1:
            raise ValueError(
                "max_dilations_per_kernel must be >= 1, "
                f"got {self.max_dilations_per_kernel}."
            )

        self._n_jobs = _check_numba_jobs(check_n_jobs(self.n_jobs))
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=3)

        n_kernels_budget = (self.n_features // 3) // 5
        reps = self._preprocess_representations(X)
        self._rep_params = [
            self._fit_single_rep(rep, n_kernels_budget, int(s))
            for rep, s in zip(reps, seeds)
        ]

        Xt = self._transform_all(X, fit_sparse=True)
        if self.n_classes_ == 1:
            self.classifier_ = None
            return self

        _, y_ind = np.unique(y, return_inverse=True)
        self.classifier_ = make_pipeline(
            StandardScaler(),
            _L2NormalisedRidgeClassifierCV(),
        ).fit(Xt, y_ind)
        return self

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        Xt = self._transform_all(X, fit_sparse=False)
        preds_ind = self.classifier_.predict(Xt)
        return self.classes_[preds_ind]

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
        return {"n_features": 930}
