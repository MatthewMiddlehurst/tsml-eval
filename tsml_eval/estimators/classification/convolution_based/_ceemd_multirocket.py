"""CEEMD with Improved MultiRocket (CEEMD-MultiRocket) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["CEEMDMultiRocketClassifier"]

from itertools import combinations

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _check_numba_jobs,
    _L2NormalisedRidgeClassifierCV,
    _quiet_ridge_warnings,
)

_INDICES_6_2 = np.array(list(combinations(range(6), 2)), dtype=np.int32)


@njit(fastmath=True, cache=True)
def _find_extrema(h):
    """Return indices of local maxima and local minima, handling flat plateaus."""
    n = h.shape[0]
    max_idx = np.empty(n, dtype=np.int32)
    min_idx = np.empty(n, dtype=np.int32)
    n_max = 0
    n_min = 0

    i = 1
    while i < n - 1:
        if h[i] > h[i - 1]:
            j = i
            while j < n - 1 and h[j] == h[j + 1]:
                j += 1
            if j < n - 1 and h[j] > h[j + 1]:
                max_idx[n_max] = (i + j) // 2
                n_max += 1
            i = j + 1
        elif h[i] < h[i - 1]:
            j = i
            while j < n - 1 and h[j] == h[j + 1]:
                j += 1
            if j < n - 1 and h[j] < h[j + 1]:
                min_idx[n_min] = (i + j) // 2
                n_min += 1
            i = j + 1
        else:
            i += 1

    return max_idx[:n_max], min_idx[:n_min]


@njit(fastmath=True, cache=True)
def _eval_natural_cubic_spline(xk, yk, n_out):
    """Evaluate natural cubic spline at integer points ``0 .. n_out - 1``."""
    m = xk.shape[0]
    out = np.empty(n_out, dtype=np.float64)
    if m == 2:
        slope = (yk[1] - yk[0]) / (xk[1] - xk[0])
        for t in range(n_out):
            out[t] = yk[0] + slope * (t - xk[0])
        return out

    dx = np.empty(m - 1, dtype=np.float64)
    slope = np.empty(m - 1, dtype=np.float64)
    for i in range(m - 1):
        dx[i] = xk[i + 1] - xk[i]
        slope[i] = (yk[i + 1] - yk[i]) / dx[i]

    # Tridiagonal Thomas solve for natural spline second derivatives M[1..m-2]
    c_prime = np.zeros(m, dtype=np.float64)
    d_prime = np.zeros(m, dtype=np.float64)
    for i in range(1, m - 1):
        diag = 2.0 * (dx[i - 1] + dx[i])
        rhs = 6.0 * (slope[i] - slope[i - 1])
        denom = diag - dx[i - 1] * c_prime[i - 1]
        c_prime[i] = dx[i] / denom
        d_prime[i] = (rhs - dx[i - 1] * d_prime[i - 1]) / denom

    M = np.zeros(m, dtype=np.float64)
    for i in range(m - 2, 0, -1):
        M[i] = d_prime[i] - c_prime[i] * M[i + 1]

    seg = 0
    for t in range(n_out):
        while seg < m - 2 and t > xk[seg + 1]:
            seg += 1
        h = dx[seg]
        a = (xk[seg + 1] - t) / h
        b = (t - xk[seg]) / h
        out[t] = (
            a * yk[seg]
            + b * yk[seg + 1]
            + ((a * a * a - a) * M[seg] + (b * b * b - b) * M[seg + 1]) * (h * h) / 6.0
        )
    return out


@njit(fastmath=True, cache=True)
def _sift_imf(signal, n_siftings):
    """Extract one IMF with Wu & Huang endpoint-clamped cubic spline sifting."""
    n = signal.shape[0]
    h = signal.copy()

    for _ in range(n_siftings):
        max_idx, min_idx = _find_extrema(h)
        n_max = max_idx.shape[0]
        n_min = min_idx.shape[0]
        if n_max < 1 or n_min < 1 or (n_max + n_min) < 3:
            break

        # Build upper-envelope knots with Wu & Huang endpoint extrapolation
        xk_u = np.empty(n_max + 2, dtype=np.float64)
        yk_u = np.empty(n_max + 2, dtype=np.float64)
        xk_u[0] = 0.0
        xk_u[n_max + 1] = n - 1.0
        for i in range(n_max):
            xk_u[i + 1] = max_idx[i]
            yk_u[i + 1] = h[max_idx[i]]
        if n_max >= 2:
            left_u = yk_u[1] - (yk_u[2] - yk_u[1]) * xk_u[1] / (xk_u[2] - xk_u[1])
            right_u = yk_u[n_max] + (yk_u[n_max] - yk_u[n_max - 1]) * (
                (n - 1.0) - xk_u[n_max]
            ) / (xk_u[n_max] - xk_u[n_max - 1])
            yk_u[0] = max(h[0], left_u)
            yk_u[n_max + 1] = max(h[n - 1], right_u)
        else:
            yk_u[0] = max(h[0], yk_u[1])
            yk_u[n_max + 1] = max(h[n - 1], yk_u[1])

        # Build lower-envelope knots
        xk_l = np.empty(n_min + 2, dtype=np.float64)
        yk_l = np.empty(n_min + 2, dtype=np.float64)
        xk_l[0] = 0.0
        xk_l[n_min + 1] = n - 1.0
        for i in range(n_min):
            xk_l[i + 1] = min_idx[i]
            yk_l[i + 1] = h[min_idx[i]]
        if n_min >= 2:
            left_l = yk_l[1] - (yk_l[2] - yk_l[1]) * xk_l[1] / (xk_l[2] - xk_l[1])
            right_l = yk_l[n_min] + (yk_l[n_min] - yk_l[n_min - 1]) * (
                (n - 1.0) - xk_l[n_min]
            ) / (xk_l[n_min] - xk_l[n_min - 1])
            yk_l[0] = min(h[0], left_l)
            yk_l[n_min + 1] = min(h[n - 1], right_l)
        else:
            yk_l[0] = min(h[0], yk_l[1])
            yk_l[n_min + 1] = min(h[n - 1], yk_l[1])

        env_u = _eval_natural_cubic_spline(xk_u, yk_u, n)
        env_l = _eval_natural_cubic_spline(xk_l, yk_l, n)
        for t in range(n):
            h[t] -= 0.5 * (env_u[t] + env_l[t])

    return h


@njit(fastmath=True, parallel=True, cache=True)
def _ceemd_batch(X, noise_bank, noise_std, n_siftings):
    """Decompose each series in ``X`` into ``(IMF1, IMF2, Residue)`` via CEEMD."""
    n_cases, n_timepoints = X.shape
    n_pairs = noise_bank.shape[0]
    imf1 = np.zeros((n_cases, n_timepoints), dtype=np.float64)
    imf2 = np.zeros((n_cases, n_timepoints), dtype=np.float64)
    residue = np.zeros((n_cases, n_timepoints), dtype=np.float64)
    inv_2n = 1.0 / (2.0 * n_pairs)

    for c in prange(n_cases):
        x = X[c]
        s_std = np.std(x)
        amp = noise_std * s_std
        if amp == 0.0:
            c1 = _sift_imf(x, n_siftings)
            r1 = x - c1
            c2 = _sift_imf(r1, n_siftings)
            imf1[c] = c1
            imf2[c] = c2
            residue[c] = r1 - c2
            continue

        acc1 = np.zeros(n_timepoints, dtype=np.float64)
        acc2 = np.zeros(n_timepoints, dtype=np.float64)
        p_seq = np.empty(n_timepoints, dtype=np.float64)
        n_seq = np.empty(n_timepoints, dtype=np.float64)

        for r in range(n_pairs):
            w = noise_bank[r]
            for t in range(n_timepoints):
                nt = amp * w[t]
                p_seq[t] = x[t] + nt
                n_seq[t] = x[t] - nt

            c1_p = _sift_imf(p_seq, n_siftings)
            c2_p = _sift_imf(p_seq - c1_p, n_siftings)

            c1_n = _sift_imf(n_seq, n_siftings)
            c2_n = _sift_imf(n_seq - c1_n, n_siftings)

            for t in range(n_timepoints):
                acc1[t] += c1_p[t] + c1_n[t]
                acc2[t] += c2_p[t] + c2_n[t]

        for t in range(n_timepoints):
            m1 = acc1[t] * inv_2n
            m2 = acc2[t] * inv_2n
            imf1[c, t] = m1
            imf2[c, t] = m2
            residue[c, t] = x[t] - m1 - m2

    return imf1, imf2, residue


def _fit_dilations_6_2(input_length, n_features_budget, max_dilations_per_kernel):
    n_kernels = 15
    n_features_per_kernel = max(1, n_features_budget // n_kernels)
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


def _quantiles(n):
    phi = (np.sqrt(5) + 1) / 2
    return np.array([((i * phi) % 1) for i in range(1, n + 1)], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _convolve_6_2(x, dilation):
    raw_len = x.shape[0]
    padded_len = raw_len + dilation
    padding = (6 * dilation) // 2

    A = np.zeros(padded_len, dtype=np.float64)
    G = np.zeros(padded_len, dtype=np.float64)
    A[:-dilation] = -x
    G[:-dilation] = x + x + x

    C_alpha = np.zeros(padded_len, dtype=np.float64)
    C_alpha[:] = A
    C_gamma = np.zeros((6, padded_len), dtype=np.float64)
    C_gamma[3] = G

    start = dilation
    end = padded_len - padding
    for gamma_index in range(3):
        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]
        end += dilation
    for gamma_index in range(4, 6):
        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]
        start += dilation

    return C_alpha, C_gamma, padding


@njit(fastmath=True, cache=True)
def _fit_biases_6_2(X, dilations, n_features_per_dilation, quantiles, indices, seed):
    if seed >= 0:
        np.random.seed(seed)

    n_cases = X.shape[0]
    n_kernels = indices.shape[0]
    n_features = n_kernels * np.sum(n_features_per_dilation)
    biases = np.zeros(n_features, dtype=np.float32)

    f_start = 0
    for d_idx in range(len(dilations)):
        dilation = dilations[d_idx]
        n_feat_d = n_features_per_dilation[d_idx]
        _padding0 = d_idx % 2

        for k in range(n_kernels):
            f_end = f_start + n_feat_d
            x = X[np.random.randint(n_cases)]
            C_alpha, C_gamma, padding = _convolve_6_2(x, dilation)
            i0, i1 = indices[k]
            C = (C_alpha + C_gamma[i0] + C_gamma[i1]).astype(np.float32)
            if (_padding0 + k) % 2 == 1:
                # Valid (unpadded) region of length raw_len - 5 * dilation
                C = C[padding : C.shape[0] - padding]
            biases[f_start:f_end] = np.quantile(C, quantiles[f_start:f_end])
            f_start = f_end

    return biases


@njit(fastmath=True, parallel=True, cache=True)
def _transform_6_2(X, dilations, n_features_per_dilation, biases, indices):
    """Extract PPV, MPV, MIPV, LSPV and NSPV features with 15 length-6 kernels."""
    n_cases = X.shape[0]
    n_kernels = indices.shape[0]
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, 5 * n_features), dtype=np.float32)

    for c in prange(n_cases):
        x = X[c]
        f_start = 0

        for d_idx in range(len(dilations)):
            dilation = dilations[d_idx]
            n_feat_d = n_features_per_dilation[d_idx]
            _padding0 = d_idx % 2
            C_alpha, C_gamma, padding = _convolve_6_2(x, dilation)

            for k in range(n_kernels):
                f_end = f_start + n_feat_d
                i0, i1 = indices[k]
                C = (C_alpha + C_gamma[i0] + C_gamma[i1]).astype(np.float32)
                if (_padding0 + k) % 2 == 0:
                    _c = C
                else:
                    _c = C[padding : C.shape[0] - padding]

                c_len = _c.shape[0]
                for fc in range(n_feat_d):
                    f_idx = f_start + fc
                    bias = biases[f_idx]

                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0.0
                    run_len = 0
                    nspv = 0

                    for j in range(c_len):
                        if _c[j] > bias:
                            ppv += 1
                            mean_index += j
                            mean += _c[j] + bias
                            run_len += 1
                        else:
                            if run_len > 1:
                                nspv += 1
                            run_len = 0
                            if _c[j] < bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                    if run_len > 1:
                        nspv += 1
                    stretch = c_len - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    features[c, f_idx] = ppv / c_len
                    features[c, f_idx + n_features] = mean / ppv if ppv > 0 else 0.0
                    features[c, f_idx + 2 * n_features] = (
                        mean_index / ppv if ppv > 0 else -1.0
                    )
                    features[c, f_idx + 3 * n_features] = max_stretch
                    features[c, f_idx + 4 * n_features] = nspv

                f_start = f_end

    return features


class CEEMDMultiRocketClassifier(BaseClassifier):
    """CEEMD with Improved MultiRocket (CEEMD-MultiRocket) classifier.

    CEEMD-MultiRocket [1]_ combines Complementary Ensemble Empirical Mode
    Decomposition (CEEMD) with an adapted MultiRocket transform in three steps:

    1. **Decomposition**: Each univariate series is decomposed by CEEMD into
       two intrinsic mode functions (``IMF1``, ``IMF2``) and a ``Residue``.
    2. **Sub-series selection**: The training data is split 1:1 with stratified
       sampling. Single-representation Improved MultiRocket ridge models are
       evaluated on the raw series (``acc_orig``) and on each decomposed
       sub-series (``acc_i``), retaining every sub-series with
       ``acc_i > selection_threshold * acc_orig`` (0 to 3 sub-series).
    3. **Feature extraction and classification**: Improved MultiRocket—using
       the 15 length-6 ``{-1, 2}`` kernels with two positive weights and five
       pooling operators (``PPV``, ``MPV``, ``MIPV``, ``LSPV``, and ``NSPV``,
       the count of contiguous positive runs of length at least 2)—is applied
       to the raw series, the retained sub-series, and the first-order
       difference of the raw series, followed by a ridge classifier.

    Parameters
    ----------
    n_features : int, default=50000
        Reference feature budget controlling the per-representation dilation
        count ``(n_features // 3) // 5`` (``222`` dilations per kernel at the
        default ``50,000``). Each active representation produces ``16,650``
        features, giving ``15 * 222 * 3 * 5 = 49,950`` total features in the
        modal single-IMF case described in Section 3.3 of [1]_.
    max_dilations_per_kernel : int, default=64
        Maximum number of dilations per kernel.
    noise_std : float, default=0.4
        Standard deviation ratio of the complementary white noise pairs added
        during CEEMD relative to each series' standard deviation.
    n_realizations : int, default=30
        Number of complementary positive/negative white-noise pairs used in
        CEEMD.
    n_siftings : int, default=10
        Number of cubic-spline envelope sifting iterations per IMF.
    selection_threshold : float, default=0.9
        Minimum validation accuracy ratio relative to the raw series
        (``acc_i > selection_threshold * acc_orig``) required to retain a
        decomposed sub-series.
    n_jobs : int, default=1
        The number of jobs to run in parallel for CEEMD and the Numba
        transforms. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling CEEMD noise realisations,
        the 1:1 sub-series selection split, and kernel bias sampling.

    Attributes
    ----------
    selected_subseries_ : list of int
        Indices in ``{0, 1, 2}`` (corresponding to ``IMF1``, ``IMF2``, and
        ``Residue``) retained during Step 2.
    classifier_ : Pipeline
        Fitted ``StandardScaler`` and ``_L2NormalisedRidgeClassifierCV``
        pipeline.

    Notes
    -----
    No reference code was published with [1]_, where CEEMD was run in MATLAB
    and Improved MultiRocket in Python. Convolution indexing, quantile bias
    sampling, and the standardised ridge pipeline follow the same authors'
    later ``KG-MTP`` implementation [2]_. ``NSPV`` counts maximal contiguous
    positive runs of length ``>= 2``, matching the worked example in Table 3 of
    [1]_.

    References
    ----------
    .. [1] Wang, Panjie, Jiang Wu, Yuan Wei, and Taiyong Li.
       "CEEMD-MultiRocket: Integrating CEEMD with Improved MultiRocket for Time
       Series Classification." Electronics 12, no. 5 (2023): 1188.
    .. [2] https://github.com/WangPanJie2024/KG-MTP

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     CEEMDMultiRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = CEEMDMultiRocketClassifier(
    ...     n_features=450, n_realizations=3, random_state=0
    ... )
    >>> clf.fit(X, y)
    CEEMDMultiRocketClassifier(n_features=450, n_realizations=3, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_features=50000,
        max_dilations_per_kernel=64,
        noise_std=0.4,
        n_realizations=30,
        n_siftings=10,
        selection_threshold=0.9,
        n_jobs=1,
        random_state=None,
    ):
        self.n_features = n_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.noise_std = noise_std
        self.n_realizations = n_realizations
        self.n_siftings = n_siftings
        self.selection_threshold = selection_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    @staticmethod
    def _preprocess(X):
        X_raw = np.nan_to_num(X[:, 0, :].astype(np.float64, copy=False), nan=0.0)
        if X_raw.shape[1] < 7:
            X_raw = np.pad(X_raw, ((0, 0), (0, 7 - X_raw.shape[1])))
        return X_raw

    def _decompose(self, X_raw, rng):
        noise_bank = rng.normal(
            0.0, 1.0, size=(self.n_realizations, X_raw.shape[1])
        )
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            imf1, imf2, residue = _ceemd_batch(
                X_raw, noise_bank, float(self.noise_std), int(self.n_siftings)
            )
        finally:
            set_num_threads(prev_threads)
        return [imf1, imf2, residue]

    def _fit_rep(self, X_rep, n_kernels_budget, seed):
        dilations, n_feat_per_dil = _fit_dilations_6_2(
            X_rep.shape[1], n_kernels_budget, self.max_dilations_per_kernel
        )
        quantiles = _quantiles(15 * int(np.sum(n_feat_per_dil)))
        biases = _fit_biases_6_2(
            X_rep, dilations, n_feat_per_dil, quantiles, _INDICES_6_2, seed
        )
        return dilations, n_feat_per_dil, biases

    def _transform_rep(self, X_rep, params):
        dilations, n_feat_per_dil, biases = params
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            Xt = _transform_6_2(
                X_rep, dilations, n_feat_per_dil, biases, _INDICES_6_2
            )
        finally:
            set_num_threads(prev_threads)
        return np.nan_to_num(Xt)

    def _score_single_rep(self, X_rep, tr_idx, val_idx, y_tr, y_val, budget, seed):
        params = self._fit_rep(X_rep[tr_idx], budget, seed)
        F_tr = self._transform_rep(X_rep[tr_idx], params)
        F_val = self._transform_rep(X_rep[val_idx], params)
        if len(np.unique(y_tr)) < 2:
            y_pred = np.repeat(y_tr[0], len(y_val))
        else:
            clf = make_pipeline(StandardScaler(), _L2NormalisedRidgeClassifierCV())
            clf.fit(F_tr, y_tr)
            y_pred = clf.predict(F_val)
        return float(np.mean(y_pred == y_val))

    def _fit(self, X, y):
        if self.n_features < 375:
            raise ValueError(f"n_features must be >= 375, got {self.n_features}.")
        if self.n_realizations < 1:
            raise ValueError(
                f"n_realizations must be >= 1, got {self.n_realizations}."
            )
        if self.n_siftings < 1:
            raise ValueError(f"n_siftings must be >= 1, got {self.n_siftings}.")

        self._n_jobs = _check_numba_jobs(check_n_jobs(self.n_jobs))
        rng = check_random_state(self.random_state)
        self._predict_noise_seed = int(rng.randint(np.iinfo(np.int32).max))

        X_raw = self._preprocess(X)
        _, y_ind = np.unique(y, return_inverse=True)

        if self.n_classes_ == 1 or len(y_ind) < 2:
            self.selected_subseries_ = []
            X_diff = np.diff(X_raw, 1)
            budget = max(15, (self.n_features // 3) // 5)
            s_raw, s_diff = rng.randint(np.iinfo(np.int32).max, size=2)
            self._rep_params = [
                self._fit_rep(X_raw, budget, int(s_raw)),
                self._fit_rep(X_diff, budget, int(s_diff)),
            ]
            self.classifier_ = None
            return self

        subseries = self._decompose(X_raw, rng)

        # Step 2: 1:1 stratified split to select sub-series
        split_seed = int(rng.randint(np.iinfo(np.int32).max))
        class_counts = np.bincount(y_ind)
        n_val = len(y_ind) - len(y_ind) // 2
        n_tr = len(y_ind) // 2
        if (
            int(np.min(class_counts)) >= 2
            and n_tr >= self.n_classes_
            and n_val >= self.n_classes_
        ):
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=split_seed
            )
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=0.5, random_state=split_seed
            )
        tr_idx, val_idx = next(splitter.split(X_raw, y_ind))
        y_tr, y_val = y_ind[tr_idx], y_ind[val_idx]

        sel_budget = max(15, (self.n_features // 3) // 5)
        sel_seeds = rng.randint(np.iinfo(np.int32).max, size=4)

        with _quiet_ridge_warnings():
            acc_orig = self._score_single_rep(
                X_raw, tr_idx, val_idx, y_tr, y_val, sel_budget, int(sel_seeds[0])
            )
            cutoff = self.selection_threshold * acc_orig
            self.selected_subseries_ = []
            for idx, sub in enumerate(subseries):
                acc_i = self._score_single_rep(
                    sub,
                    tr_idx,
                    val_idx,
                    y_tr,
                    y_val,
                    sel_budget,
                    int(sel_seeds[idx + 1]),
                )
                if acc_i > cutoff:
                    self.selected_subseries_.append(idx)

            # Step 3: Fit Improved MultiRocket on raw, selected IMFs*, and 1st diff
            X_diff = np.diff(X_raw, 1)
            active_reps = (
                [X_raw]
                + [subseries[i] for i in self.selected_subseries_]
                + [X_diff]
            )
            n_reps = len(active_reps)
            rep_seeds = rng.randint(np.iinfo(np.int32).max, size=n_reps)

            self._rep_params = [
                self._fit_rep(rep, sel_budget, int(s))
                for rep, s in zip(active_reps, rep_seeds)
            ]
            Xt = np.hstack(
                [
                    self._transform_rep(rep, params)
                    for rep, params in zip(active_reps, self._rep_params)
                ]
            )
            self.classifier_ = make_pipeline(
                StandardScaler(),
                _L2NormalisedRidgeClassifierCV(),
            ).fit(Xt, y_ind)

        return self

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        X_raw = self._preprocess(X)
        X_diff = np.diff(X_raw, 1)

        if len(self.selected_subseries_) > 0:
            pred_rng = np.random.RandomState(self._predict_noise_seed)
            subseries = self._decompose(X_raw, pred_rng)
            active_reps = (
                [X_raw]
                + [subseries[i] for i in self.selected_subseries_]
                + [X_diff]
            )
        else:
            active_reps = [X_raw, X_diff]

        Xt = np.hstack(
            [
                self._transform_rep(rep, params)
                for rep, params in zip(active_reps, self._rep_params)
            ]
        )
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
        return {"n_features": 450, "n_realizations": 2, "n_siftings": 5}
