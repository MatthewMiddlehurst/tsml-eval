"""Selected Prototype ROCKET (SPROCKET) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SPROCKETClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.distances import (
    adtw_distance,
    dtw_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    manhattan_distance,
    msm_distance,
    squared_distance,
    twe_distance,
    wdtw_distance,
)
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._rocket_utils import (
    _check_numba_jobs,
    _quiet_ridge_warnings,
)

_MSM = 0
_EUCLIDEAN = 1
_ADTW = 2
_DTW = 3
_WDTW = 4
_ERP = 5
_LCSS = 6
_TWE = 7
_SQUARED = 8
_MANHATTAN = 9
_EMD_UNSORT = 10
_EMD_SORT = 11
_COSINE = 12
_CHEBYSHEV = 13
_CORRELATION = 14

_DIST_MAP = {
    "msm": _MSM,
    "euclidean": _EUCLIDEAN,
    "adtw": _ADTW,
    "dtw": _DTW,
    "wdtw": _WDTW,
    "erp": _ERP,
    "lcss": _LCSS,
    "twe": _TWE,
    "squared": _SQUARED,
    "manhattan": _MANHATTAN,
    "emd_unsort": _EMD_UNSORT,
    "emd_sort": _EMD_SORT,
    "cosine": _COSINE,
    "chebyshev": _CHEBYSHEV,
    "correlation": _CORRELATION,
}


@njit(fastmath=True, cache=True)
def _compute_signed_emd(series_1, series_2):
    n_channels, n_timepoints = series_1.shape
    total_emd = 0.0
    for i in range(n_channels):
        s1 = series_1[i]
        s2 = series_2[i]
        leftover_pos = 0.0
        total_pos = 0.0
        leftover_neg = 0.0
        total_neg = 0.0
        for j in range(n_timepoints):
            v1 = s1[j]
            v2 = s2[j]
            s1_pos = v1 if v1 > 0.0 else 0.0
            s2_pos = v2 if v2 > 0.0 else 0.0
            s1_neg = -v1 if -v1 > 0.0 else 0.0
            s2_neg = -v2 if -v2 > 0.0 else 0.0

            leftover_pos = s1_pos + leftover_pos - s2_pos
            total_pos += abs(leftover_pos)
            leftover_neg = s1_neg + leftover_neg - s2_neg
            total_neg += abs(leftover_neg)
        total_emd += total_pos + total_neg + 0.5 * (leftover_pos + leftover_neg)
    return total_emd


@njit(fastmath=True, cache=True)
def _sorted_emd_wrapper(series_1, series_2):
    n_channels = series_1.shape[0]
    sorted_s1 = np.empty_like(series_1)
    sorted_s2 = np.empty_like(series_2)
    for i in range(n_channels):
        sorted_s1[i] = np.sort(series_1[i])
        sorted_s2[i] = np.sort(series_2[i])
    return _compute_signed_emd(sorted_s1, sorted_s2)


@njit(fastmath=True, cache=True)
def _cosine_similarity_wrapper(series_1, series_2):
    n_channels, n_timepoints = series_1.shape
    total = 0.0
    for c in range(n_channels):
        u = series_1[c]
        v = series_2[c]
        uv = 0.0
        uu = 0.0
        vv = 0.0
        for t in range(n_timepoints):
            uv += u[t] * v[t]
            uu += u[t] * u[t]
            vv += v[t] * v[t]
        cos_theta = 1.0
        if uu != 0.0 and vv != 0.0:
            cos_theta = uv / np.sqrt(uu * vv)
        total += cos_theta
    return total


@njit(fastmath=True, cache=True)
def _chebyshev_wrapper(series_1, series_2):
    n_channels = series_1.shape[0]
    total = 0.0
    for c in range(n_channels):
        total += np.max(np.abs(series_1[c] - series_2[c]))
    return total


@njit(fastmath=True, cache=True)
def _correlation_wrapper(series_1, series_2):
    n_channels = series_1.shape[0]
    total = 0.0
    for c in range(n_channels):
        x = series_1[c]
        y = series_2[c]
        shifted_x = x - np.mean(x)
        shifted_y = y - np.mean(y)
        norm_x = np.sqrt(np.dot(shifted_x, shifted_x))
        norm_y = np.sqrt(np.dot(shifted_y, shifted_y))
        if norm_x == 0.0 or norm_y == 0.0:
            dist = 0.0
        else:
            dist = 1.0 - np.dot(shifted_x, shifted_y) / (norm_x * norm_y)
        total += dist
    return total


@njit(fastmath=True, cache=True)
def _call_distance(dist_id, x, y, window_frac):
    if dist_id == _EUCLIDEAN:
        return euclidean_distance(x, y)
    elif dist_id == _MSM:
        return msm_distance(x, y, window=window_frac)
    elif dist_id == _MANHATTAN:
        return manhattan_distance(x, y)
    elif dist_id == _SQUARED:
        return squared_distance(x, y)
    elif dist_id == _EMD_UNSORT:
        return _compute_signed_emd(x, y)
    elif dist_id == _EMD_SORT:
        return _sorted_emd_wrapper(x, y)
    elif dist_id == _COSINE:
        return _cosine_similarity_wrapper(x, y)
    elif dist_id == _CHEBYSHEV:
        return _chebyshev_wrapper(x, y)
    elif dist_id == _CORRELATION:
        return _correlation_wrapper(x, y)
    elif dist_id == _ADTW:
        return adtw_distance(x, y, window=window_frac)
    elif dist_id == _DTW:
        return dtw_distance(x, y, window=window_frac)
    elif dist_id == _WDTW:
        return wdtw_distance(x, y, window=window_frac)
    elif dist_id == _ERP:
        return erp_distance(x, y, window=window_frac)
    elif dist_id == _LCSS:
        return lcss_distance(x, y, window=window_frac)
    elif dist_id == _TWE:
        return twe_distance(x, y, window=window_frac)
    return euclidean_distance(x, y)


@njit(fastmath=True, cache=True)
def _stratified_random_indices(labels, k, seed):
    if seed >= 0:
        np.random.seed(seed)

    n = labels.shape[0]
    unique_labels = np.unique(labels)
    n_classes = unique_labels.shape[0]

    remapped = np.empty(n, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        lbl = labels[i]
        idx = 0
        for j in range(n_classes):
            if unique_labels[j] == lbl:
                idx = j
                break
        remapped[i] = idx
        counts[idx] += 1

    if k < n_classes:
        cls_arr = np.arange(n_classes)
        np.random.shuffle(cls_arr)
        chosen_classes = cls_arr[:k]
        n_sel = k
    else:
        chosen_classes = np.arange(n_classes)
        n_sel = n_classes

    total_counts = 0
    for i in range(n_sel):
        total_counts += counts[chosen_classes[i]]

    per_class = np.zeros(n_sel, dtype=np.int64)
    for i in range(n_sel):
        c = chosen_classes[i]
        per_class[i] = (counts[c] * k) // total_counts
        if k >= n_sel and per_class[i] == 0:
            per_class[i] = 1
        if per_class[i] > counts[c]:
            per_class[i] = counts[c]

    total_alloc = 0
    for i in range(n_sel):
        total_alloc += per_class[i]
    diff = k - total_alloc

    tmp_order = np.arange(n_sel)
    np.random.shuffle(tmp_order)
    if diff > 0:
        step = 0
        while diff > 0 and step < n * n_sel:
            j = tmp_order[step % n_sel]
            if per_class[j] < counts[chosen_classes[j]]:
                per_class[j] += 1
                diff -= 1
            step += 1
    elif diff < 0:
        diff = -diff
        step = 0
        while diff > 0 and step < n * n_sel:
            j = tmp_order[step % n_sel]
            if per_class[j] > 1:
                per_class[j] -= 1
                diff -= 1
            step += 1

    result = np.empty(k, dtype=np.int64)
    pos = 0
    for i in range(n_sel):
        cls = chosen_classes[i]
        n_pick = per_class[i]
        if n_pick == 0:
            continue
        cnt = counts[cls]
        cls_indices = np.empty(cnt, dtype=np.int64)
        t = 0
        for j in range(n):
            if remapped[j] == cls:
                cls_indices[t] = j
                t += 1
        np.random.shuffle(cls_indices)
        for j in range(n_pick):
            result[pos] = cls_indices[j]
            pos += 1

    np.random.shuffle(result)
    return result


@njit(fastmath=True, cache=True)
def _convolve_kernel(
    x, weights, w_start, length, bias, dilation, n_ch, ch_indices, c_start, out_buf
):
    """Apply a dilated kernel centered by ``((length - 1) * dilation) // 2``."""
    n_timepoints = x.shape[1]
    midpoint = ((length - 1) * dilation) // 2
    for k_ch in range(n_ch):
        x_ch = x[ch_indices[c_start + k_ch]]
        w_off = w_start + k_ch * length
        for t in range(n_timepoints):
            idx = t - midpoint
            conv_sum = bias
            for j in range(length):
                if -1 < idx < n_timepoints:
                    conv_sum += weights[w_off + j] * x_ch[idx]
                idx += dilation
            out_buf[k_ch, t] = conv_sum


@njit(fastmath=True, parallel=True, cache=True)
def _compute_prototypes(X, cand_indices, kernels):
    weights, lengths, biases, dilations, _, n_ch_indices, ch_indices = kernels
    n_kernels = lengths.shape[0]
    n_protos = cand_indices.shape[1]
    n_channels = X.shape[1]
    n_timepoints = X.shape[2]

    w_starts = np.zeros(n_kernels, dtype=np.int32)
    c_starts = np.zeros(n_kernels, dtype=np.int32)
    w_pos = 0
    c_pos = 0
    for i in range(n_kernels):
        w_starts[i] = w_pos
        w_pos += n_ch_indices[i] * lengths[i]
        c_starts[i] = c_pos
        c_pos += n_ch_indices[i]

    protos = np.zeros((n_kernels, n_protos, n_channels, n_timepoints), dtype=np.float32)
    for i in prange(n_kernels):
        n_ch = n_ch_indices[i]
        for p in range(n_protos):
            x = X[cand_indices[i, p]]
            _convolve_kernel(
                x,
                weights,
                w_starts[i],
                lengths[i],
                biases[i],
                dilations[i],
                n_ch,
                ch_indices,
                c_starts[i],
                protos[i, p, :n_ch],
            )
    return protos


@njit(fastmath=True, parallel=True, cache=True)
def _apply_sprocket_kernels(X, kernels, protos, dist_id, window_frac):
    weights, lengths, biases, dilations, _, n_ch_indices, ch_indices = kernels
    n_cases, _, n_timepoints = X.shape
    n_kernels = lengths.shape[0]
    n_protos = protos.shape[1]

    w_starts = np.zeros(n_kernels, dtype=np.int32)
    c_starts = np.zeros(n_kernels, dtype=np.int32)
    w_pos = 0
    c_pos = 0
    for i in range(n_kernels):
        w_starts[i] = w_pos
        w_pos += n_ch_indices[i] * lengths[i]
        c_starts[i] = c_pos
        c_pos += n_ch_indices[i]

    features = np.zeros((n_cases, n_kernels * n_protos), dtype=np.float32)
    for i in prange(n_kernels):
        n_ch = n_ch_indices[i]
        f_col = i * n_protos
        for c in range(n_cases):
            conv = np.zeros((n_ch, n_timepoints), dtype=np.float32)
            _convolve_kernel(
                X[c],
                weights,
                w_starts[i],
                lengths[i],
                biases[i],
                dilations[i],
                n_ch,
                ch_indices,
                c_starts[i],
                conv,
            )
            for p in range(n_protos):
                features[c, f_col + p] = _call_distance(
                    dist_id, conv, protos[i, p, :n_ch], window_frac
                )
    return features


class SPROCKETClassifier(BaseClassifier):
    """Selected Prototype ROCKET (SPROCKET) classifier.

    SPROCKET [1]_ replaces ROCKET's summary pooling operators (PPV and Max)
    with prototype distances in activation-map space. During training,
    ``n_kernels`` random dilated convolutional kernels are generated with
    aeon's ``Rocket`` and ``M = 1 + floor(log(n_cases) / log(proto_per_kernel))``
    training instances are sampled per kernel and convolved to serve as
    activation prototypes. Each series is transformed into its ``n_kernels * M``
    distances to those prototypes across the active kernel channels, and a
    ``RidgeClassifierCV`` is fitted on the resulting distance matrix.

    This follows [1]_ and the reference implementation [2]_, with bugs in the
    reference code fixed as detailed in the Notes.

    Parameters
    ----------
    n_kernels : int, default=512
        Number of random convolutional kernels.
    proto_per_kernel : float, default=4.0
        Logarithm base (> 1) controlling the number of sampled prototypes per
        kernel, ``min(n_cases, 1 + int(log(n_cases) / log(proto_per_kernel)))``.
    distance : str or int, default="euclidean"
        Distance measure between convolved series and prototypes. Supported
        names are ``"euclidean"``, ``"msm"``, ``"dtw"``, ``"adtw"``, ``"wdtw"``,
        ``"erp"``, ``"lcss"``, ``"twe"``, ``"squared"``, ``"manhattan"``,
        ``"emd_unsort"``, ``"emd_sort"``, ``"cosine"``, ``"chebyshev"``, and
        ``"correlation"``.
    window_frac : float, default=0.2
        Sakoe-Chiba bounding window fraction passed to elastic distance
        measures.
    normalise : bool, default=False
        Whether to z-normalise each input series per channel using aeon's
        ``Normalizer`` before convolution.
    stratified_sampling : bool, default=False
        Whether to sample training prototype indices per kernel with class
        stratification instead of uniform sampling without replacement.
    n_jobs : int, default=1
        The number of jobs to run in parallel for convolution and distance
        computation. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling kernel generation and
        prototype selection.

    Attributes
    ----------
    n_prototypes_per_kernel_ : int
        Number of prototypes sampled per kernel during ``fit``.
    classifier_ : RidgeClassifierCV
        Fitted ridge classifier.

    Notes
    -----
    Random kernels are generated with aeon's ``Rocket(normalise=False)`` (whose
    kernel tuple matches ``_generate_kernels`` in [2]_ exactly). Four bugs in
    the reference implementation [2]_ are fixed:

    1. ``univariate_apply_kernel_return_convolved`` and
       ``multivariate_apply_kernel_return_convolved`` in [2]_ use
       ``midpoint = length // 2`` regardless of ``dilation``, causing dilated
       kernels to run off the right edge of the series. Centering by
       ``((length - 1) * dilation) // 2`` changes predictions on most datasets
       (5-seed mean accuracy on ``Beef`` ``0.7200 -> 0.8000``, ``ACSF1``
       ``0.5860 -> 0.6040``, ``OSULeaf`` ``0.6124 -> 0.6231``,
       ``EqualMinimalJapaneseVowels`` ``0.9200 -> 0.9400``, ``GunPoint``
       ``0.9800 -> 0.9747``).
    2. On multivariate inputs, kernels with ``n_channel_indices[i] == 1`` in
       [2]_ hardcode ``x[0]`` instead of ``x[channel_indices[...]]`` and
       broadcast the 1-channel prototype across all ``n_channels`` slots. The
       sampled channel is indexed directly instead (leaving univariate outputs
       unchanged and altering 2 of 5 seeds on ``EqualMinimalJapaneseVowels``).
    3. The dynamic fallback dispatcher in [2]_ raises ``NameError`` /
       ``TypeError`` and drops ``call_distance`` return values for all elastic
       distances (``"msm"``, ``"dtw"``, ``"adtw"``, ``"wdtw"``, ``"erp"``,
       ``"lcss"``, ``"twe"``) and omits ``"squared"``, and passes an extra
       ``window`` argument to ``chebyshev`` and ``correlation``. All 15
       distances in ``DIST_MAP`` are wired cleanly in Numba.
    4. ``normalise=True`` in [2]_ only normalises in ``_transform`` while
       extracting prototypes from unnormalised ``X`` in ``_fit``; prototype
       sampling uses global unseeded ``np.random`` without capping at
       ``len(X)``; and ``__init__`` mutates ``dist_id``. ``Normalizer`` is now
       applied in both ``fit`` and ``predict``, prototype sampling is seeded
       from ``random_state`` and capped at ``len(X)``, and ``distance`` is left
       unmodified on ``self``.

    References
    ----------
    .. [1] Harner, Nicholas. "SPROCKET: Extending ROCKET to Distance-Based
       Time-Series Transformations With Prototypes." arXiv preprint
       arXiv:2512.08246 (2025).
    .. [2] https://github.com/username76543/sprocket_public

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     SPROCKETClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=2, random_state=0)
    >>> clf = SPROCKETClassifier(n_kernels=32, random_state=0)
    >>> clf.fit(X, y)
    SPROCKETClassifier(n_kernels=32, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=512,
        proto_per_kernel=4.0,
        distance="euclidean",
        window_frac=0.2,
        normalise=False,
        stratified_sampling=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.proto_per_kernel = proto_per_kernel
        self.distance = distance
        self.window_frac = window_frac
        self.normalise = normalise
        self.stratified_sampling = stratified_sampling
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _resolve_dist_id(self):
        if isinstance(self.distance, (int, np.integer)):
            if 0 <= int(self.distance) <= 14:
                return int(self.distance)
            raise ValueError(f"Unknown integer distance id: {self.distance!r}.")
        if isinstance(self.distance, str):
            key = self.distance.lower()
            if key in _DIST_MAP:
                return _DIST_MAP[key]
        raise ValueError(
            f"Unknown distance={self.distance!r}. Expected one of {sorted(_DIST_MAP)}."
        )

    def _preprocess(self, X, fit=False):
        X_clean = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        if self.normalise:
            if fit:
                self._normalizer = Normalizer().fit(X_clean)
            X_clean = self._normalizer.transform(X_clean)
        return X_clean

    def _fit(self, X, y):
        if self.n_kernels < 1:
            raise ValueError(f"n_kernels must be >= 1, got {self.n_kernels}.")
        if self.proto_per_kernel <= 1.0:
            raise ValueError(
                f"proto_per_kernel must be > 1.0, got {self.proto_per_kernel}."
            )
        self._dist_id = self._resolve_dist_id()
        self._n_jobs = _check_numba_jobs(check_n_jobs(self.n_jobs))

        X_proc = self._preprocess(X, fit=True)
        n_cases = X_proc.shape[0]
        raw_protos = 1 + int(np.log(max(1, n_cases)) / np.log(self.proto_per_kernel))
        self.n_prototypes_per_kernel_ = min(n_cases, max(1, raw_protos))

        rng = check_random_state(self.random_state)
        kernel_seed = int(rng.randint(np.iinfo(np.int32).max))

        self._kernels = (
            Rocket(
                n_kernels=self.n_kernels,
                normalise=False,
                n_jobs=self._n_jobs,
                random_state=kernel_seed,
            )
            .fit(X_proc)
            .kernels
        )

        _, y_ind = np.unique(y, return_inverse=True)
        y_ind_i64 = y_ind.astype(np.int64)

        cand_indices = np.empty(
            (self.n_kernels, self.n_prototypes_per_kernel_), dtype=np.int64
        )
        if self.stratified_sampling:
            proto_seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_kernels)
            for i in range(self.n_kernels):
                cand_indices[i] = _stratified_random_indices(
                    y_ind_i64, self.n_prototypes_per_kernel_, int(proto_seeds[i])
                )
        else:
            all_idx = np.arange(n_cases, dtype=np.int64)
            for i in range(self.n_kernels):
                cand_indices[i] = rng.choice(
                    all_idx, size=self.n_prototypes_per_kernel_, replace=False
                )

        X_f32 = X_proc.astype(np.float32)
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            self._prototypes = _compute_prototypes(X_f32, cand_indices, self._kernels)
            if self.n_classes_ == 1:
                self.classifier_ = None
                return self
            Xt = _apply_sprocket_kernels(
                X_f32,
                self._kernels,
                self._prototypes,
                self._dist_id,
                float(self.window_frac),
            )
        finally:
            set_num_threads(prev_threads)

        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        with _quiet_ridge_warnings():
            self.classifier_ = RidgeClassifierCV().fit(Xt, y_ind)
        return self

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        X_f32 = self._preprocess(X, fit=False).astype(np.float32)
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            Xt = _apply_sprocket_kernels(
                X_f32,
                self._kernels,
                self._prototypes,
                self._dist_id,
                float(self.window_frac),
            )
        finally:
            set_num_threads(prev_threads)

        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
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
        return {"n_kernels": 32}
