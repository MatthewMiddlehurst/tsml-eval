"""Selected Feature ROCKET (SelF-Rocket) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SelFRocketClassifier"]

import random

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.utils.validation import check_n_jobs
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _check_numba_jobs,
    _quiet_ridge_warnings,
)

_CANDIDATE_NAMES = (
    "PPV",
    "ZC",
    "MPV",
    "MIPV",
    "LSPV",
    "PPV_DIFF",
    "ZC_DIFF",
    "MPV_DIFF",
    "MIPV_DIFF",
    "LSPV_DIFF",
    "PPV_MIX",
    "ZC_MIX",
    "MPV_MIX",
    "MIPV_MIX",
    "LSPV_MIX",
)


@njit(fastmath=True, parallel=True, cache=True)
def _self_rocket_transform(X, X1, parameters, parameters1, indices):
    """Compute PPV, LSPV, MPV, MIPV and ZC features on X and its first difference."""
    n_cases, input_length = X.shape
    _, _, dilations, n_features_per_dilation, biases = parameters
    _, _, dilations1, n_features_per_dilation1, biases1 = parameters1

    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_dilations1 = len(dilations1)

    n_features = n_kernels * np.sum(n_features_per_dilation)
    n_features1 = n_kernels * np.sum(n_features_per_dilation1)

    features = np.zeros((n_cases, 5 * (n_features + n_features1)), dtype=np.float32)
    diff_offset = 5 * n_features

    for example_index in prange(n_cases):
        _X = X[example_index]
        A = -_X
        G = _X + _X + _X

        feature_index_start = 0
        for dilation_index in range(n_dilations):
            _padding0 = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            n_features_this_dilation = n_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[4] = G

            start = dilation
            end = input_length - padding
            for gamma_index in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]
                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation
                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    _c = C
                else:
                    _c = C[padding:-padding]

                c_len = _c.shape[0]
                for feature_count in range(n_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]

                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0.0
                    zero_count = 0

                    for j in range(c_len):
                        if j < c_len - 1 and (
                            (_c[j] > _bias and _c[j + 1] < _bias)
                            or (_c[j] < _bias and _c[j + 1] > _bias)
                        ):
                            zero_count += 1
                        if _c[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += _c[j] + _bias
                        elif _c[j] < _bias:
                            stretch = j - last_val
                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = c_len - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    features[example_index, feature_index] = ppv / c_len
                    features[example_index, feature_index + n_features] = max_stretch
                    features[example_index, feature_index + 2 * n_features] = (
                        mean / ppv if ppv > 0 else 0.0
                    )
                    features[example_index, feature_index + 3 * n_features] = (
                        mean_index / ppv if ppv > 0 else -1.0
                    )
                    features[example_index, feature_index + 4 * n_features] = (
                        zero_count / c_len
                    )

                feature_index_start = feature_index_end

        # First-order difference
        _X1 = X1[example_index]
        A1 = -_X1
        G1 = _X1 + _X1 + _X1

        feature_index_start = 0
        for dilation_index in range(n_dilations1):
            _padding0 = dilation_index % 2
            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            n_features_this_dilation = n_features_per_dilation1[dilation_index]

            C_alpha = np.zeros(input_length - 1, dtype=np.float32)
            C_alpha[:] = A1
            C_gamma = np.zeros((9, input_length - 1), dtype=np.float32)
            C_gamma[4] = G1

            start = dilation
            # Matches the reference MultiRocket/SelF-Rocket offset using input_length
            end = input_length - padding
            for gamma_index in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A1[:end]
                C_gamma[gamma_index, -end:] = G1[:end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A1[start:]
                C_gamma[gamma_index, :-start] = G1[start:]
                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation
                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    _c = C
                else:
                    _c = C[padding:-padding]

                c_len = _c.shape[0]
                for feature_count in range(n_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _bias = biases1[feature_index]

                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0.0
                    zero_count = 0

                    for j in range(c_len):
                        if j < c_len - 1 and (
                            (_c[j] > _bias and _c[j + 1] < _bias)
                            or (_c[j] < _bias and _c[j + 1] > _bias)
                        ):
                            zero_count += 1
                        if _c[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += _c[j] + _bias
                        elif _c[j] < _bias:
                            stretch = j - last_val
                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = c_len - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    col = diff_offset + feature_index
                    features[example_index, col] = ppv / c_len
                    features[example_index, col + n_features1] = max_stretch
                    features[example_index, col + 2 * n_features1] = (
                        mean / ppv if ppv > 0 else 0.0
                    )
                    features[example_index, col + 3 * n_features1] = (
                        mean_index / ppv if ppv > 0 else -1.0
                    )
                    features[example_index, col + 4 * n_features1] = (
                        zero_count / c_len
                    )

                feature_index_start = feature_index_end

    return features


class SelFRocketClassifier(BaseClassifier):
    """Selected Feature ROCKET (SelF-Rocket) classifier.

    SelF-Rocket [1]_ extends MiniRocket across two input representations (the
    raw series ``I``, its first-order difference ``DIFF``, and their
    concatenation ``MIX``) and five pooling operators (``PPV``, zero-crossings
    ``ZC``, ``MPV``, ``MIPV``, and ``LSPV``), forming 15 candidate
    representation-pooling feature sets (or 5 when ``only_mix=True``).

    During training, repeated 2-fold stratified validation splits (or 250/250
    stratified subsamples when ``n_cases >= 500``) score each candidate using a
    ``RidgeClassifierCV`` fitted on a random subset of ``n_features_per_split``
    features. The candidate with the highest median validation accuracy is
    chosen, subject to a 90% top-``k`` vote consistency check; if consistency is
    below 90%, it falls back to ``"ZC_MIX"`` for long series (length ``> 512``)
    or ``"PPV_MIX"`` otherwise. A final ``RidgeClassifierCV`` is trained on all
    features of the selected candidate.

    This follows [1]_ and the authors' corrected reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of MiniRocket features per pooling operator and input
        representation (rounded down to a multiple of 84, minimum 84).
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per MiniRocket kernel.
    n_runs : int, default=10
        Number of 2-fold validation repeats (``2 * n_runs`` splits in total)
        used when scoring candidate representation-pooling pairs.
    n_features_per_split : int, default=5000
        Maximum number of features randomly subsampled per candidate in each
        validation split. Defaults to 5000 as in [2]_.
    only_mix : bool, default=False
        If True, restrict candidate selection to the five ``MIX`` candidates
        concatenating base and first-difference features, using top-2 vote
        validation as in [2]_.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling MiniRocket bias sampling,
        validation splits, and random feature subsampling.

    Attributes
    ----------
    selected_combination_ : str
        Name of the selected representation-pooling pair (for example,
        ``"PPV_MIX"`` or ``"ZC"``).
    selected_combination_index_ : int
        Index of the selected candidate within the active candidate list.
    classifier_ : RidgeClassifierCV
        Final ridge classifier fitted on the selected feature set.

    Notes
    -----
    Dilations and biases on the series and its first difference are fitted with
    aeon's ``MiniRocket``. Three bugs in the reference code [2]_ are fixed:

    1. ``np.squeeze(X)`` in ``transform_sr`` collapsed single-instance inputs to
       1D and crashed ``predict`` when ``n_cases == 1``; channel 0 is indexed
       explicitly instead, with no change when ``n_cases > 1``.
    2. ``random.sample`` raised ``ValueError`` whenever ``n_kernels`` produced
       fewer features than ``n_features_per_split``; the subsample size is
       capped at the candidate width, with no change at the default
       ``n_kernels=10000``.
    3. ``only_MIX=True`` mutated ``self.topvt = 2`` in place during ``fit`` and
       stratified 2-fold splitting failed if a class had only one training
       instance; ``top`` is computed locally and unstratified splits are used
       only when a class has fewer than two instances.

    References
    ----------
    .. [1] Lo, Mouhamadou Mansour, Gildas Morvan, Mathieu Rossi, Fabrice
       Morganti, and David Mercier. "Time series classification with random
       convolution kernels: pooling operators and input representations matter."
       Data Mining and Knowledge Discovery (2026).
    .. [2] https://github.com/ANR-MYEL/SelF-Rocket

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     SelFRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = SelFRocketClassifier(n_kernels=84, n_runs=2, random_state=0)
    >>> clf.fit(X, y)
    SelFRocketClassifier(n_kernels=84, n_runs=2, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=10000,
        max_dilations_per_kernel=32,
        n_runs=10,
        n_features_per_split=5000,
        only_mix=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_runs = n_runs
        self.n_features_per_split = n_features_per_split
        self.only_mix = only_mix
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    @staticmethod
    def _preprocess(X):
        X_2d = np.nan_to_num(X[:, 0, :].astype(np.float64, copy=False), nan=0.0)
        if X_2d.shape[1] < 10:
            X_2d = np.pad(X_2d, ((0, 0), (0, 10 - X_2d.shape[1])))
        return X_2d

    def _extract_candidates(self, X_2d, fit_scaler=False):
        X_diff = np.diff(X_2d, 1)

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            X_transform = _self_rocket_transform(
                X_2d,
                X_diff,
                self._parameters,
                self._parameters_diff,
                MiniRocket._indices,
            )
        finally:
            set_num_threads(prev_threads)

        if fit_scaler:
            self._scaler = StandardScaler().fit(X_transform)
        X_transform = self._scaler.transform(X_transform)

        b = self._n_features_per_pool
        ppv = X_transform[:, 0:b]
        lspv = X_transform[:, b : 2 * b]
        mpv = X_transform[:, 2 * b : 3 * b]
        mipv = X_transform[:, 3 * b : 4 * b]
        zc = X_transform[:, 4 * b : 5 * b]
        ppv_diff = X_transform[:, 5 * b : 6 * b]
        lspv_diff = X_transform[:, 6 * b : 7 * b]
        mpv_diff = X_transform[:, 7 * b : 8 * b]
        mipv_diff = X_transform[:, 8 * b : 9 * b]
        zc_diff = X_transform[:, 9 * b : 10 * b]

        candidates = [
            ppv,
            zc,
            mpv,
            mipv,
            lspv,
            ppv_diff,
            zc_diff,
            mpv_diff,
            mipv_diff,
            lspv_diff,
            np.concatenate((ppv, ppv_diff), axis=1),
            np.concatenate((zc, zc_diff), axis=1),
            np.concatenate((mpv, mpv_diff), axis=1),
            np.concatenate((mipv, mipv_diff), axis=1),
            np.concatenate((lspv, lspv_diff), axis=1),
        ]
        if self.only_mix:
            return candidates[10:]
        return candidates

    def _fit(self, X, y):
        if self.n_kernels < 84:
            raise ValueError(f"n_kernels must be >= 84, got {self.n_kernels}.")
        if self.n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {self.n_runs}.")
        if self.n_features_per_split < 1:
            raise ValueError(
                f"n_features_per_split must be >= 1, got {self.n_features_per_split}."
            )

        self._n_jobs = _check_numba_jobs(check_n_jobs(self.n_jobs))
        raw_len = X.shape[2]
        X_2d = self._preprocess(X)
        X_diff = np.diff(X_2d, 1)

        rng = check_random_state(self.random_state)
        seed_base, seed_diff, seed_split, seed_sample = rng.randint(
            np.iinfo(np.int32).max, size=4
        )

        self._parameters = (
            MiniRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_jobs=self._n_jobs,
                random_state=int(seed_base),
            )
            .fit(X_2d[:, None, :])
            .parameters
        )
        self._parameters_diff = (
            MiniRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_jobs=self._n_jobs,
                random_state=int(seed_diff),
            )
            .fit(X_diff[:, None, :])
            .parameters
        )
        self._n_features_per_pool = 84 * int(np.sum(self._parameters[3]))

        _, y_ind = np.unique(y, return_inverse=True)
        if self.n_classes_ == 1:
            self._extract_candidates(X_2d, fit_scaler=True)
            names = _CANDIDATE_NAMES[10:] if self.only_mix else _CANDIDATE_NAMES
            self.selected_combination_index_ = names.index("PPV_MIX")
            self.selected_combination_ = "PPV_MIX"
            self.classifier_ = None
            return self

        pooling_op = self._extract_candidates(X_2d, fit_scaler=True)
        pooling_names = (
            list(_CANDIDATE_NAMES[10:]) if self.only_mix else list(_CANDIDATE_NAMES)
        )
        n_comb = len(pooling_names)

        class_counts = np.bincount(y_ind)
        can_stratify = int(np.min(class_counts)) >= 2
        if len(y_ind) >= 500:
            if can_stratify:
                splitter = StratifiedShuffleSplit(
                    n_splits=2 * self.n_runs,
                    train_size=250,
                    test_size=250,
                    random_state=int(seed_split),
                )
            else:
                splitter = ShuffleSplit(
                    n_splits=2 * self.n_runs,
                    train_size=250,
                    test_size=250,
                    random_state=int(seed_split),
                )
        else:
            if can_stratify:
                splitter = RepeatedStratifiedKFold(
                    n_splits=2,
                    n_repeats=self.n_runs,
                    random_state=int(seed_split),
                )
            else:
                splitter = RepeatedKFold(
                    n_splits=2,
                    n_repeats=self.n_runs,
                    random_state=int(seed_split),
                )

        py_rng = random.Random(int(seed_sample))
        n_votes = 2 * self.n_runs
        scores = np.zeros((n_votes, n_comb), dtype=np.float32)
        alphas = np.logspace(-3, 3, 10)

        with _quiet_ridge_warnings():
            for split_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_2d, y_ind)):
                y_tr = y_ind[tr_idx]
                y_val = y_ind[val_idx]
                for k in range(n_comb):
                    cand = pooling_op[k]
                    n_sub = min(self.n_features_per_split, cand.shape[1])
                    feat_idx = py_rng.sample(range(cand.shape[1]), n_sub)
                    if len(np.unique(y_tr)) < 2:
                        y_pred = np.repeat(y_tr[0], len(y_val))
                    else:
                        clf = RidgeClassifierCV(alphas=alphas)
                        clf.fit(cand[tr_idx][:, feat_idx], y_tr)
                        y_pred = clf.predict(cand[val_idx][:, feat_idx])
                    scores[split_idx, k] = np.mean(y_pred == y_val)

            medians = np.median(scores, axis=0)
            # First candidate achieving the maximum median wins ties, matching max()
            best_idx = int(np.argmax(medians))

            top = 2 if self.only_mix else 5
            in_top = 0
            for split_idx in range(n_votes):
                row_sorted = sorted(scores[split_idx].tolist(), reverse=True)
                if scores[split_idx, best_idx] >= row_sorted[top - 1]:
                    in_top += 1

            if (in_top / n_votes) * 100.0 >= 90.0:
                idx_fin = best_idx
            elif raw_len > 512:
                idx_fin = pooling_names.index("ZC_MIX")
            else:
                idx_fin = pooling_names.index("PPV_MIX")

            self.selected_combination_index_ = idx_fin
            self.selected_combination_ = pooling_names[idx_fin]
            self.classifier_ = RidgeClassifierCV(alphas=alphas).fit(
                pooling_op[idx_fin], y_ind
            )

        return self

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        X_2d = self._preprocess(X)
        pooling_op = self._extract_candidates(X_2d, fit_scaler=False)
        preds_ind = self.classifier_.predict(
            pooling_op[self.selected_combination_index_]
        )
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
        return {"n_kernels": 84, "n_runs": 2, "n_features_per_split": 42}
