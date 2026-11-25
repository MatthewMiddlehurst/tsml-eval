"""Class-wise MiniRocket classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ClassWiseMiniRocketClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.distances import distance as aeon_distance
from aeon.transformations.collection.convolution_based import MiniRocket
from scipy.special import expit
from scipy.stats import median_abs_deviation
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._rocket_utils import (
    _check_numba_jobs,
    _quiet_ridge_warnings,
)


class ClassWiseMiniRocketClassifier(BaseClassifier):
    """Class-wise MiniRocket classifier with local channel selection.

    Class-wise MiniRocket [1]_ decomposes ``C``-class multivariate time series
    classification into ``C`` independent one-vs-rest sub-problems. For each
    class ``K``, a class-specific channel subset ``S_K`` is identified using
    local prototype-distance elbow detection (adapting ECP or ECS [2]_ to keep
    channels discriminative for class ``K`` rather than taking a global union),
    and a dedicated ``MiniRocket`` transform, ``StandardScaler``, and
    ``RidgeClassifierCV`` are trained on ``X[:, S_K, :]`` with binary labels
    ``y == K``. At test time, each binary classifier's decision score is mapped
    through the logistic sigmoid and the class with highest score is predicted.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of MiniRocket features generated for each class sub-problem.
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per kernel passed to ``MiniRocket``.
    channel_selection : {"ecp", "ecs", None}, default="ecp"
        Per-class channel selection strategy: ``"ecp"`` uses local Elbow Class
        Pairwise selection (``CW+L`` with ECP, the highest-ranked local
        channel-selection variant in Figure 4 of [1]_), ``"ecs"`` uses local
        Elbow Class Sum selection (``CW+L`` with ECS), and None retains all
        channels for every class (``CW`` in [1]_).
    prototype_type : {"mad", "mean", "median", None}, default=None
        Class prototype representation used for channel selection. If None,
        uses the publication defaults from [1]_ and [2]_: ``"mad"`` when
        ``channel_selection="ecp"`` and ``"mean"`` when
        ``channel_selection="ecs"``.
    distance : {"euclidean", "dtw"}, default="euclidean"
        Distance measure between per-channel class prototypes.
    n_jobs : int, default=1
        The number of jobs to run in parallel for ``MiniRocket``. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to seed each class's
        ``MiniRocket`` transform.

    Attributes
    ----------
    channels_selected_ : list of np.ndarray
        Selected channel index array ``S_K`` for each class ``K``.
    classifiers_ : list of RidgeClassifierCV
        Fitted one-vs-rest ridge classifiers, one per class.

    References
    ----------
    .. [1] Lo, Mouhamadou Mansour, Gildas Morvan, Mathieu Rossi, Fabrice
       Morganti, and David Mercier. "Improving Multivariate Time Series
       Classification with Class-Wise Training and Model Aggregation." arXiv
       preprint arXiv:2609.07493 (2026).
    .. [2] Dhariyal, Bhaskar, Thach Le Nguyen, and Georgiana Ifrim. "Scalable
       classifier-agnostic channel selection for multivariate time series
       classification." Data Mining and Knowledge Discovery 37, no. 2 (2023):
       1010-1054.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     ClassWiseMiniRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=20, n_channels=4, n_labels=3, random_state=0
    ... )
    >>> clf = ClassWiseMiniRocketClassifier(n_kernels=84, random_state=0)
    >>> clf.fit(X, y)
    ClassWiseMiniRocketClassifier(n_kernels=84, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=10000,
        max_dilations_per_kernel=32,
        channel_selection="ecp",
        prototype_type=None,
        distance="euclidean",
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.channel_selection = channel_selection
        self.prototype_type = prototype_type
        self.distance = distance
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_kernels < 1:
            raise ValueError(f"n_kernels must be at least 1, got {self.n_kernels}.")
        if self.max_dilations_per_kernel < 1:
            raise ValueError(
                "max_dilations_per_kernel must be at least 1, "
                f"got {self.max_dilations_per_kernel}."
            )
        method = (
            None if self.channel_selection is None else self.channel_selection.lower()
        )
        if method not in (None, "ecp", "ecs"):
            raise ValueError(
                f"Unknown channel_selection={self.channel_selection!r}. "
                "Choose from 'ecp', 'ecs', or None."
            )
        proto_mode = (
            ("mad" if method == "ecp" else "mean")
            if self.prototype_type is None
            else self.prototype_type.lower()
        )
        if proto_mode not in ("mad", "mean", "median"):
            raise ValueError(
                f"Unknown prototype_type={self.prototype_type!r}. "
                "Choose from 'mad', 'mean', 'median', or None."
            )
        dist_mode = self.distance.lower()
        if dist_mode not in ("euclidean", "dtw"):
            raise ValueError(
                f"Unknown distance={self.distance!r}. "
                "Choose from 'euclidean' or 'dtw'."
            )

        self._n_jobs = _check_numba_jobs(self.n_jobs)
        n_channels = X.shape[1]

        if self.n_classes_ == 1:
            self.channels_selected_ = [np.arange(n_channels, dtype=int)]
            self.classifiers_ = []
            self._transformers = []
            self._scalers = []
            return self

        rng = check_random_state(self.random_state)
        class_seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_classes_)

        X_clean = self._preprocess_series(X)
        y_ind = np.array([self._class_dictionary[label] for label in y], dtype=int)

        self.channels_selected_ = _select_classwise_channels(
            X_clean, y_ind, self.n_classes_, method, proto_mode, dist_mode
        )

        self._transformers = []
        self._scalers = []
        self.classifiers_ = []
        alphas = np.logspace(-3, 3, 10)

        for k in range(self.n_classes_):
            ch_idx = self.channels_selected_[k]
            trf = MiniRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_jobs=self._n_jobs,
                random_state=int(class_seeds[k]),
            )
            feat_k = np.asarray(
                trf.fit_transform(X_clean[:, ch_idx, :]), dtype=np.float64
            )
            scaler = StandardScaler()
            feat_k_scaled = scaler.fit_transform(feat_k)

            y_bin = (y_ind == k).astype(int)
            clf = RidgeClassifierCV(alphas=alphas)
            with _quiet_ridge_warnings():
                clf.fit(feat_k_scaled, y_bin)

            self._transformers.append(trf)
            self._scalers.append(scaler)
            self.classifiers_.append(clf)

        return self

    @staticmethod
    def _preprocess_series(X):
        X_clean = np.nan_to_num(X, nan=0.0)
        if X_clean.shape[2] < 9:
            X_clean = np.pad(X_clean, ((0, 0), (0, 0), (0, 9 - X_clean.shape[2])))
        return X_clean

    def _class_scores(self, X):
        X_clean = self._preprocess_series(X)
        scores = np.zeros((len(X_clean), self.n_classes_), dtype=np.float64)
        for k in range(self.n_classes_):
            ch_idx = self.channels_selected_[k]
            feat_k = np.asarray(
                self._transformers[k].transform(X_clean[:, ch_idx, :]),
                dtype=np.float64,
            )
            feat_k_scaled = self._scalers[k].transform(feat_k)
            decision = np.asarray(
                self.classifiers_[k].decision_function(feat_k_scaled), dtype=np.float64
            ).ravel()
            scores[:, k] = expit(decision)
        return scores

    def _predict_proba(self, X):
        if self.n_classes_ == 1:
            return np.ones((len(X), 1), dtype=float)

        scores = self._class_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        zero_rows = (totals == 0).ravel()
        if np.any(zero_rows):
            scores[zero_rows] = 1.0 / self.n_classes_
            totals[zero_rows] = 1.0
        return scores / totals

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        scores = self._class_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

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
        return {"n_kernels": 84, "max_dilations_per_kernel": 6}


def _detect_knee_channels(sorted_values, sorted_indices):
    """Select channels strictly above the maximum-perpendicular-distance knee."""
    n_points = len(sorted_values)
    if n_points <= 1:
        return [int(i) for i in sorted_indices]

    coords = np.vstack((np.arange(n_points, dtype=float), sorted_values)).T
    first = coords[0]
    line_vec = coords[-1] - first
    line_norm = float(np.sqrt(np.sum(line_vec**2)))
    if line_norm == 0.0:
        return [int(sorted_indices[0])]

    line_unit = line_vec / line_norm
    vec_from_first = coords - first
    proj = np.sum(vec_from_first * line_unit, axis=1)
    perp = vec_from_first - np.outer(proj, line_unit)
    dist_to_line = np.sqrt(np.sum(perp**2, axis=1))

    knee_idx = int(np.argmax(dist_to_line))
    knee_val = sorted_values[knee_idx]
    selected = [
        int(idx) for val, idx in zip(sorted_values, sorted_indices) if val > knee_val
    ]
    if len(selected) == 0:
        return [int(sorted_indices[knee_idx])]
    return selected


def _select_classwise_channels(X, y_ind, n_classes, method, proto_mode, dist_mode):
    """Compute local class-specific channel subsets ``S_K`` via ECS or ECP."""
    n_channels, n_timepoints = X.shape[1], X.shape[2]
    if method is None or n_channels == 1:
        return [np.arange(n_channels, dtype=int) for _ in range(n_classes)]

    prototypes = np.zeros((n_classes, n_channels, n_timepoints), dtype=np.float64)
    for c in range(n_classes):
        Xc = X[y_ind == c]
        if proto_mode == "mean":
            prototypes[c] = Xc.mean(axis=0)
        elif proto_mode == "median":
            prototypes[c] = np.median(Xc, axis=0)
        else:
            med = np.median(Xc, axis=0)
            mad_val = median_abs_deviation(Xc, axis=0)
            clipped = np.clip(Xc, med - 0.5 * mad_val, med + 0.5 * mad_val)
            prototypes[c] = clipped.mean(axis=0)

    if dist_mode == "euclidean":
        dists = np.linalg.norm(
            prototypes[:, None, :, :] - prototypes[None, :, :, :], axis=-1
        )
    else:
        dists = np.zeros((n_classes, n_classes, n_channels), dtype=np.float64)
        for k in range(n_classes):
            for m in range(k + 1, n_classes):
                for ch in range(n_channels):
                    d_val = float(
                        aeon_distance(
                            prototypes[k, ch], prototypes[m, ch], method="dtw"
                        )
                    )
                    dists[k, m, ch] = d_val
                    dists[m, k, ch] = d_val

    selected_per_class = []
    for k in range(n_classes):
        if method == "ecs":
            scores = dists[k].sum(axis=0)
            order = np.argsort(-scores, kind="mergesort")
            chosen = _detect_knee_channels(scores[order], order)
            selected_per_class.append(np.array(sorted(chosen), dtype=int))
        else:
            union_ch = set()
            for m in range(n_classes):
                if m == k:
                    continue
                scores = dists[k, m]
                order = np.argsort(-scores, kind="mergesort")
                union_ch.update(_detect_knee_channels(scores[order], order))
            selected_per_class.append(np.array(sorted(union_ch), dtype=int))

    return selected_per_class
