"""Pruning random convolution kernels (POCKET) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["POCKETClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
from aeon.utils.validation import check_n_jobs
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _apply_rocket_kernels,
    _L2NormalisedRidgeClassifierCV,
    _prune_rocket_kernels,
)

_DEFAULT_K_GRID = (10.0, 1.0, 0.1, 0.01, 100.0, 1000.0)


class POCKETClassifier(BaseClassifier):
    """Pruning random convolution kernels (POCKET) classifier.

    POCKET [1]_ prunes redundant kernels from ROCKET or MiniRocket in two
    sequential stages from a feature selection perspective. In Stage 1, an
    accelerated ADMM solver fits a group-sparse linear classifier with a
    dynamically varying soft threshold on the ``l_{2,1}`` norm so that at most
    ``int(n_kernels * remain_ratio)`` kernel feature groups remain non-zero,
    while keeping a constant ratio ``k`` between the group-sparsity and data-fit
    penalties so the Woodbury matrix inverse is computed only once. For ROCKET,
    the PPV and max features of each kernel form a group of size two and are
    pruned together along with the kernel itself. In the optional Stage 2, a
    ridge classifier with leave-one-out cross-validation is refitted on the
    unpruned kernels' features.

    This follows the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of initial kernels (or features for MiniRocket).
    remain_ratio : float, default=0.5
        Proportion of kernels to retain in ``(0, 1]``.
    n_iter : int, default=50
        Number of Stage 1 ADMM iterations.
    k : float, sequence of float or None, default=None
        Penalty ratio ``rho_3 / rho_2`` in Stage 1. If None, ``k`` is selected
        from ``[10, 1, 0.1, 0.01, 100, 1000]`` using 5-fold stratified
        cross-validation on the training data (falling back to fewer folds or
        ``k=1.0`` when classes have too few instances, as in [1]_). If a single
        float is given, it is used directly without cross-validation.
    stage2 : bool, default=True
        If True, refit an ``l_2``-regularised ridge classifier on the retained
        features in Stage 2. If False, predict directly with the Stage 1
        thresholded weights.
    rocket_transform : {"rocket", "rocket-ppv", "minirocket"}, default="rocket"
        Base convolution transform to prune: ``"rocket"`` uses both PPV and max
        features grouped per kernel, ``"rocket-ppv"`` uses ROCKET PPV features
        only, and ``"minirocket"`` uses MiniRocket PPV features.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to initialise the transform.

    Attributes
    ----------
    k_ : float
        The Stage 1 penalty ratio used by the final model.
    kernel_mask_ : np.ndarray
        Boolean mask of kept kernels (or features for MiniRocket).
    n_remaining_kernels_ : int
        Number of kernels retained after Stage 1 pruning.
    classifier_ : _L2NormalisedRidgeClassifierCV or None
        The Stage 2 ridge classifier when ``stage2=True``, otherwise None.

    See Also
    --------
    SRocketClassifier, DetachRocketClassifier

    References
    ----------
    .. [1] Chen, Shaowu, Weize Sun, Lei Huang, Xiao Peng Li, Qingyuan Wang, and
       Deepu John. "POCKET: Pruning random convolution kernels for time series
       classification from a feature selection perspective." Knowledge-Based
       Systems 300 (2024): 112253.
    .. [2] https://github.com/ShaowuChen/POCKET

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     POCKETClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = POCKETClassifier(n_kernels=50, k=1.0, n_iter=10, random_state=0)
    >>> clf.fit(X, y)
    POCKETClassifier(k=1.0, n_iter=10, n_kernels=50, random_state=0)
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
        remain_ratio=0.5,
        n_iter=50,
        k=None,
        stage2=True,
        rocket_transform="rocket",
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.remain_ratio = remain_ratio
        self.n_iter = n_iter
        self.k = k
        self.stage2 = stage2
        self.rocket_transform = rocket_transform
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if not 0 < self.remain_ratio <= 1:
            raise ValueError(
                f"remain_ratio must be in (0, 1], got {self.remain_ratio}."
            )
        transform_type = self.rocket_transform.lower()
        if transform_type not in ("rocket", "rocket-ppv", "minirocket"):
            raise ValueError(
                f"Unknown rocket_transform={self.rocket_transform!r}. "
                "Choose from 'rocket', 'rocket-ppv', or 'minirocket'."
            )

        self._n_jobs = check_n_jobs(self.n_jobs)
        if self.n_classes_ == 1:
            self.k_ = 1.0 if self.k is None else float(np.atleast_1d(self.k)[0])
            self.kernel_mask_ = np.array([True])
            self.n_remaining_kernels_ = 1
            self.classifier_ = None
            return self

        seed = (
            int(self.random_state)
            if isinstance(self.random_state, (int, np.integer))
            else check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
        )
        y_ind = np.array([self._class_dictionary[label] for label in y], dtype=int)

        X_clean = np.nan_to_num(X, nan=0.0)
        if transform_type in ("rocket", "rocket-ppv"):
            trf = Rocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=seed,
            ).fit(X_clean)
            self._rocket_normalise = trf.normalise
            full_kernels = trf.kernels
            Xt_raw = _apply_rocket_kernels(
                X_clean, full_kernels, self._rocket_normalise, self._n_jobs
            ).astype(np.float64)
            if transform_type == "rocket-ppv":
                Xt_raw = np.ascontiguousarray(Xt_raw[:, ::2])
                group_size = 1
            else:
                Xt_raw = np.ascontiguousarray(Xt_raw)
                group_size = 2
        else:
            X_mr = self._pad_minirocket(X_clean)
            self._minirocket = MiniRocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=seed,
            ).fit(X_mr)
            Xt_raw = self._minirocket.transform(X_mr).astype(np.float64)
            group_size = 1

        n_groups = Xt_raw.shape[1] // group_size
        if self.remain_ratio >= 1.0 or n_groups <= 1:
            remain_num = n_groups
        else:
            remain_num = min(max(1, int(n_groups * self.remain_ratio)), n_groups - 1)

        # Normalise each feature column to zero mean and unit L2 norm for Stage 1
        self._stage1_mean = Xt_raw.mean(axis=0)
        Xt_centred = Xt_raw - self._stage1_mean
        norm = np.linalg.norm(Xt_centred, axis=0)
        self._stage1_norm = np.where(norm == 0, 1.0, norm)
        Xt_norm = Xt_centred / self._stage1_norm

        self.k_ = self._select_k(Xt_norm, y_ind, remain_num, group_size)
        sparse_w, kept_groups, y_mean = _pocket_stage1_fit(
            Xt_norm,
            y_ind,
            self.n_classes_,
            self.k_,
            remain_num,
            self.n_iter,
            group_size,
        )

        self.kernel_mask_ = kept_groups
        self.n_remaining_kernels_ = int(np.sum(kept_groups))
        if group_size == 2:
            feat_mask = np.repeat(kept_groups, 2)
        else:
            feat_mask = kept_groups

        if transform_type in ("rocket", "rocket-ppv"):
            self._pruned_kernels = _prune_rocket_kernels(full_kernels, kept_groups)

        if self.stage2:
            self.classifier_ = _L2NormalisedRidgeClassifierCV().fit(
                Xt_raw[:, feat_mask], y_ind
            )
            self._stage1_weights = None
            self._stage1_y_mean = None
        else:
            self.classifier_ = None
            self._stage1_weights = sparse_w[feat_mask]
            self._stage1_mean = self._stage1_mean[feat_mask]
            self._stage1_norm = self._stage1_norm[feat_mask]
            self._stage1_y_mean = y_mean

        return self

    def _select_k(self, Xt_norm, y_ind, remain_num, group_size):
        if self.k is not None and np.ndim(self.k) == 0:
            val = float(self.k)
            if val <= 0:
                raise ValueError(f"k must be positive, got {val}.")
            return val

        k_candidates = _DEFAULT_K_GRID if self.k is None else [float(v) for v in self.k]
        if any(v <= 0 for v in k_candidates):
            raise ValueError("All candidate k values must be positive.")
        if len(k_candidates) == 1:
            return k_candidates[0]

        min_class_count = int(np.bincount(y_ind).min())
        if min_class_count < 2:
            return 1.0 if 1.0 in k_candidates else k_candidates[0]

        n_splits = min(5, min_class_count)
        skf = StratifiedKFold(n_splits=n_splits)
        fold_scores = np.zeros(len(k_candidates))

        for train_idx, val_idx in skf.split(Xt_norm, y_ind):
            X_tr, y_tr = Xt_norm[train_idx], y_ind[train_idx]
            X_val, y_val = Xt_norm[val_idx], y_ind[val_idx]
            precomp = _precompute_stage1_gram(X_tr, y_tr, self.n_classes_)
            for j, cand_k in enumerate(k_candidates):
                sw, _, ym = _pocket_stage1_fit(
                    X_tr,
                    y_tr,
                    self.n_classes_,
                    cand_k,
                    remain_num,
                    self.n_iter,
                    group_size,
                    precomputed=precomp,
                )
                preds = np.argmax(X_val @ sw + ym, axis=1)
                fold_scores[j] += np.mean(preds == y_val)

        return float(k_candidates[int(np.argmax(fold_scores))])

    def _transform_pruned(self, X):
        X_clean = np.nan_to_num(X, nan=0.0)
        transform_type = self.rocket_transform.lower()
        if transform_type in ("rocket", "rocket-ppv"):
            Xt = _apply_rocket_kernels(
                X_clean, self._pruned_kernels, self._rocket_normalise, self._n_jobs
            ).astype(np.float64)
            if transform_type == "rocket-ppv":
                Xt = Xt[:, ::2]
            return Xt

        X_mr = self._pad_minirocket(X_clean)
        Xt = self._minirocket.transform(X_mr).astype(np.float64)
        return Xt[:, self.kernel_mask_]

    @staticmethod
    def _pad_minirocket(X):
        if X.shape[2] < 9:
            X = np.pad(X, ((0, 0), (0, 0), (0, 9 - X.shape[2])))
        return X

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        Xt = self._transform_pruned(X)
        if self.stage2:
            preds = self.classifier_.predict(Xt)
        else:
            Xt_norm = (Xt - self._stage1_mean) / self._stage1_norm
            preds = np.argmax(
                Xt_norm @ self._stage1_weights + self._stage1_y_mean, axis=1
            )
        return self.classes_[preds]

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
        return {"n_kernels": 20, "n_iter": 10, "k": 1.0}


def _precompute_stage1_gram(X, y_ind, n_classes):
    """Precompute eigendecomposition and target projection for Stage 1 ADMM."""
    n_samples, n_features = X.shape
    y_coded = np.full((n_samples, n_classes), -1.0)
    y_coded[np.arange(n_samples), y_ind] = 1.0
    y_mean = y_coded.mean(axis=0)
    y_demean = y_coded - y_mean
    xty = X.T @ y_demean

    if n_samples < n_features:
        eigvals, eigvecs = np.linalg.eigh(X @ X.T)
        eigvals = np.maximum(eigvals, 0.0)
        return ("woodbury", eigvals, eigvecs, xty, y_mean)

    eigvals, eigvecs = np.linalg.eigh(X.T @ X)
    eigvals = np.maximum(eigvals, 0.0)
    return ("direct", eigvals, eigvecs, xty, y_mean)


def _pocket_stage1_fit(
    X,
    y_ind,
    n_classes,
    k,
    remain_num,
    n_iter,
    group_size,
    precomputed=None,
):
    """Run the POCKET Stage 1 group-sparse ADMM loop."""
    if precomputed is None:
        precomputed = _precompute_stage1_gram(X, y_ind, n_classes)
    mode, eigvals, eigvecs, xty, y_mean = precomputed

    n_features = X.shape[1]
    n_groups = n_features // group_size
    theta = np.zeros((n_features, n_classes))
    u_dual = np.zeros((n_features, n_classes))

    if mode == "woodbury":
        inv_diag = 1.0 / (k + eigvals)
    else:
        inv_diag = 1.0 / (k + eigvals)

    w = np.zeros((n_features, n_classes))
    shrinkage = np.ones(n_groups)

    for _ in range(n_iter):
        rhs = k * (theta + u_dual) + xty
        if mode == "woodbury":
            proj = eigvecs.T @ (X @ rhs)
            w = (rhs - X.T @ (eigvecs @ (inv_diag[:, None] * proj))) / k
        else:
            w = eigvecs @ (inv_diag[:, None] * (eigvecs.T @ rhs))

        w_tilde_group = (w - u_dual).reshape(n_groups, -1)
        group_norm = np.linalg.norm(w_tilde_group, axis=1)

        if remain_num >= n_groups:
            threshold = 0.0
            shrinkage = np.ones(n_groups)
        else:
            threshold = np.sort(group_norm)[::-1][remain_num]
            with np.errstate(divide="ignore", invalid="ignore"):
                shrinkage = np.maximum(0.0, 1.0 - threshold / group_norm)
            shrinkage[~np.isfinite(shrinkage)] = 0.0

        theta = (shrinkage[:, None] * w_tilde_group).reshape(n_features, n_classes)
        u_dual = u_dual + theta - w

    kept_groups = shrinkage > 0
    if not np.any(kept_groups):
        kept_groups = np.zeros(n_groups, dtype=bool)
        order = np.argsort(-np.linalg.norm(w.reshape(n_groups, -1), axis=1))
        kept_groups[order[: max(1, remain_num)]] = True

    sparse_group_w = w.reshape(n_groups, -1).copy()
    sparse_group_w[~kept_groups] = 0.0
    sparse_w = sparse_group_w.reshape(n_features, n_classes)
    return sparse_w, kept_groups, y_mean
