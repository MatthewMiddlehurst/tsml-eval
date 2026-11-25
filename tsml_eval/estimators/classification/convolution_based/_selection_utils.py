"""Shared utilities for kernel and feature selection ROCKET classifiers."""

__maintainer__ = ["MatthewMiddlehurst"]

import copy
import warnings
from contextlib import contextmanager

import numpy as np
from aeon.utils.validation import check_n_jobs
from numba import config
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV


def _check_numba_jobs(n_jobs):
    """Resolve ``n_jobs`` and cap at numba's configured threadpool limit."""
    return min(check_n_jobs(n_jobs), config.NUMBA_NUM_THREADS)


@contextmanager
def _quiet_ridge_warnings():
    """Suppress expected ill-conditioned matrix warnings during ridge sweeps."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Singular matrix in solving dual problem",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        yield


def _prune_rocket_kernels(kernels, kernel_mask):
    """Slice an aeon ``Rocket`` kernel tuple to keep only selected kernels.

    Parameters
    ----------
    kernels : tuple
        The 7-tuple ``(weights, lengths, biases, dilations, paddings,
        num_channel_indices, channel_indices)`` produced by ``Rocket``.
    kernel_mask : array-like of bool
        Boolean mask of length ``n_kernels`` indicating which kernels to retain.

    Returns
    -------
    pruned : tuple
        A 7-tuple in the same format containing only the retained kernels.
    """
    weights, lengths, biases, dilations, paddings, n_ch, ch_idx = kernels
    mask = np.asarray(kernel_mask, dtype=bool)
    kept = np.flatnonzero(mask)

    w_sizes = n_ch * lengths
    w_ends = np.cumsum(w_sizes)
    w_starts = w_ends - w_sizes
    c_ends = np.cumsum(n_ch)
    c_starts = c_ends - n_ch

    if len(kept) == 0:
        kept_weights = np.empty(0, dtype=weights.dtype)
        kept_ch_idx = np.empty(0, dtype=ch_idx.dtype)
    else:
        kept_weights = np.concatenate([weights[w_starts[i] : w_ends[i]] for i in kept])
        kept_ch_idx = np.concatenate([ch_idx[c_starts[i] : c_ends[i]] for i in kept])

    return (
        kept_weights,
        lengths[mask],
        biases[mask],
        dilations[mask],
        paddings[mask],
        n_ch[mask],
        kept_ch_idx,
    )


def _rocket_transform(rocket, X, kernels=None):
    """Transform ``X`` with a fitted ``Rocket``, optionally using other kernels.

    Going through ``Rocket.transform`` keeps the normalisation, input dtype and
    threading of the installed aeon version, so the features match those of the
    full transform used by the reference code. ``kernels`` replaces the fitted
    kernel tuple, e.g. with one pruned by ``_prune_rocket_kernels``.
    """
    if kernels is not None:
        rocket = copy.copy(rocket)
        rocket.kernels = kernels
    return rocket.transform(X)


class _L2NormalisedRidgeClassifierCV:
    """RidgeClassifierCV with column centring and unit L2-norm scaling.

    Reproduces ``RidgeClassifierCV(alphas=alphas, normalize=True)`` from
    ``scikit-learn<=0.24``, used in the S-ROCKET and POCKET reference codes.
    """

    def __init__(self, alphas=None):
        self.alphas = np.logspace(-3, 3, 10) if alphas is None else alphas

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        scale = np.linalg.norm(Xc, axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)

        with _quiet_ridge_warnings():
            self.ridge_ = RidgeClassifierCV(alphas=self.alphas).fit(Xc / self.scale_, y)

        self.alpha_ = float(self.ridge_.alpha_)
        self.classes_ = self.ridge_.classes_
        self.coef_ = np.atleast_2d(self.ridge_.coef_) / self.scale_
        self.intercept_ = np.atleast_1d(self.ridge_.intercept_) - (
            self.coef_ @ self.mean_
        )
        return self

    def predict(self, X):
        Xn = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return self.ridge_.predict(Xn)


def _sequential_feature_detachment(
    alpha,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    drop_ratio=0.05,
    n_steps=150,
    multiclass_type="max",
):
    """Run Sequential Feature Detachment (SFD) on a scaled feature matrix.

    Parameters
    ----------
    alpha : float
        Fixed ridge regularisation parameter used throughout detachment.
    X_train : np.ndarray of shape (n_cases, n_features)
        Standardised training feature matrix.
    y_train : np.ndarray of shape (n_cases,)
        Training labels.
    X_val : np.ndarray of shape (n_val, n_features) or None, default=None
        Standardised validation feature matrix.
    y_val : np.ndarray of shape (n_val,) or None, default=None
        Validation labels.
    drop_ratio : float, default=0.05
        Fraction of active features dropped at each detachment step.
    n_steps : int, default=150
        Maximum number of detachment steps.
    multiclass_type : {"max", "norm", "avg"}, default="max"
        Aggregation of multiclass ridge coefficients into feature importances.

    Returns
    -------
    retained_ratios : np.ndarray
        Proportion of features retained at each step.
    train_scores : np.ndarray
        Training accuracy at each step.
    val_scores : np.ndarray or None
        Validation accuracy at each step, or None if ``X_val`` is None.
    importance_matrix : np.ndarray
        Feature importance vector of shape ``(n_actual_steps, n_features)``.
    """
    multiclass_methods = {
        "max": lambda coef: np.linalg.norm(coef, axis=0, ord=np.inf),
        "norm": lambda coef: np.linalg.norm(coef, axis=0, ord=2),
        "avg": lambda coef: np.linalg.norm(coef, axis=0, ord=1),
    }
    if multiclass_type not in multiclass_methods:
        raise ValueError(
            f"Unknown multiclass_type={multiclass_type!r}. "
            "Choose from 'max', 'norm', or 'avg'."
        )

    classifier = RidgeClassifier(alpha=alpha)
    with _quiet_ridge_warnings():
        classifier.fit(X_train, y_train)

    total_features = X_train.shape[1]
    retained_ratios_uniform = np.power(1.0 - drop_ratio, np.arange(n_steps))
    retained_features = np.unique(
        (retained_ratios_uniform * total_features).astype(int)
    )[::-1]
    retained_features = retained_features[retained_features > 0]
    if len(retained_features) == 0:
        retained_features = np.array([1], dtype=int)
    retained_ratios = retained_features / total_features

    def _calc_importance(coef):
        if coef.ndim > 1 and coef.shape[0] > 1:
            return multiclass_methods[multiclass_type](coef)
        return np.abs(coef).ravel()

    feature_importance = _calc_importance(classifier.coef_)
    selection_mask = np.zeros(total_features, dtype=bool)
    importance_matrix = np.zeros((len(retained_features), total_features))
    train_scores = np.zeros(len(retained_features))
    val_scores = (
        np.zeros(len(retained_features))
        if X_val is not None and y_val is not None
        else None
    )

    for step, n_feat in enumerate(retained_features):
        selected_idxs = np.argsort(feature_importance)[-n_feat:]
        selection_mask[:] = False
        selection_mask[selected_idxs] = True

        X_sub = X_train[:, selection_mask]
        with _quiet_ridge_warnings():
            classifier.fit(X_sub, y_train)

        train_scores[step] = classifier.score(X_sub, y_train)
        if val_scores is not None:
            val_scores[step] = classifier.score(X_val[:, selection_mask], y_val)

        feature_importance[~selection_mask] = 0.0
        importance_matrix[step] = feature_importance
        feature_importance[selection_mask] = _calc_importance(classifier.coef_)

    return retained_ratios, train_scores, val_scores, importance_matrix


def _select_optimal_pruning(
    retained_ratios, val_scores, trade_off=0.1, smoothing_points=3
):
    """Select the optimal SFD step from validation accuracies."""
    norm_factor = val_scores[0] if val_scores[0] > 0 else 1.0
    x_vec = 1.0 - retained_ratios
    y_vec = val_scores / norm_factor

    pts = min(smoothing_points, len(y_vec))
    if pts > 1:
        box = np.ones(pts) / pts
        y_vec = np.convolve(y_vec, box, mode="same")

    optimality_curve = trade_off * x_vec + y_vec
    best_idx = int(np.argmax(optimality_curve))
    return best_idx, float(retained_ratios[best_idx])


def _retrain_optimal_ridge(feature_mask, X_scaled, y, fixed_alpha=None, alphas=None):
    """Retrain a ``RidgeClassifier`` on the selected scaled features."""
    if alphas is None:
        alphas = np.logspace(-10, 10, 20)
    X_masked = X_scaled[:, feature_mask]
    if fixed_alpha is None:
        cv_clf = RidgeClassifierCV(alphas=alphas)
        with _quiet_ridge_warnings():
            cv_clf.fit(X_masked, y)
        fixed_alpha = float(cv_clf.alpha_)

    clf = RidgeClassifier(alpha=fixed_alpha)
    with _quiet_ridge_warnings():
        clf.fit(X_masked, y)
    train_acc = float(clf.score(X_masked, y))
    return clf, train_acc


def _minirocket_block_channels(params, combo_params, n_channels):
    """Map one MiniRocket/MultiRocket pooling block's features to channels."""
    dilations = np.asarray(params[-3], dtype=int)
    npd = np.asarray(params[-2], dtype=int)
    n_features = 84 * int(npd.sum())

    if len(combo_params) != 5:
        return np.ones((n_features, n_channels), dtype=float)

    n_ch_per_comb, ch_indices = combo_params[0], combo_params[1]
    comb_of_feat = np.concatenate(
        [np.repeat(d * 84 + np.arange(84), npd[d]) for d in range(len(dilations))]
    )
    n_combinations = len(n_ch_per_comb)
    indicator = np.zeros((n_combinations, n_channels), dtype=float)
    rows = np.repeat(np.arange(n_combinations), n_ch_per_comb)
    indicator[rows, ch_indices] = 1.0
    return indicator[comb_of_feat]


def _get_kernel_channel_indicator(transformer, rocket_transform, n_channels):
    """Return ``(n_features, n_channels)`` channel indicator matrix for a transform."""
    if rocket_transform == "minirocket":
        return _minirocket_block_channels(
            transformer.parameters, transformer.parameters, n_channels
        )
    if rocket_transform == "multirocket":
        base_block = _minirocket_block_channels(
            transformer.parameter, transformer.parameter, n_channels
        )
        diff_block = _minirocket_block_channels(
            transformer.parameter1, transformer.parameter, n_channels
        )
        return np.concatenate(
            [np.tile(base_block, (4, 1)), np.tile(diff_block, (4, 1))], axis=0
        )
    if rocket_transform == "rocket":
        n_ch, ch_idx = transformer.kernels[5], transformer.kernels[6]
        n_kernels = len(n_ch)
        indicator = np.zeros((n_kernels, n_channels), dtype=float)
        rows = np.repeat(np.arange(n_kernels), n_ch)
        indicator[rows, ch_idx] = 1.0
        return np.repeat(indicator, 2, axis=0)
    raise ValueError(f"Unsupported rocket_transform: {rocket_transform!r}")
