"""Sequential feature detachment ROCKET (Detach-ROCKET) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DetachRocketClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _apply_rocket_kernels,
    _prune_rocket_kernels,
    _quiet_ridge_warnings,
    _retrain_optimal_ridge,
    _select_optimal_pruning,
    _sequential_feature_detachment,
)


class DetachRocketClassifier(BaseClassifier):
    """Sequential feature detachment ROCKET (Detach-ROCKET) classifier.

    Detach-ROCKET [1]_ prunes redundant random convolutional features using
    Sequential Feature Detachment (SFD). Starting from a standardised feature
    matrix and a baseline regularisation parameter selected by
    ``RidgeClassifierCV`` over ``np.logspace(-10, 10, 20)``, SFD iteratively
    drops a fixed proportion ``drop_ratio`` of the lowest-magnitude ridge
    weights over ``n_steps`` steps and refits the ridge classifier on the
    surviving features.

    When ``set_percentage`` is None, a stratified validation split of size
    ``val_ratio`` is held out to select the pruning step maximising
    ``trade_off * (1 - retained_ratio) + smoothed_normalised_val_accuracy``
    before retraining on the combined training and validation features. If the
    training set is too small or imbalanced for a valid stratified split, the
    fixed 10% retention fallback from Table 2 of [1]_ is used automatically.
    For ``rocket_transform="rocket"``, completely detached kernels are removed
    from the convolution tuple so inference only evaluates active kernels.

    This follows [1]_ and the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels for the base convolution transform.
    trade_off : float, default=0.1
        Weighting factor ``c`` balancing pruning rate and validation accuracy.
    set_percentage : float or None, default=None
        If given in ``(0, 100]``, selects the SFD step closest to retaining
        ``set_percentage`` percent of features without holding out a validation
        split (e.g. ``set_percentage=10`` for Detach-ROCKET 10% in [1]_).
    val_ratio : float, default=0.33
        Fraction of training instances reserved for validation step selection
        when ``set_percentage`` is None.
    drop_ratio : float, default=0.05
        Fraction ``r`` of active features detached per SFD step.
    n_steps : int, default=150
        Number of SFD steps.
    smoothing_points : int, default=3
        Moving-average window width applied to the normalised validation curve.
    recompute_alpha : bool, default=True
        If True, re-estimate the ridge regularisation alpha via LOOCV on the
        pruned features during final retraining.
    multiclass_type : {"max", "norm", "avg"}, default="max"
        Norm used to aggregate multiclass ridge coefficients into feature
        importances (``"max"`` is ``L_inf``, used in [1]_).
    rocket_transform : {"rocket", "minirocket", "multirocket"}, default="rocket"
        Base ROCKET-family transform.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for kernel generation and the
        internal train/validation split.

    Attributes
    ----------
    feature_mask_ : np.ndarray
        Boolean mask of retained features in the full transform output.
    retained_ratio_ : float
        Proportion of features retained at the selected SFD step.
    selected_step_index_ : int
        Index of the selected step along the SFD curve.
    feature_importance_matrix_ : np.ndarray
        Matrix of SFD feature importances across steps.
    full_model_alpha_ : float
        Regularisation parameter selected on the unpruned feature set.
    optimal_alpha_ : float
        Regularisation parameter of the final pruned ridge classifier.
    acc_train_ : float
        Training accuracy of the final retrained ridge classifier.
    classifier_ : RidgeClassifier or None
        Final ridge classifier fitted on the retained scaled features.

    See Also
    --------
    DetachRocketEnsembleClassifier, POCKETClassifier, SRocketClassifier

    References
    ----------
    .. [1] Uribarri, Gonzalo, Federico Barone, Alessio Ansuini, and Erik
       Fransen. "Detach-ROCKET: sequential feature selection for time series
       classification with random convolutional kernels." Data Mining and
       Knowledge Discovery 38, no. 6 (2024): 3922-3947.
    .. [2] https://github.com/gon-uri/detach_rocket

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     DetachRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = DetachRocketClassifier(n_kernels=50, n_steps=15, random_state=0)
    >>> clf.fit(X, y)
    DetachRocketClassifier(n_kernels=50, n_steps=15, random_state=0)
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
        trade_off=0.1,
        set_percentage=None,
        val_ratio=0.33,
        drop_ratio=0.05,
        n_steps=150,
        smoothing_points=3,
        recompute_alpha=True,
        multiclass_type="max",
        rocket_transform="rocket",
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.val_ratio = val_ratio
        self.drop_ratio = drop_ratio
        self.n_steps = n_steps
        self.smoothing_points = smoothing_points
        self.recompute_alpha = recompute_alpha
        self.multiclass_type = multiclass_type
        self.rocket_transform = rocket_transform
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._validate_hyperparameters()
        self._n_jobs = check_n_jobs(self.n_jobs)

        if self.n_classes_ == 1:
            self.feature_mask_ = np.array([True])
            self.retained_ratio_ = 1.0
            self.selected_step_index_ = 0
            self.feature_importance_matrix_ = np.ones((1, 1))
            self.full_model_alpha_ = 1.0
            self.optimal_alpha_ = 1.0
            self.acc_train_ = 1.0
            self.classifier_ = None
            return self

        if isinstance(self.random_state, (int, np.integer)):
            trf_seed = int(self.random_state)
            split_seed = int(self.random_state)
        else:
            rng = check_random_state(self.random_state)
            trf_seed = int(rng.randint(np.iinfo(np.int32).max))
            split_seed = int(rng.randint(np.iinfo(np.int32).max))

        y_ind = np.array([self._class_dictionary[label] for label in y], dtype=int)
        X_tr, X_val, y_tr, y_val, eff_pct = _prepare_detach_split(
            X, y_ind, self.n_classes_, self.set_percentage, self.val_ratio, split_seed
        )
        self._fit_from_split(X_tr, y_tr, X_val, y_val, trf_seed, eff_pct)
        return self

    def _validate_hyperparameters(self):
        if self.set_percentage is not None and not 0 < self.set_percentage <= 100:
            raise ValueError(
                f"set_percentage must be in (0, 100], got {self.set_percentage}."
            )
        if not 0 < self.drop_ratio < 1:
            raise ValueError(f"drop_ratio must be in (0, 1), got {self.drop_ratio}.")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}.")
        if not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}.")
        if self.rocket_transform.lower() not in (
            "rocket",
            "minirocket",
            "multirocket",
        ):
            raise ValueError(
                f"Unknown rocket_transform={self.rocket_transform!r}. "
                "Choose from 'rocket', 'minirocket', or 'multirocket'."
            )

    def _fit_from_split(self, X_tr, y_tr, X_val, y_val, trf_seed, effective_percentage):
        transform_type = self.rocket_transform.lower()
        X_tr_clean = self._preprocess_series(X_tr, transform_type)

        if transform_type == "rocket":
            self._transformer = Rocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=trf_seed,
            ).fit(X_tr_clean)
            self._rocket_normalise = self._transformer.normalise
            feat_tr = _apply_rocket_kernels(
                X_tr_clean,
                self._transformer.kernels,
                self._rocket_normalise,
                self._n_jobs,
            )
        elif transform_type == "minirocket":
            self._transformer = MiniRocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=trf_seed,
            ).fit(X_tr_clean)
            feat_tr = np.asarray(self._transformer.transform(X_tr_clean))
        else:
            self._transformer = MultiRocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=trf_seed,
            ).fit(X_tr_clean)
            feat_tr = np.asarray(self._transformer.transform(X_tr_clean))

        scaler = StandardScaler(with_mean=True)
        feat_tr_scaled = scaler.fit_transform(feat_tr)

        use_val = effective_percentage is None and X_val is not None
        if use_val:
            X_val_clean = self._preprocess_series(X_val, transform_type)
            if transform_type == "rocket":
                feat_val = _apply_rocket_kernels(
                    X_val_clean,
                    self._transformer.kernels,
                    self._rocket_normalise,
                    self._n_jobs,
                )
            else:
                feat_val = np.asarray(self._transformer.transform(X_val_clean))
            feat_val_scaled = scaler.transform(feat_val)
        else:
            feat_val_scaled = None

        alphas = np.logspace(-10, 10, 20)
        full_clf = RidgeClassifierCV(alphas=alphas)
        with _quiet_ridge_warnings():
            full_clf.fit(feat_tr_scaled, y_tr)
        self.full_model_alpha_ = float(full_clf.alpha_)

        retained_ratios, _, val_scores, imp_matrix = _sequential_feature_detachment(
            self.full_model_alpha_,
            feat_tr_scaled,
            y_tr,
            X_val=feat_val_scaled,
            y_val=y_val if use_val else None,
            drop_ratio=self.drop_ratio,
            n_steps=self.n_steps,
            multiclass_type=self.multiclass_type,
        )
        self.feature_importance_matrix_ = imp_matrix

        if use_val:
            step_idx, _ = _select_optimal_pruning(
                retained_ratios,
                val_scores,
                trade_off=self.trade_off,
                smoothing_points=self.smoothing_points,
            )
        else:
            target_ratio = effective_percentage / 100.0
            step_idx = int(np.argmin(np.abs(retained_ratios - target_ratio)))

        self.selected_step_index_ = step_idx
        self.retained_ratio_ = float(retained_ratios[step_idx])
        mask = imp_matrix[step_idx] > 0
        if not np.any(mask):
            mask[int(np.argmax(imp_matrix[0]))] = True
        self.feature_mask_ = mask

        if use_val:
            X_retrain = np.concatenate([feat_tr_scaled, feat_val_scaled], axis=0)
            y_retrain = np.concatenate([y_tr, y_val], axis=0)
        else:
            X_retrain = feat_tr_scaled
            y_retrain = y_tr

        fixed_alpha = None if self.recompute_alpha else self.full_model_alpha_
        self.classifier_, self.acc_train_ = _retrain_optimal_ridge(
            self.feature_mask_, X_retrain, y_retrain, fixed_alpha=fixed_alpha
        )
        self.optimal_alpha_ = float(self.classifier_.alpha)

        self._selected_mean = scaler.mean_[self.feature_mask_]
        self._selected_scale = scaler.scale_[self.feature_mask_]

        if transform_type == "rocket":
            pair_mask = self.feature_mask_.reshape(-1, 2)
            active_kernels = pair_mask.any(axis=1)
            self._pruned_kernels = _prune_rocket_kernels(
                self._transformer.kernels, active_kernels
            )
            self._pruned_submask = pair_mask[active_kernels].ravel()

    @staticmethod
    def _preprocess_series(X, transform_type):
        X_clean = np.nan_to_num(X, nan=0.0)
        min_len = (
            10
            if transform_type == "multirocket"
            else (9 if transform_type == "minirocket" else 1)
        )
        if X_clean.shape[2] < min_len:
            X_clean = np.pad(X_clean, ((0, 0), (0, 0), (0, min_len - X_clean.shape[2])))
        return X_clean

    def _transform_selected_scaled(self, X):
        transform_type = self.rocket_transform.lower()
        X_clean = self._preprocess_series(X, transform_type)
        if transform_type == "rocket":
            Xt = _apply_rocket_kernels(
                X_clean, self._pruned_kernels, self._rocket_normalise, self._n_jobs
            )
            Xt_sel = Xt[:, self._pruned_submask]
        else:
            Xt = np.asarray(self._transformer.transform(X_clean))
            Xt_sel = Xt[:, self.feature_mask_]
        return (Xt_sel - self._selected_mean) / self._selected_scale

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        Xt_scaled = self._transform_selected_scaled(X)
        preds = self.classifier_.predict(Xt_scaled)
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
        return {"n_kernels": 20, "n_steps": 10}


def _prepare_detach_split(X, y_ind, n_classes, set_percentage, val_ratio, split_seed):
    """Build stratified train/validation split, falling back to 10% if impossible."""
    if set_percentage is not None:
        return X, None, y_ind, None, float(set_percentage)

    counts = np.bincount(y_ind, minlength=n_classes)
    if counts.min() < 2:
        return X, None, y_ind, None, 10.0

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y_ind,
            test_size=val_ratio,
            random_state=split_seed,
            stratify=y_ind,
        )
    except ValueError:
        return X, None, y_ind, None, 10.0

    if len(np.unique(y_tr)) < n_classes or len(np.unique(y_val)) < n_classes:
        return X, None, y_ind, None, 10.0

    return X_tr, X_val, y_tr, y_val, None
