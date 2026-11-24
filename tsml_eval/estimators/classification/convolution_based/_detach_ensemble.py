"""Detach-Rocket Ensemble classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DetachRocketEnsembleClassifier"]

import numpy as np
from aeon.classification import BaseClassifier

from tsml_eval.estimators.classification.convolution_based._detach_rocket import (
    DetachRocketClassifier,
    _prepare_detach_split,
)
from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _check_numba_jobs,
    _get_kernel_channel_indicator,
)


class DetachRocketEnsembleClassifier(BaseClassifier):
    """Detach-Rocket Ensemble classifier.

    Detach-Rocket Ensemble [1]_ combines ``n_estimators`` independently seeded
    :class:`DetachRocketClassifier` models (using MiniRocket by default, as in
    [1]_) trained on a shared stratified train/validation split. Predictions are
    combined by accuracy-weighted (``"soft"``) or uniform (``"hard"``) voting.

    After fitting, multivariate channel relevance scores are computed across all
    ensemble members by dividing each retained feature's SFD importance equally
    among the input channels present in its kernel combination, normalising per
    member, and aggregating across members via the median (renormalised to unit
    sum, as in Section 3.2 of [1]_) or the mean [2]_.

    This follows [1]_ and the reference implementation [2]_.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of Detach-ROCKET models in the ensemble.
    n_kernels : int, default=10000
        Feature budget per ensemble member. For ``rocket_transform="multirocket"``,
        ``max(84, n_kernels // 8)`` kernels are passed to ``MultiRocket`` so each
        member generates approximately ``n_kernels`` features, matching [1]_ and
        [2]_.
    trade_off : float, default=0.1
        Weighting factor ``c`` passed to each member's SFD step selector.
    set_percentage : float or None, default=None
        Optional fixed feature retention percentage in ``(0, 100]``.
    val_ratio : float, default=0.33
        Fraction of training instances held out for validation step selection.
    drop_ratio : float, default=0.05
        Fraction ``r`` of active features dropped at each SFD step.
    n_steps : int, default=150
        Number of SFD steps per member.
    smoothing_points : int, default=3
        Moving-average window width for validation curve smoothing.
    recompute_alpha : bool, default=True
        Whether each member re-estimates its ridge regularisation alpha after
        pruning.
    multiclass_type : {"max", "norm", "avg"}, default="max"
        Norm used to aggregate multiclass ridge coefficients into feature
        importances.
    voting : {"soft", "hard"}, default="soft"
        Ensemble vote weighting: ``"soft"`` weights each member's vote by its
        retrained training accuracy, while ``"hard"`` uses equal weights.
    channel_aggregation : {"median", "mean"}, default="median"
        Aggregation across members when computing ``channel_relevance_``.
        ``"median"`` (renormalised to sum to 1) is the publication method in
        Section 3.2 of [1]_; ``"mean"`` is also supported as in [2]_.
    rocket_transform : {"minirocket", "multirocket", "rocket"}, default="minirocket"
        Base ROCKET-family transform for each member.
    n_jobs : int, default=1
        The number of jobs to run in parallel for member transforms. ``-1``
        means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling member seeds and the
        internal train/validation split.

    Attributes
    ----------
    estimators_ : list of DetachRocketClassifier
        Fitted Detach-ROCKET ensemble members.
    channel_relevance_ : np.ndarray of shape (n_channels,)
        Normalised relevance score for each input channel.

    See Also
    --------
    DetachRocketClassifier

    References
    ----------
    .. [1] Solana, Alvaro, Erik Fransen, and Gonzalo Uribarri. "Classification
       of raw MEG/EEG data with Detach-Rocket Ensemble: an improved ROCKET
       algorithm for multivariate time series analysis." International Workshop
       on Advanced Analytics and Learning on Temporal Data (AALTD), 2024.
    .. [2] https://github.com/gon-uri/detach_rocket

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     DetachRocketEnsembleClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=3, random_state=0)
    >>> clf = DetachRocketEnsembleClassifier(
    ...     n_estimators=3, n_kernels=84, n_steps=10, random_state=0
    ... )
    >>> clf.fit(X, y)
    DetachRocketEnsembleClassifier(n_estimators=3, n_kernels=84, n_steps=10,
                                   random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_estimators=25,
        n_kernels=10000,
        trade_off=0.1,
        set_percentage=None,
        val_ratio=0.33,
        drop_ratio=0.05,
        n_steps=150,
        smoothing_points=3,
        recompute_alpha=True,
        multiclass_type="max",
        voting="soft",
        channel_aggregation="median",
        rocket_transform="minirocket",
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.val_ratio = val_ratio
        self.drop_ratio = drop_ratio
        self.n_steps = n_steps
        self.smoothing_points = smoothing_points
        self.recompute_alpha = recompute_alpha
        self.multiclass_type = multiclass_type
        self.voting = voting
        self.channel_aggregation = channel_aggregation
        self.rocket_transform = rocket_transform
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_estimators < 1:
            raise ValueError(
                f"n_estimators must be at least 1, got {self.n_estimators}."
            )
        if self.voting not in ("soft", "hard"):
            raise ValueError(
                f"Unknown voting={self.voting!r}. Choose from 'soft' or 'hard'."
            )
        if self.channel_aggregation not in ("median", "mean"):
            raise ValueError(
                f"Unknown channel_aggregation={self.channel_aggregation!r}. "
                "Choose from 'median' or 'mean'."
            )
        DetachRocketClassifier._validate_hyperparameters(self)
        transform_type = self.rocket_transform.lower()

        self._n_jobs = _check_numba_jobs(self.n_jobs)
        n_channels = X.shape[1]

        if self.n_classes_ == 1:
            self.estimators_ = []
            self.channel_relevance_ = np.full(n_channels, 1.0 / n_channels)
            return self

        if isinstance(self.random_state, np.random.RandomState):
            base_seed = int(self.random_state.randint(np.iinfo(np.int32).max))
            seed_rng = np.random.default_rng(base_seed)
            split_seed = base_seed
        elif isinstance(self.random_state, (int, np.integer)):
            seed_rng = np.random.default_rng(int(self.random_state))
            split_seed = int(self.random_state)
        else:
            seed_rng = np.random.default_rng(self.random_state)
            split_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))

        member_seeds = (
            seed_rng.integers(0, 2**32 - 1, size=self.n_estimators, dtype=np.int64)
            & np.iinfo(np.int32).max
        )
        member_kernels = (
            max(84, self.n_kernels // 8)
            if transform_type == "multirocket"
            else self.n_kernels
        )

        fallback_pct = 5.0 if transform_type == "multirocket" else 10.0
        y_ind = np.array([self._class_dictionary[label] for label in y], dtype=int)
        X_tr, X_val, y_tr, y_val, eff_pct = _prepare_detach_split(
            X,
            y_ind,
            self.n_classes_,
            self.set_percentage,
            self.val_ratio,
            split_seed,
            fallback_pct=fallback_pct,
        )

        self.estimators_ = []
        for m in range(self.n_estimators):
            member = DetachRocketClassifier(
                n_kernels=member_kernels,
                trade_off=self.trade_off,
                set_percentage=self.set_percentage,
                val_ratio=self.val_ratio,
                drop_ratio=self.drop_ratio,
                n_steps=self.n_steps,
                smoothing_points=self.smoothing_points,
                recompute_alpha=self.recompute_alpha,
                multiclass_type=self.multiclass_type,
                rocket_transform=transform_type,
                n_jobs=self._n_jobs,
                random_state=int(member_seeds[m]),
            )
            member._validate_hyperparameters()
            member._n_jobs = self._n_jobs
            member.classes_ = self.classes_
            member.n_classes_ = self.n_classes_
            member._class_dictionary = self._class_dictionary
            member._fit_from_split(
                X_tr, y_tr, X_val, y_val, int(member_seeds[m]), eff_pct
            )
            member._is_fitted = True
            self.estimators_.append(member)

        self.channel_relevance_ = self.estimate_channel_relevance(
            aggregation=self.channel_aggregation, n_channels=n_channels
        )
        return self

    def estimate_channel_relevance(self, aggregation=None, n_channels=None):
        """Compute normalised multivariate channel relevance across members.

        Parameters
        ----------
        aggregation : {"median", "mean"} or None, default=None
            How to aggregate per-member channel relevance vectors. Defaults to
            ``self.channel_aggregation``.
        n_channels : int or None, default=None
            Number of input channels. Inferred from ``metadata_`` if None.

        Returns
        -------
        relevance : np.ndarray of shape (n_channels,)
            Normalised relevance score for each channel.
        """
        agg = self.channel_aggregation if aggregation is None else aggregation
        if agg not in ("median", "mean"):
            raise ValueError(
                f"Unknown aggregation={agg!r}. Choose from 'median' or 'mean'."
            )
        if n_channels is None:
            n_channels = len(self.channel_relevance_)

        if len(self.estimators_) == 0:
            return np.full(n_channels, 1.0 / n_channels)

        transform_type = self.rocket_transform.lower()
        rel_matrix = np.zeros((len(self.estimators_), n_channels), dtype=float)

        for m, member in enumerate(self.estimators_):
            weights = member.feature_importance_matrix_[member.selected_step_index_]
            sel = weights > 0
            if not np.any(sel):
                continue

            indicator = _get_kernel_channel_indicator(
                member._transformer, transform_type, n_channels
            )[sel]
            ch_counts = indicator.sum(axis=1)
            valid = ch_counts > 0
            if not np.any(valid):
                continue

            per_feat = weights[sel][valid] / ch_counts[valid]
            weighted = indicator[valid] * per_feat[:, None]
            total = float(weighted.sum())
            if total > 0:
                rel_matrix[m] = weighted.sum(axis=0) / total

        if agg == "mean":
            mean_rel = np.mean(rel_matrix, axis=0)
            total = float(mean_rel.sum())
            if total > 0:
                return mean_rel / total
            return np.full(n_channels, 1.0 / n_channels)

        med_rel = np.median(rel_matrix, axis=0)
        total = float(med_rel.sum())
        if total > 0:
            return med_rel / total
        return np.full(n_channels, 1.0 / n_channels)

    def _predict_proba(self, X):
        n_cases = len(X)
        if self.n_classes_ == 1:
            return np.ones((n_cases, 1), dtype=float)

        votes = np.zeros((n_cases, self.n_classes_), dtype=float)
        rows = np.arange(n_cases)
        for member in self.estimators_:
            Xt_scaled = member._transform_selected_scaled(X)
            preds_ind = member.classifier_.predict(Xt_scaled)
            weight = float(member.acc_train_) if self.voting == "soft" else 1.0
            votes[rows, preds_ind] += weight

        row_sums = votes.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).ravel()
        if np.any(zero_rows):
            votes[zero_rows] = 1.0 / self.n_classes_
            row_sums[zero_rows] = 1.0
        return votes / row_sums

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        probas = self._predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

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
        return {"n_estimators": 2, "n_kernels": 84, "n_steps": 10}
