"""Selection of random convolution kernels (S-ROCKET) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SRocketClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
from aeon.utils.validation import check_n_jobs
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _apply_rocket_kernels,
    _L2NormalisedRidgeClassifierCV,
    _prune_rocket_kernels,
)


class SRocketClassifier(BaseClassifier):
    """Selection of random convolution kernels (S-ROCKET) classifier.

    S-ROCKET [1]_ prunes redundant kernels from ROCKET or MiniRocket in three
    stages:

    1. **Pre-training**: Generate ``n_kernels`` random kernels, transform the
       training data (using PPV features by default), and fit an ``l_2``-norm
       standardised ``RidgeClassifierCV`` on the full feature set.
    2. **Kernel selection**: Keep the pre-trained classifier weights frozen and
       optimise a population of ``pop_size`` binary kernel-selection state
       vectors over ``n_epochs`` generations of binary differential evolution,
       minimising ``1 - (train_accuracy - active_kernel_ratio)``.
    3. **Post-training**: Retain only the kernels active in the best state
       vector and refit ``RidgeClassifierCV`` on their features.

    This follows [1]_ and the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of initial kernels (or features for MiniRocket).
    n_epochs : int, default=500
        Number of evolutionary optimisation generations in Step 2.
    pop_size : int, default=8
        Population size ``S`` of candidate binary state vectors (must be at
        least 3; ``pop_size=8`` is recommended in [1]_).
    mutation_factor : float, default=0.9
        Binary differential evolution mutation threshold ``F`` in ``[0, 1]``.
    crossover_rate : float, default=0.9
        Recombination rate ``Cr`` in ``[0, 1]``.
    init_population : {"ones", "half_bernoulli"}, default="ones"
        Population initialisation strategy. ``"ones"`` initialises every state
        vector to all ones, matching ``main.py`` in the official reference
        implementation [2]_ and avoiding premature collapse to empty state
        vectors on balanced binary datasets. ``"half_bernoulli"`` initialises
        half of the state vectors from ``Bernoulli(0.5)`` and the other half as
        all ones, as described in Section III-B of [1]_.
    rocket_transform : {"rocket", "rocket-ppvmax", "minirocket"}, default="rocket"
        Base convolution transform: ``"rocket"`` uses ROCKET PPV features only
        (default in [1]_), ``"rocket-ppvmax"`` uses both PPV and max features
        masked per kernel, and ``"minirocket"`` uses MiniRocket PPV features
        (S-MiniROCKET in Section IV-I of [1]_).
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for kernel generation and
        evolutionary optimisation.

    Attributes
    ----------
    kernel_mask_ : np.ndarray
        Boolean mask of kernels (or MiniRocket features) retained after Step 2.
    n_selected_kernels_ : int
        Number of kernels retained in the post-trained model.
    best_cost_ : float
        Objective value ``1 - (train_accuracy - active_kernel_ratio)`` of the
        selected binary state vector on the pre-trained model.
    pre_classifier_ : _L2NormalisedRidgeClassifierCV or None
        Pre-trained ridge classifier fitted on all features.
    classifier_ : _L2NormalisedRidgeClassifierCV or None
        Post-trained ridge classifier refitted on retained features.

    See Also
    --------
    POCKETClassifier, DetachRocketClassifier

    References
    ----------
    .. [1] Salehinejad, Hojjat, Yang Wang, Yuanhao Yu, Tang Jin, and Shahrokh
       Valaee. "S-ROCKET: Selective random convolution kernels for time series
       classification." IEEE/CAA Journal of Automatica Sinica (arXiv:2203.03445,
       2022).
    .. [2] https://github.com/salehinejad/srocket

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     SRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = SRocketClassifier(n_kernels=50, n_epochs=20, random_state=0)
    >>> clf.fit(X, y)
    SRocketClassifier(n_epochs=20, n_kernels=50, random_state=0)
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
        n_epochs=500,
        pop_size=8,
        mutation_factor=0.9,
        crossover_rate=0.9,
        init_population="ones",
        rocket_transform="rocket",
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_epochs = n_epochs
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.init_population = init_population
        self.rocket_transform = rocket_transform
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.pop_size < 3:
            raise ValueError(f"pop_size must be at least 3, got {self.pop_size}.")
        if not 0.0 <= self.mutation_factor <= 1.0:
            raise ValueError(
                f"mutation_factor must be in [0, 1], got {self.mutation_factor}."
            )
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(
                f"crossover_rate must be in [0, 1], got {self.crossover_rate}."
            )
        if self.init_population not in ("half_bernoulli", "ones"):
            raise ValueError(
                f"Unknown init_population={self.init_population!r}. "
                "Choose from 'half_bernoulli' or 'ones'."
            )
        transform_type = self.rocket_transform.lower()
        if transform_type not in ("rocket", "rocket-ppvmax", "minirocket"):
            raise ValueError(
                f"Unknown rocket_transform={self.rocket_transform!r}. "
                "Choose from 'rocket', 'rocket-ppvmax', or 'minirocket'."
            )

        self._n_jobs = check_n_jobs(self.n_jobs)
        if self.n_classes_ == 1:
            self.kernel_mask_ = np.array([True])
            self.n_selected_kernels_ = 1
            self.best_cost_ = 0.0
            self.pre_classifier_ = None
            self.classifier_ = None
            return self

        rng = check_random_state(self.random_state)
        trf_seed = (
            int(self.random_state)
            if isinstance(self.random_state, (int, np.integer))
            else rng.randint(np.iinfo(np.int32).max)
        )
        y_ind = np.array([self._class_dictionary[label] for label in y], dtype=int)

        X_clean = np.nan_to_num(X, nan=0.0)
        if transform_type in ("rocket", "rocket-ppvmax"):
            trf = Rocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=trf_seed,
            ).fit(X_clean)
            self._rocket_normalise = trf.normalise
            full_kernels = trf.kernels
            Xt = _apply_rocket_kernels(
                X_clean, full_kernels, self._rocket_normalise, self._n_jobs
            ).astype(np.float64)
            if transform_type == "rocket":
                Xt = np.ascontiguousarray(Xt[:, ::2])
                group_size = 1
            else:
                Xt = np.ascontiguousarray(Xt)
                group_size = 2
            n_units = self.n_kernels
        else:
            X_mr = self._pad_minirocket(X_clean)
            self._minirocket = MiniRocket(
                n_kernels=self.n_kernels,
                n_jobs=self._n_jobs,
                random_state=trf_seed,
            ).fit(X_mr)
            Xt = np.ascontiguousarray(
                self._minirocket.transform(X_mr), dtype=np.float64
            )
            group_size = 1
            n_units = Xt.shape[1]

        # Step 1: Pre-training on full features
        self.pre_classifier_ = _L2NormalisedRidgeClassifierCV().fit(Xt, y_ind)
        coef_t = np.ascontiguousarray(self.pre_classifier_.coef_.T)
        intercept = self.pre_classifier_.intercept_
        n_out = coef_t.shape[1]
        w_buf = np.empty((Xt.shape[1], n_out, self.pop_size), dtype=np.float64)

        # Step 2: Binary differential evolution over kernel state vectors
        pop = np.ones((n_units, self.pop_size), dtype=np.float64)
        if self.init_population == "half_bernoulli":
            half = self.pop_size // 2
            if half > 0:
                rand_init = np.round(rng.random_sample((n_units, self.pop_size)))
                pop[:, :half] = rand_init[:, :half]

        cost_pop, score_pop = _evaluate_srocket_population(
            Xt, y_ind, coef_t, intercept, pop, group_size, w_buf
        )

        for _ in range(self.n_epochs):
            parents = np.asarray(
                [rng.permutation(self.pop_size)[:3] for _ in range(self.pop_size)]
            )
            f_mask = rng.random_sample((n_units, self.pop_size)) < self.mutation_factor
            or_mask = np.logical_or(pop[:, parents[:, 1]], pop[:, parents[:, 2]])
            flip = np.logical_and(f_mask, or_mask)
            mutant = np.where(flip, 1.0 - pop[:, parents[:, 0]], pop[:, parents[:, 0]])

            cr_mask = rng.random_sample((n_units, self.pop_size)) < self.crossover_rate
            cand_pop = np.where(cr_mask, mutant, pop)

            cost_cand, score_cand = _evaluate_srocket_population(
                Xt, y_ind, coef_t, intercept, cand_pop, group_size, w_buf
            )
            improved = cost_cand < cost_pop
            if np.any(improved):
                pop[:, improved] = cand_pop[:, improved]
                cost_pop[improved] = cost_cand[improved]
                score_pop[improved] = score_cand[improved]

        best_idx = int(np.argmin(cost_pop))
        self.best_cost_ = float(cost_pop[best_idx])
        kept = pop[:, best_idx] > 0.5
        if not np.any(kept):
            coef_group = self.pre_classifier_.coef_.T.reshape(n_units, -1)
            top_unit = int(np.argmax(np.linalg.norm(coef_group, axis=1)))
            kept[top_unit] = True

        self.kernel_mask_ = kept
        self.n_selected_kernels_ = int(np.sum(kept))

        if transform_type in ("rocket", "rocket-ppvmax"):
            self._pruned_kernels = _prune_rocket_kernels(full_kernels, kept)

        feat_mask = np.repeat(kept, 2) if group_size == 2 else kept

        # Step 3: Post-training on selected kernels
        self.classifier_ = _L2NormalisedRidgeClassifierCV().fit(Xt[:, feat_mask], y_ind)
        return self

    @staticmethod
    def _pad_minirocket(X):
        if X.shape[2] < 9:
            X = np.pad(X, ((0, 0), (0, 0), (0, 9 - X.shape[2])))
        return X

    def _transform_selected(self, X):
        X_clean = np.nan_to_num(X, nan=0.0)
        transform_type = self.rocket_transform.lower()
        if transform_type in ("rocket", "rocket-ppvmax"):
            Xt = _apply_rocket_kernels(
                X_clean, self._pruned_kernels, self._rocket_normalise, self._n_jobs
            ).astype(np.float64)
            if transform_type == "rocket":
                Xt = Xt[:, ::2]
            return Xt

        X_mr = self._pad_minirocket(X_clean)
        Xt = self._minirocket.transform(X_mr).astype(np.float64)
        return Xt[:, self.kernel_mask_]

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        Xt = self._transform_selected(X)
        preds = self.classifier_.predict(Xt)
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
        return {"n_kernels": 20, "n_epochs": 10, "pop_size": 4}


def _evaluate_srocket_population(Xt, y_ind, coef_t, intercept, pop, group_size, w_buf):
    """Evaluate S-ROCKET evolutionary cost across all population members."""
    n_cases, n_features = Xt.shape
    n_units, pop_size = pop.shape
    pop_feat = np.repeat(pop, 2, axis=0) if group_size == 2 else pop
    n_out = coef_t.shape[1]

    if n_out == 1:
        w_view = w_buf[:, 0, :]
        np.multiply(coef_t[:, 0:1], pop_feat, out=w_view)
        scores = Xt @ w_view + intercept[0]
        preds = (scores > 0.0).astype(int)
    else:
        np.multiply(coef_t[:, :, None], pop_feat[:, None, :], out=w_buf)
        w_view = w_buf.reshape(n_features, n_out * pop_size)
        scores = (Xt @ w_view).reshape(n_cases, n_out, pop_size) + intercept[
            None, :, None
        ]
        preds = np.argmax(scores, axis=1)

    acc = np.mean(preds == y_ind[:, None], axis=0)
    dcost = np.sum(pop, axis=0) / n_units
    cost = 1.0 - (acc - dcost)
    return cost, acc
