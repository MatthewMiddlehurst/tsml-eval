"""MASHT classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MASHTClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import (
    HydraTransformer,
    MultiRocket,
)
from aeon.utils.validation import check_n_jobs
from numba import config
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._tabpfn_utils import (
    _build_tabpfn_classifier,
    _fit_tabpfn_classifier,
    _predict_proba_tabpfn,
)


def _adaptive_feature_budget(n_samples):
    """Return the MASHT adaptive feature budget from sample count."""
    if n_samples < 1_000:
        return 10_000
    if n_samples < 100_000:
        return 2_000
    return 200


class MASHTClassifier(BaseClassifier):
    """MultiRocket and Hydra features with TabPFN (MASHT).

    MASHT [1]_ extracts concatenated Hydra [3]_ and MultiRocket [4]_ convolutional
    features under an adaptive sample-size feature budget and classifies them with
    TabPFN-3. Unless ``max_features`` is given explicitly, the total feature budget
    is set from the number of training instances ``n_cases``: 10,000 features for
    ``n_cases < 1,000``, 2,000 for ``1,000 <= n_cases < 100,000``, and 200 for
    ``n_cases >= 100,000``. Half of the budget is allocated to Hydra (sparse-scaled)
    and half to MultiRocket (unnormalised).

    The concatenated ``float32`` features are passed to ``TabPFNClassifier`` with
    ``fit_mode="low_memory"`` and automatic estimator scaling so that wide feature
    matrices remain covered across TabPFN's feature-subsampled ensemble members.

    This follows the reference implementation [2]_, using aeon's ``HydraTransformer``
    and ``MultiRocket`` transforms, with the reference bugs noted below fixed.

    Parameters
    ----------
    max_features : int or None, default=None
        Total feature budget split equally between Hydra and MultiRocket. If None,
        uses the adaptive schedule ``{10_000, 2_000, 200}`` based on ``n_cases``.
    n_estimators : int or "auto", default="auto"
        Number of TabPFN ensemble members. ``"auto"`` starts at 8 and scales up to
        32 so every extracted feature is covered by at least one member, matching
        ``n_estimators=8, auto_scale_n_estimators=True`` in [2]_.
    model_version : str, default="v3"
        TabPFN model version passed to ``TabPFNClassifier.create_default_for_version``
        when ``model_path`` is None.
    model_path : str, Path, ModelSpecs or None, default=None
        Optional explicit checkpoint path or in-memory ``ModelSpecs`` passed to
        ``TabPFNClassifier``. Overrides ``model_version`` when set.
    device : str, default="auto"
        Device passed to ``TabPFNClassifier`` (for example ``"auto"``, ``"cpu"``, or
        ``"cuda"``).
    n_jobs : int, default=1
        Number of threads used for Hydra, MultiRocket, and TabPFN preprocessing.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for Hydra, MultiRocket, and TabPFN.

    Attributes
    ----------
    feature_budget_ : int
        Resolved total feature budget.
    n_hydra_groups_ : int
        Number of Hydra groups per dilation.
    n_transformed_features_ : int
        Total number of concatenated Hydra and MultiRocket features.
    classifier_ : TabPFNClassifier or None
        The fitted TabPFN classifier (``None`` when ``n_classes_ == 1``).

    See Also
    --------
    RocketPFNClassifier

    Notes
    -----
    A GPU is recommended for TabPFN inference (requires ``tabpfn>=8.1.0`` and
    ``torch>=2.5``); ``ignore_pretraining_limits=True`` is set so larger datasets can
    still run on CPU, though CPU execution can be very slow as sample size grows. When
    running many jobs at once or without network access, it is advisable to
    pre-download the TabPFN checkpoint into a shared cache directory (via
    ``TABPFN_MODEL_CACHE_DIR`` or ``model_path``) rather than downloading on first
    ``fit``.

    Several clear bugs in the reference implementation [2]_ are fixed:

    1. **Hydra dilation count in group budgeting**: ``Masht._hydra_groups`` in [2]_
       computes ``num_dilations = np.ceil(np.log2((L - 1) / 8))``, whereas both
       Section 3.3 of [1]_ and ``Hydra`` itself use ``int(np.log2((L - 1) / 8)) + 1``
       dilations. Whenever ``(L - 1) / 8`` is an exact power of two (such as
       ``L = 65`` on UEA ``ERing``, or ``L in {17, 33, 65, 129, 257, 513, 1025,
       2049}``), ``ceil`` undercounts ``num_dilations`` by one, inflating the Hydra
       group count and exceeding ``max_features`` (for example producing 11,360
       features instead of 9,696 on ``ERing`` at ``max_features=10000``). Using
       ``int(np.log2((L - 1) / 8)) + 1`` keeps the feature count within budget and
       changed mean accuracy over 5 seeds on ``ERing`` by -0.07 percentage points at
       budget 10,000 (0.9822 to 0.9815) and -0.44 percentage points at budget 2,000
       (0.9637 to 0.9593).
    2. **Small feature budgets**: Whenever ``max_features < 1344`` (including the
       ``n_cases >= 100,000`` branch of [2]_ where ``feature_budget = 200`` and
       ``component_budget = 100``), ``num_kernels = int((component_budget / 2) / 4)
       < 84`` raises ``ZeroDivisionError`` inside ``MultiRocket._fit_dilations``, and
       ``_hydra_groups`` raises ``ValueError`` when ``groups < 1`` on long series.
       Here ``n_hydra_groups_`` is clamped to at least 1 and MultiRocket's kernel
       argument is clamped to at least 84.
    3. **Short series and RNG seeding**: Series shorter than 10 time points (rejected
       in [2]_) are zero padded to length 10. All three components (Hydra,
       MultiRocket, and TabPFN) are seeded from ``random_state``, whereas [2]_ leaves
       Hydra and MultiRocket unseeded. As in ``Masht()``'s default estimator mode,
       the adaptive budget uses the training set size ``n_cases`` (the paper's
       benchmark script passed ``n_train + n_test``).

    References
    ----------
    .. [1] Cuppers, Joscha, and Jilles Vreeken. "In-Context Time Series Classification
       with Random Convolutional Features." arXiv preprint arXiv:2607.19234, 2026.
    .. [2] https://github.com/joschac/masht
    .. [3] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb. "Hydra:
       competing convolutional kernels for fast and accurate time series
       classification." *Data Mining and Knowledge Discovery*, 37(5), 1779-1805, 2023.
    .. [4] Tan, Chang Wei, Angus Dempster, Christoph Bergmeir, and Geoffrey I. Webb.
       "MultiRocket: multiple pooling operators and transformations for fast and
       effective time series classification." *Data Mining and Knowledge Discovery*,
       36(5), 1623-1646, 2022.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     MASHTClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = MASHTClassifier(
    ...     max_features=400, n_estimators=2, random_state=0
    ... )  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    MASHTClassifier(max_features=400, n_estimators=2, random_state=0)
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": ["tabpfn>=8.1.0", "torch>=2.5"],
    }

    def __init__(
        self,
        max_features=None,
        n_estimators="auto",
        model_version="v3",
        model_path=None,
        device="auto",
        n_jobs=1,
        random_state=None,
    ):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.model_version = model_version
        self.model_path = model_path
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.max_features is not None and self.max_features < 1:
            raise ValueError(
                f"max_features must be at least 1, got {self.max_features}."
            )
        if isinstance(self.n_estimators, (int, np.integer)) and self.n_estimators < 1:
            raise ValueError(
                f"n_estimators must be at least 1, got {self.n_estimators}."
            )

        import torch

        self._n_jobs = check_n_jobs(self.n_jobs)
        self._numba_jobs = min(self._n_jobs, config.NUMBA_NUM_THREADS)

        if self.random_state is None or isinstance(
            self.random_state, np.random.RandomState
        ):
            rng = check_random_state(self.random_state)
            seed = int(rng.randint(np.iinfo(np.int32).max))
        else:
            seed = int(self.random_state)

        X_prep = self._preprocess(X)
        self.feature_budget_ = (
            int(self.max_features)
            if self.max_features is not None
            else _adaptive_feature_budget(len(X_prep))
        )
        component_budget = self.feature_budget_ // 2

        num_dilations = int(np.log2((X_prep.shape[2] - 1) / 8)) + 1
        self.n_hydra_groups_ = max(1, component_budget // (16 * num_dilations))
        n_mr_kernels = max(84, int((component_budget / 2) / 4))

        prev_torch_threads = torch.get_num_threads()
        try:
            self._hydra_transformer = HydraTransformer(
                n_kernels=8,
                n_groups=self.n_hydra_groups_,
                n_jobs=self._n_jobs,
                random_state=seed,
            ).fit(X_prep)
            X_tensor = torch.from_numpy(np.ascontiguousarray(X_prep, dtype=np.float32))
            with torch.no_grad():
                hydra_raw = self._hydra_transformer._hydra.batch(X_tensor)
            self._hydra_scaler = _SparseScaler()
            hydra_features = self._hydra_scaler.fit_transform(hydra_raw).numpy()

            self._multirocket = MultiRocket(
                n_kernels=n_mr_kernels,
                normalise=False,
                n_jobs=self._numba_jobs,
                random_state=seed,
            ).fit(X_prep)
            mr_features = self._multirocket.transform(X_prep)

            features = np.concatenate((hydra_features, mr_features), axis=1)
            self.n_transformed_features_ = int(features.shape[1])

            if self.n_classes_ == 1:
                self.classifier_ = None
                return self

            y_enc = np.array([self._class_dictionary[label] for label in y], dtype=int)
            torch.set_num_threads(self._n_jobs)
            self.classifier_ = _build_tabpfn_classifier(
                model_version=self.model_version,
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                fit_mode="low_memory",
                device=self.device,
                inference_precision="auto",
                n_preprocessing_jobs=self._n_jobs,
                tuning_config=None,
                show_progress_bar=False,
                ignore_pretraining_limits=True,
                random_state=seed,
            )
            _fit_tabpfn_classifier(self.classifier_, features, y_enc)
        finally:
            torch.set_num_threads(prev_torch_threads)

        return self

    def _transform(self, X):
        import torch

        X_prep = self._preprocess(X)
        prev_torch_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._n_jobs)
            X_tensor = torch.from_numpy(np.ascontiguousarray(X_prep, dtype=np.float32))
            with torch.no_grad():
                hydra_raw = self._hydra_transformer._hydra.batch(X_tensor)
            hydra_features = self._hydra_scaler.transform(hydra_raw).numpy()
        finally:
            torch.set_num_threads(prev_torch_threads)

        mr_features = self._multirocket.transform(X_prep)
        return np.concatenate((hydra_features, mr_features), axis=1)

    def _predict_proba(self, X):
        if self.n_classes_ == 1:
            return np.ones((len(X), 1), dtype=np.float32)

        import torch

        Xt = self._transform(X)
        prev_torch_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._n_jobs)
            return _predict_proba_tabpfn(self.classifier_, Xt)
        finally:
            torch.set_num_threads(prev_torch_threads)

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    @staticmethod
    def _preprocess(X):
        if X.shape[2] < 10:
            X = np.pad(X, ((0, 0), (0, 0), (0, 10 - X.shape[2])))
        return X

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
        return {
            "max_features": 400,
            "n_estimators": 2,
            "model_path": "_tiny",
        }
