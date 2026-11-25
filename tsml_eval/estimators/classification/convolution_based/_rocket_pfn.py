"""RocketPFN classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RocketPFNClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation import check_n_jobs
from numba import config
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._tabpfn_utils import (
    _build_tabpfn_classifier,
    _fit_tabpfn_classifier,
    _predict_proba_tabpfn,
)


class RocketPFNClassifier(BaseClassifier):
    """ROCKET features with a grouped TabPFN ensemble (RocketPFN).

    RocketPFN [1]_ pairs the random dilated convolutional feature transform of ROCKET
    [2]_ with the tabular prior-data fitted network TabPFN v2.5 [3]_. Each input time
    series is z-normalised along the time axis, and ``n_groups`` independent groups of
    ``n_kernels`` ROCKET kernels are generated. Each group produces ``2 * n_kernels``
    features (proportion of positive values and global max pooling) and fits its own
    ``TabPFNClassifier`` by in-context forward passes, keeping each tabular head
    within the 2,000-feature pretraining regime of TabPFN v2.5 while retaining the
    total kernel budget across groups.

    At prediction time, class probability distributions from all ``n_groups`` TabPFN
    classifiers are averaged with equal weight.

    Parameters
    ----------
    n_groups : int, default=10
        Number of independent ROCKET feature groups and TabPFN classifiers (``G`` in
        [1]_).
    n_kernels : int, default=1000
        Number of random convolutional kernels per group (``k`` in [1]_, giving
        ``2 * n_kernels`` features per group).
    n_estimators : int, default=8
        Number of internal TabPFN ensemble configurations per group.
    model_version : str, default="v2.5"
        TabPFN model version passed to ``TabPFNClassifier.create_default_for_version``
        when ``model_path`` is None.
    model_path : str, Path, ModelSpecs or None, default=None
        Optional explicit checkpoint path or in-memory ``ModelSpecs`` passed to
        ``TabPFNClassifier``. Overrides ``model_version`` when set.
    device : str, default="auto"
        Device passed to ``TabPFNClassifier`` (for example ``"auto"``, ``"cpu"``, or
        ``"cuda"``).
    n_jobs : int, default=1
        Number of threads used for ROCKET kernel transforms and TabPFN preprocessing.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling the per-group ROCKET kernels and
        TabPFN ensemble seeds.

    Attributes
    ----------
    transformers_ : list of Rocket
        The fitted ROCKET transformers, one per group.
    classifiers_ : list of TabPFNClassifier
        The fitted TabPFN classifiers, one per group.

    See Also
    --------
    MASHTClassifier

    Notes
    -----
    TabPFN v2.5 supports at most 10 classes natively; datasets with more than 10
    classes raise an error from ``TabPFNClassifier``, matching the scope of [1]_.
    Series shorter than 11 time points are zero padded to length 11 after instance
    z-normalisation.

    A GPU is recommended for TabPFN inference (requires ``tabpfn>=8.1.0`` and
    ``torch>=2.5``); ``ignore_pretraining_limits=True`` is set so larger datasets can
    still run on CPU, though CPU execution can be very slow as sample size grows. When
    running many jobs at once or without network access, it is advisable to
    pre-download the TabPFN checkpoint into a shared cache directory (via
    ``TABPFN_MODEL_CACHE_DIR`` or ``model_path``) rather than downloading on first
    ``fit``.

    References
    ----------
    .. [1] O'Rourke, John, Ana Trisovic, and Dimitris Bertsimas. "RocketPFN: Accurate
       Time Series Classification via In-Context Learning." arXiv preprint
       arXiv:2606.21786, 2026.
    .. [2] Dempster, Angus, Francois Petitjean, and Geoffrey I. Webb. "ROCKET:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." *Data Mining and Knowledge Discovery*, 34(5),
       1454-1495, 2020.
    .. [3] Grinsztajn, Leo, et al. "TabPFN-2.5: Advancing the State of the Art in
       Tabular Foundation Models." arXiv preprint arXiv:2511.08667, 2025.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     RocketPFNClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = RocketPFNClassifier(
    ...     n_groups=2, n_kernels=50, random_state=0
    ... )  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    RocketPFNClassifier(n_groups=2, n_kernels=50, random_state=0)
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
        n_groups=10,
        n_kernels=1000,
        n_estimators=8,
        model_version="v2.5",
        model_path=None,
        device="auto",
        n_jobs=1,
        random_state=None,
    ):
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.n_estimators = n_estimators
        self.model_version = model_version
        self.model_path = model_path
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_groups < 1:
            raise ValueError(f"n_groups must be at least 1, got {self.n_groups}.")
        if self.n_kernels < 1:
            raise ValueError(f"n_kernels must be at least 1, got {self.n_kernels}.")
        if isinstance(self.n_estimators, (int, np.integer)) and self.n_estimators < 1:
            raise ValueError(
                f"n_estimators must be at least 1, got {self.n_estimators}."
            )

        import torch

        self._n_jobs = check_n_jobs(self.n_jobs)
        self._numba_jobs = min(self._n_jobs, config.NUMBA_NUM_THREADS)

        if self.n_classes_ == 1:
            self.transformers_ = []
            self.classifiers_ = []
            return self

        X_prep = self._preprocess(X)
        y_enc = np.array([self._class_dictionary[label] for label in y], dtype=int)
        rng = check_random_state(self.random_state)

        self.transformers_ = []
        self.classifiers_ = []

        prev_torch_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._n_jobs)
            for _ in range(self.n_groups):
                rocket_seed = int(rng.randint(np.iinfo(np.int32).max))
                pfn_seed = int(rng.randint(np.iinfo(np.int32).max))

                trf = Rocket(
                    n_kernels=self.n_kernels,
                    normalise=False,
                    n_jobs=self._numba_jobs,
                    random_state=rocket_seed,
                ).fit(X_prep)
                Xt = trf.transform(X_prep)

                clf = _build_tabpfn_classifier(
                    model_version=self.model_version,
                    model_path=self.model_path,
                    n_estimators=self.n_estimators,
                    device=self.device,
                    n_preprocessing_jobs=self._n_jobs,
                    ignore_pretraining_limits=True,
                    random_state=pfn_seed,
                )
                _fit_tabpfn_classifier(clf, Xt, y_enc)

                self.transformers_.append(trf)
                self.classifiers_.append(clf)
        finally:
            torch.set_num_threads(prev_torch_threads)

        return self

    def _predict_proba(self, X):
        n_cases = len(X)
        if self.n_classes_ == 1:
            return np.ones((n_cases, 1), dtype=np.float32)

        import torch

        X_prep = self._preprocess(X)

        prev_torch_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._n_jobs)
            probas = [
                _predict_proba_tabpfn(clf, trf.transform(X_prep))
                for trf, clf in zip(self.transformers_, self.classifiers_)
            ]
        finally:
            torch.set_num_threads(prev_torch_threads)

        return np.mean(probas, axis=0)

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    @staticmethod
    def _preprocess(X):
        X = Normalizer().fit_transform(X)
        if X.shape[2] < 11:
            X = np.pad(X, ((0, 0), (0, 0), (0, 11 - X.shape[2])))
        return np.asarray(X, dtype=np.float32)

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
            "n_groups": 2,
            "n_kernels": 10,
            "n_estimators": 2,
            "model_path": "_tiny",
        }
