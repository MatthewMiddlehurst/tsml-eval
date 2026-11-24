"""HDC-MultiRocket-Hydra classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HDCMultiRocketHydraClassifier"]

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsml_eval.estimators.classification.convolution_based._hdc_hydra import (
    _fit_hdc_hydra,
    _hdc_hydra_transform,
    _n_hdc_hydra_features,
)
from tsml_eval.estimators.classification.convolution_based._hdc_multirocket import (
    _fit_multirocket,
    _transform_multirocket,
)
from tsml_eval.estimators.classification.convolution_based._hdc_utils import (
    _BaseHDCRocketClassifier,
    _SparseScaler,
)


class HDCMultiRocketHydraClassifier(_BaseHDCRocketClassifier):
    """MultiRocket-Hydra with explicit time encoding using hyperdimensional computing.

    HDC-MultiRocket-Hydra [1]_ concatenates the HDC-MultiRocket and HDC-Hydra
    features, with both transforms using the same time encodings. Each time point is
    encoded as a hyperdimensional vector using sinusoidal fractional power encoding,
    with the similarity of the vectors of two time points decreasing with their
    distance at a rate set by ``beta``. See ``HDCMultiRocketClassifier`` and
    ``HDCHydraClassifier`` for the two transforms.

    Each channel is z-normalised using the training data. The Hydra features are
    scaled with the Hydra sparse scaler, then all features are scaled to ``[-1, 1]``
    and standardised before a ``RidgeClassifierCV`` is fitted. Unless ``beta`` is
    set, it is selected from ``2 ** np.linspace(0, 3, 7) - 1`` using the
    leave-one-out score of the ridge classifier on the training data.

    This follows the "hdc_auto" variant of the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=6250
        Number of MiniRocket features for each of the series and its first order
        difference in the MultiRocket transform. Rounded down to a multiple of 84,
        with a minimum of 84.
    hydra_n_kernels : int, default=8
        Number of kernels per Hydra group.
    hydra_n_groups : int, default=64
        Number of Hydra groups per dilation.
    max_num_channels : int, default=8
        Maximum number of channels summed by each Hydra group.
    hdc_dim : int, default=50000
        Dimension of the time encodings. Must be at least the number of features of
        each transform.
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per MiniRocket kernel.
    beta : float or None, default=None
        Bandwidth of the time encoding. If None, ``beta`` is selected from the
        reference grid using the training data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transforms. ``-1`` means using
        all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for the kernels of both transforms and
        the random phases of the time encoding.

    Attributes
    ----------
    beta_ : float
        The bandwidth of the time encoding used by the final model.
    beta_scores_ : np.ndarray
        The leave-one-out score of the ridge classifier for each candidate ``beta``.
    classifier_ : RidgeClassifierCV
        The ridge classifier fitted on the transformed training data.

    See Also
    --------
    HDCMultiRocketClassifier
    HDCHydraClassifier

    Notes
    -----
    The kernels are fitted with the aeon ``MiniRocket`` and ``HydraTransformer``
    transforms, both seeded with ``random_state``. The reference seeds its Hydra
    kernels with 42 regardless of the seed of the run. See
    ``HDCMultiRocketClassifier`` and ``HDCHydraClassifier`` for other differences
    to the reference.

    References
    ----------
    .. [1] Schlegel, Kenny, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter
       Protzel, and Peer Neubert. "Structured temporal representation in time series
       classification with ROCKETs and hyperdimensional computing." Data Mining and
       Knowledge Discovery, 2025.
    .. [2] https://github.com/TUC-ProAut/HDC-ROCKETS

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     HDCMultiRocketHydraClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = HDCMultiRocketHydraClassifier(
    ...     n_kernels=84, hydra_n_groups=8, hdc_dim=500, beta=1, random_state=0
    ... )  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    HDCMultiRocketHydraClassifier(beta=1, hdc_dim=500, hydra_n_groups=8, n_kernels=84,
                                  random_state=0)
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "torch",
    }

    _min_length = 10
    _hdc_dim_param = "hdc_dim"

    def __init__(
        self,
        n_kernels=6250,
        hydra_n_kernels=8,
        hydra_n_groups=64,
        max_num_channels=8,
        hdc_dim=50000,
        max_dilations_per_kernel=32,
        beta=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.hydra_n_kernels = hydra_n_kernels
        self.hydra_n_groups = hydra_n_groups
        self.max_num_channels = max_num_channels
        self.hdc_dim = hdc_dim
        self.max_dilations_per_kernel = max_dilations_per_kernel

        super().__init__(beta=beta, n_jobs=n_jobs, random_state=random_state)

    def _fit_kernels(self, X):
        self._parameters, self._parameters1 = _fit_multirocket(
            X,
            self.n_kernels,
            self.max_dilations_per_kernel,
            self._seed,
            self._n_jobs,
        )
        self._hydra = _fit_hdc_hydra(
            X,
            self.hydra_n_kernels,
            self.hydra_n_groups,
            self.max_num_channels,
            self._seed,
            self._n_jobs,
        )
        # MultiRocket has three features for each kernel, for both inputs
        self._n_multirocket_features = (
            3 * 84 * int(np.sum(self._parameters[3]) + np.sum(self._parameters1[3]))
        )

    def _n_hdc_columns(self):
        return max(
            84 * int(np.sum(self._parameters[3])), _n_hdc_hydra_features(self._hydra)
        )

    def _transform_hdc(self, X, poses):
        return np.hstack(
            (
                _transform_multirocket(X, self._parameters, self._parameters1, poses),
                _hdc_hydra_transform(X, self._hydra, poses),
            )
        )

    def _make_scaler(self):
        return make_pipeline(
            _SparseScaler(start=self._n_multirocket_features),
            MinMaxScaler(feature_range=(-1, 1), clip=True),
            StandardScaler(),
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {"n_kernels": 84, "hydra_n_groups": 8, "hdc_dim": 500, "beta": 1}
