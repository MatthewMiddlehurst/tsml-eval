"""HDC-Hydra classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HDCHydraClassifier"]

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsml_eval.estimators.classification.convolution_based._hdc_utils import (
    _BaseHDCRocketClassifier,
    _SparseScaler,
)


class HDCHydraClassifier(_BaseHDCRocketClassifier):
    """Hydra with explicit time encoding using hyperdimensional computing.

    HDC-Hydra [1]_ adds the time of each convolution output to the Hydra features.
    As in Hydra, groups of random kernels are applied to the series and to its first
    order difference, and at each time point the kernel with the largest output in a
    group gets the output value and the kernel with the smallest output gets a count
    of one. Each time point is encoded as a hyperdimensional vector using sinusoidal
    fractional power encoding, with the similarity of the vectors of two time points
    decreasing with their distance at a rate set by ``beta``. Feature ``i`` is the
    sum over time of the value or count multiplied by dimension ``i`` of the time
    point vector, in place of the sum of the values or counts used by Hydra.

    The kernels and channels of each group are drawn as in the reference
    implementation, which uses its multivariate version of Hydra for all data. Each
    channel is z-normalised using the training data, and the features are scaled with
    the Hydra sparse scaler, scaled to ``[-1, 1]`` and standardised before a
    ``RidgeClassifierCV`` is fitted. Unless ``beta`` is set, it is selected from
    ``2 ** np.linspace(0, 3, 7) - 1`` using the leave-one-out score of the ridge
    classifier on the training data.

    This follows the "hdc_auto" variant of the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of groups per dilation. Half of the groups are applied to the first
        order difference if greater than one.
    max_num_channels : int, default=8
        Maximum number of channels summed by each group.
    hdc_dim : int, default=20000
        Dimension of the time encodings. Must be at least the number of features,
        ``2 * n_kernels * n_groups`` for each dilation.
    beta : float or None, default=None
        Bandwidth of the time encoding. If None, ``beta`` is selected from the
        reference grid using the training data.
    n_jobs : int, default=1
        The number of threads used by torch. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for the Hydra kernels and the random
        phases of the time encoding.

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
    HDCMiniRocketClassifier
    HDCMultiRocketHydraClassifier

    Notes
    -----
    The reference implementation always seeds the Hydra kernels and time encoding
    with 42, ``random_state=42`` gives the same kernels and encodings. As in the
    reference, every second time point is used for series longer than 10,000. The
    reference stores the transform in half precision, here it is kept in single
    precision.

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
    ...     HDCHydraClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = HDCHydraClassifier(
    ...     n_groups=8, hdc_dim=500, beta=1, random_state=0
    ... )  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    HDCHydraClassifier(beta=1, hdc_dim=500, n_groups=8, random_state=0)
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "torch",
    }

    _hdc_dim_param = "hdc_dim"

    def __init__(
        self,
        n_kernels=8,
        n_groups=64,
        max_num_channels=8,
        hdc_dim=20000,
        beta=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_num_channels = max_num_channels
        self.hdc_dim = hdc_dim

        super().__init__(beta=beta, n_jobs=n_jobs, random_state=random_state)

    def _fit_kernels(self, X):
        self._hydra = _fit_hdc_hydra(
            X.shape[2],
            X.shape[1],
            self.n_kernels,
            self.n_groups,
            self.max_num_channels,
            self._seed,
            self._n_jobs,
        )

    def _n_hdc_columns(self):
        return _n_hdc_hydra_features(self._hydra)

    def _transform_hdc(self, X, poses):
        return _hdc_hydra_transform(X, self._hydra, poses)

    def _make_scaler(self):
        return make_pipeline(
            _SparseScaler(),
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
        return {"n_groups": 8, "hdc_dim": 500, "beta": 1}


def _fit_hdc_hydra(n_timepoints, n_channels, k, g, max_num_channels, seed, n_jobs):
    """Draw the Hydra kernels and channels as in the HDC-Hydra reference."""
    import torch

    torch.set_num_threads(n_jobs)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    dilations = 2 ** np.arange(int(np.log2((n_timepoints - 1) / (9 - 1))) + 1)
    divisor = 2 if g > 1 else 1
    h = g // divisor

    # all kernels are drawn before the channels, as in the reference
    weights = []
    for _ in dilations:
        w = torch.randn(divisor, k * h, 1, 9, generator=generator, dtype=torch.float32)
        w -= w.mean(-1, keepdim=True)
        w /= w.abs().sum(-1, keepdim=True)
        weights.append(w)

    n_channels_per = int(np.clip(n_channels // 2, 2, max_num_channels))
    channels = [
        torch.randint(0, n_channels, (divisor, h, n_channels_per), generator=generator)
        for _ in dilations
    ]

    return {
        "dilations": dilations,
        "weights": weights,
        "channels": channels,
        "k": k,
        "h": h,
        "divisor": divisor,
    }


def _n_hdc_hydra_features(hydra):
    """Return the number of HDC-Hydra features, two for each kernel and input."""
    return len(hydra["dilations"]) * hydra["divisor"] * 2 * hydra["h"] * hydra["k"]


def _hdc_hydra_transform(X, hydra, poses):
    """Apply the HDC-Hydra transform, with feature i bound to row i of poses."""
    import torch
    import torch.nn.functional as F

    # the reference uses every second time point of long series
    if X.shape[2] > 10000:
        X = X[:, :, ::2]
        poses = poses[:, ::2]

    k = hydra["k"]
    h = hydra["h"]
    n_cases = X.shape[0]
    poses = torch.from_numpy(np.ascontiguousarray(poses))
    features = np.zeros((n_cases, _n_hdc_hydra_features(hydra)), dtype=np.float32)
    batch_size = max(1, int(2e7 // (h * k * X.shape[2])))

    with torch.no_grad():
        for start in range(0, n_cases, batch_size):
            Xb = torch.from_numpy(np.ascontiguousarray(X[start : start + batch_size]))
            inputs = [Xb, torch.diff(Xb)] if hydra["divisor"] > 1 else [Xb]
            n = Xb.shape[0]

            col = 0
            for i, d in enumerate(hydra["dilations"]):
                d = int(d)
                for diff_index, inp in enumerate(inputs):
                    Z = F.conv1d(
                        inp[:, hydra["channels"][i][diff_index]].sum(2),
                        hydra["weights"][i][diff_index],
                        dilation=d,
                        padding=(8 * d) // 2,
                        groups=h,
                    ).view(n, h, k, -1)
                    length = Z.shape[-1]

                    max_values, max_indices = Z.max(2)
                    _, min_indices = Z.min(2)
                    min_hard = torch.zeros_like(Z).scatter_(
                        2, min_indices.unsqueeze(2), 1.0
                    )
                    max_soft = torch.zeros_like(Z).scatter_(
                        2, max_indices.unsqueeze(2), max_values.unsqueeze(2)
                    )

                    # bind each output to the time encoding of its time point
                    mask = torch.cat(
                        (min_hard.view(n, h * k, -1), max_soft.view(n, h * k, -1)), 1
                    )
                    n_cols = mask.shape[1]
                    pose = poses[col : col + n_cols, :length]
                    features[start : start + n, col : col + n_cols] = (
                        (mask * pose[None]).sum(-1).numpy()
                    )
                    col += n_cols

    return features.astype(np.float64)
