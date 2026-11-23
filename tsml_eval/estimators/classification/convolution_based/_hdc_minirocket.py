"""HDC-MiniRocket classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HDCMiniRocketClassifier"]

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.transformations.collection.convolution_based._minirocket import _static_fit
from numba import njit, prange

from tsml_eval.estimators.classification.convolution_based._hdc_utils import (
    _BaseHDCRocketClassifier,
)


class HDCMiniRocketClassifier(_BaseHDCRocketClassifier):
    """MiniRocket with explicit time encoding using hyperdimensional computing.

    HDC-MiniRocket [1]_ [2]_ adds the time of each convolution output to the
    MiniRocket features. Each time point is encoded as a hyperdimensional vector
    using sinusoidal fractional power encoding, with the similarity of the vectors of
    two time points decreasing with their distance at a rate set by ``beta``. In place
    of the proportion of positive values (PPV), feature ``i`` is the sum, over the
    time points where the convolution output exceeds the bias, of dimension ``i`` of
    the time point vectors. With ``beta=0`` all time points share one encoding, and
    the features are the MiniRocket features up to a scaling of each feature.

    MiniRocket's kernels, dilations and biases are fitted as in MiniRocket. Each
    channel is z-normalised using the training data, and the features are scaled to
    ``[-1, 1]`` and standardised before a ``RidgeClassifierCV`` is fitted. Unless
    ``beta`` is set, it is selected from ``2 ** np.linspace(0, 3, 7) - 1`` using the
    leave-one-out score of the ridge classifier on the training data.

    This follows the "hdc_auto" variant of the reference implementation [3]_.

    Parameters
    ----------
    n_kernels : int, default=9996
        Number of MiniRocket features, also the dimension of the time encodings.
        The number of features is rounded down to a multiple of 84, with a minimum of
        84.
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per MiniRocket kernel.
    beta : float or None, default=None
        Bandwidth of the time encoding. If None, ``beta`` is selected from the
        reference grid using the training data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the transform. ``-1`` means using
        all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for the MiniRocket parameters and the
        random phases of the time encoding.

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
    HDCMultiRocketHydraClassifier

    Notes
    -----
    As in the reference implementation, the features sum the time encodings rather
    than averaging them over the series.

    References
    ----------
    .. [1] Schlegel, Kenny, Peer Neubert, and Peter Protzel. "HDC-MiniROCKET: Explicit
       time encoding in time series classification with hyperdimensional computing."
       International Joint Conference on Neural Networks, 2022.
    .. [2] Schlegel, Kenny, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter
       Protzel, and Peer Neubert. "Structured temporal representation in time series
       classification with ROCKETs and hyperdimensional computing." Data Mining and
       Knowledge Discovery, 2025.
    .. [3] https://github.com/TUC-ProAut/HDC-ROCKETS

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     HDCMiniRocketClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = HDCMiniRocketClassifier(n_kernels=84, beta=1, random_state=0)
    >>> clf.fit(X, y)
    HDCMiniRocketClassifier(beta=1, n_kernels=84, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _hdc_dim_param = "n_kernels"

    def __init__(
        self,
        n_kernels=9996,
        max_dilations_per_kernel=32,
        beta=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel

        super().__init__(beta=beta, n_jobs=n_jobs, random_state=random_state)

    def _fit_kernels(self, X):
        self._parameters = _static_fit(
            X, max(self.n_kernels, 84), self.max_dilations_per_kernel, self._seed
        )

    def _n_hdc_columns(self):
        return 84 * int(np.sum(self._parameters[3]))

    def _transform_hdc(self, X, poses):
        return _hdc_minirocket_transform(
            X, self._parameters, MiniRocket._indices, poses
        ).astype(np.float64)

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
        return {"n_kernels": 84, "beta": 1}


@njit(fastmath=True, parallel=True, cache=True)
def _hdc_minirocket_transform(X, parameters, indices, poses):
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, n_features), dtype=np.float32)

    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        comb = 0
        n_channels_start = 0

        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features_this_dilation = n_features_per_dilation[j]

            C_alpha = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_channels, n_timepoints), dtype=np.float32)
            C_gamma[4] = G

            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            for k in range(n_kernels):
                n_channels_this_combo = n_channels_per_combination[comb]
                n_channels_end = n_channels_start + n_channels_this_combo
                channels_this_combo = channel_indices[n_channels_start:n_channels_end]
                index_0, index_1, index_2 = indices[k]

                C = (
                    C_alpha[channels_this_combo]
                    + C_gamma[index_0][channels_this_combo]
                    + C_gamma[index_1][channels_this_combo]
                    + C_gamma[index_2][channels_this_combo]
                )
                C = np.sum(C, axis=0)

                # alternate kernels use only the outputs without padding
                if (_padding0 + k) % 2 == 0:
                    lo = 0
                    hi = n_timepoints
                else:
                    lo = padding
                    hi = n_timepoints - padding

                for f in range(f_start, f_start + n_features_this_dilation):
                    bias = biases[f]
                    total = 0.0
                    for t in range(lo, hi):
                        if C[t] > bias:
                            total += poses[f, t]
                    features[i, f] = total

                f_start += n_features_this_dilation
                comb += 1
                n_channels_start = n_channels_end

    return features
