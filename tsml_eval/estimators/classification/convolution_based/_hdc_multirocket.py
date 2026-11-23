"""HDC-MultiRocket classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HDCMultiRocketClassifier"]

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.transformations.collection.convolution_based._minirocket import _static_fit
from numba import njit, prange

from tsml_eval.estimators.classification.convolution_based._hdc_utils import (
    _BaseHDCRocketClassifier,
)


class HDCMultiRocketClassifier(_BaseHDCRocketClassifier):
    """MultiRocket with explicit time encoding using hyperdimensional computing.

    HDC-MultiRocket [1]_ adds the time of each convolution output to the MultiRocket
    features. As in MultiRocket, MiniRocket kernels are applied to the series and to
    its first order difference. Each time point is encoded as a hyperdimensional
    vector using sinusoidal fractional power encoding, with the similarity of the
    vectors of two time points decreasing with their distance at a rate set by
    ``beta``.

    In place of the proportion of positive values (PPV), feature ``i`` is the mean,
    over all time points, of dimension ``i`` of the time point vector where the
    convolution output exceeds the bias and zero elsewhere. The longest stretch of
    positive values (LSPV) and mean of positive values (MPV) features of MultiRocket
    are kept. The mean of indices of positive values (MIPV) is dropped, as its timing
    information is now in the time encoded PPV.

    Each channel is z-normalised using the training data, and the features are scaled
    to ``[-1, 1]`` and standardised before a ``RidgeClassifierCV`` is fitted. Unless
    ``beta`` is set, it is selected from ``2 ** np.linspace(0, 3, 7) - 1`` using the
    leave-one-out score of the ridge classifier on the training data.

    This follows the "hdc_auto" variant of the reference implementation [2]_.

    Parameters
    ----------
    n_kernels : int, default=6250
        Number of MiniRocket features for each of the series and its first order
        difference. Rounded down to a multiple of 84, with a minimum of 84. The
        transform has three features for each.
    hdc_dim : int, default=50000
        Dimension of the time encodings. Must be at least the number of MiniRocket
        features.
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
    HDCMiniRocketClassifier
    HDCMultiRocketHydraClassifier

    Notes
    -----
    The reference implementation allocates a fourth feature for each kernel but
    leaves it at zero, these constant features are not included. For multivariate
    data, the reference applies the kernels to the first order difference using
    only the channels of the first kernel. Here each kernel uses the channels it was
    fitted with, as in MultiRocket.

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
    ...     HDCMultiRocketClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = HDCMultiRocketClassifier(
    ...     n_kernels=84, hdc_dim=200, beta=1, random_state=0
    ... )
    >>> clf.fit(X, y)
    HDCMultiRocketClassifier(beta=1, hdc_dim=200, n_kernels=84, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    # the first order difference must be at least 9 long
    _min_length = 10
    _hdc_dim_param = "hdc_dim"

    def __init__(
        self,
        n_kernels=6250,
        hdc_dim=50000,
        max_dilations_per_kernel=32,
        beta=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.hdc_dim = hdc_dim
        self.max_dilations_per_kernel = max_dilations_per_kernel

        super().__init__(beta=beta, n_jobs=n_jobs, random_state=random_state)

    def _fit_kernels(self, X):
        self._parameters, self._parameters1 = _fit_multirocket(
            X, self.n_kernels, self.max_dilations_per_kernel, self._seed
        )

    def _n_hdc_columns(self):
        return 84 * int(np.sum(self._parameters[3]))

    def _transform_hdc(self, X, poses):
        return _transform_multirocket(X, self._parameters, self._parameters1, poses)

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
        return {"n_kernels": 84, "hdc_dim": 200, "beta": 1}


def _fit_multirocket(X, n_kernels, max_dilations_per_kernel, seed):
    """Fit MiniRocket parameters to a series and its first order difference."""
    n_kernels = max(n_kernels, 84)
    parameters = _static_fit(X, n_kernels, max_dilations_per_kernel, seed)
    parameters1 = _static_fit(
        np.diff(X, 1).astype(np.float32), n_kernels, max_dilations_per_kernel, seed
    )
    return parameters, parameters1


def _transform_multirocket(X, parameters, parameters1, poses):
    """Apply the HDC-MultiRocket transform to a series and its first difference."""
    X1 = np.diff(X, 1).astype(np.float32)
    Xt = _hdc_multirocket_transform(
        X, X1, parameters, parameters1, MiniRocket._indices, poses
    )
    return np.nan_to_num(Xt).astype(np.float64)


@njit(fastmath=True, parallel=True, cache=True)
def _hdc_multirocket_transform(X, X1, parameters, parameters1, indices, poses):
    n_cases, _, n_timepoints = X.shape
    n_features = len(indices) * np.sum(parameters[3])
    n_features1 = len(indices) * np.sum(parameters1[3])
    features = np.zeros((n_cases, 3 * (n_features + n_features1)), dtype=np.float32)

    for i in prange(n_cases):
        _hdc_multirocket_series(
            X[i], n_timepoints, parameters, indices, poses, features[i], 0
        )
        _hdc_multirocket_series(
            X1[i],
            n_timepoints,
            parameters1,
            indices,
            poses,
            features[i],
            3 * n_features,
        )

    return features


@njit(fastmath=True, cache=True)
def _hdc_multirocket_series(_X, n_timepoints, parameters, indices, poses, out, offset):
    """Write the time encoded PPV, LSPV and MPV features of one series to out.

    ``n_timepoints`` is the length of the undifferenced series, used for the
    convolution offsets of both the series and its difference as in MultiRocket.
    """
    n_channels, series_length = _X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_features = n_kernels * np.sum(n_features_per_dilation)

    A = -_X
    G = 3 * _X
    f_start = 0
    comb = 0
    n_channels_start = 0

    for j in range(len(dilations)):
        _padding0 = j % 2
        dilation = dilations[j]
        padding = (8 * dilation) // 2
        n_features_this_dilation = n_features_per_dilation[j]

        C_alpha = np.zeros((n_channels, series_length), dtype=np.float32)
        C_alpha[:] = A
        C_gamma = np.zeros((9, n_channels, series_length), dtype=np.float32)
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
                hi = series_length
            else:
                lo = padding
                hi = series_length - padding
            n = hi - lo

            for f in range(f_start, f_start + n_features_this_dilation):
                bias = biases[f]
                ppv = 0
                ppv_hdc = 0.0
                mean = 0.0
                last_val = 0
                max_stretch = 0.0

                for t in range(n):
                    c = C[lo + t]
                    if c > bias:
                        ppv += 1
                        ppv_hdc += poses[f, lo + t]
                        mean += c + bias
                    elif c < bias:
                        stretch = t - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch
                        last_val = t
                stretch = n - 1 - last_val
                if stretch > max_stretch:
                    max_stretch = stretch

                out[offset + f] = ppv_hdc / n if n > 0 else 0
                out[offset + n_features + f] = max_stretch
                out[offset + 2 * n_features + f] = mean / ppv if ppv > 0 else 0

            f_start += n_features_this_dilation
            comb += 1
            n_channels_start = n_channels_end
