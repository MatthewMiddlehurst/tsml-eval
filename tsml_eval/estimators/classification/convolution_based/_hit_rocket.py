"""Hadamard-vector Inner-product Transform ROCKET (HIT-ROCKET) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HITRocketClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _quiet_ridge_warnings,
)


def _hadamard_matrix(n):
    """Construct a Sylvester Hadamard matrix of power-of-two order ``n``."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard_order must be a power of 2, got {n}.")
    H = np.array([[1.0]], dtype=np.float32)
    size = 1
    while size < n:
        H = np.block([[H, H], [H, -H]])
        size *= 2
    return H


def _resolve_quantiles(bias_quantiles):
    if isinstance(bias_quantiles, (int, np.integer)):
        if int(bias_quantiles) < 1:
            raise ValueError(
                f"Integer bias_quantiles must be >= 1, got {bias_quantiles}."
            )
        phi = (np.sqrt(5) + 1) / 2
        return [float((i * phi) % 1) for i in range(1, int(bias_quantiles) + 1)]
    q_arr = np.asarray(bias_quantiles, dtype=np.float32)
    if q_arr.ndim != 1 or q_arr.size < 1 or np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError(
            "bias_quantiles must be a positive integer or a 1D sequence in [0, 1]."
        )
    return q_arr.tolist()


class HITRocketClassifier(BaseClassifier):
    """Hadamard-vector Inner-product Transform ROCKET (HIT-ROCKET) classifier.

    HIT-ROCKET [1]_ replaces random continuous or ternary convolution weights
    with deterministic, mutually orthogonal ``{+1, -1}`` vectors from an
    ``N``-order Sylvester Hadamard matrix ``H_N``. Across each dilation factor,
    ``n_samples`` training instances are randomly sampled and convolved with all
    ``N`` Hadamard kernels to extract quantile bias thresholds. Proportion of
    positive values (PPV) features are computed for every
    ``(dilation, kernel, quantile, sample)`` combination (producing
    ``hadamard_order * len(dilations) * n_quantiles * n_samples`` features,
    4,400 by default), scaled with ``StandardScaler(with_mean=False)``, and
    classified with a balanced ``RidgeClassifierCV``.

    This follows [1]_ and the authors' experimental implementation in
    ``notebooks/Exp01_Linear_Classification.ipynb`` of [2]_.

    Parameters
    ----------
    hadamard_order : int, default=16
        Order ``N`` of the Sylvester Hadamard matrix (must be a power of 2),
        giving ``N`` orthogonal kernels of length ``N``.
    dilations : sequence of int or None, default=None
        Dilation factors applied to the Hadamard kernels. If None, uses
        ``range(1, 12)`` (dilations 1 through 11) as in [1]_ and [2]_.
    bias_quantiles : int or sequence of float, default=5
        If an integer ``P``, generates ``P`` low-discrepancy golden-ratio
        quantiles ``[(i * phi) % 1 for i in range(1, P + 1)]`` as in [2]_; if a
        1D sequence of floats in ``[0, 1]``, uses those quantile levels.
    n_samples : int, default=5
        Maximum number of training instances sampled without replacement to
        estimate per-sample quantile biases (clamped to ``n_cases``).
    n_jobs : int, default=1
        The number of PyTorch CPU threads used during convolution. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator controlling bias-instance sampling.

    Attributes
    ----------
    biases_ : np.ndarray of shape (n_actual_samples, n_dilations * N, n_quantiles)
        Fitted quantile bias thresholds across sampled training instances.
    classifier_ : Pipeline
        Fitted ``StandardScaler(with_mean=False)`` and ``RidgeClassifierCV``
        pipeline.

    Notes
    -----
    Requires ``torch>=2.0`` (a CPU build is sufficient; ``n_jobs`` controls
    ``torch.set_num_threads``).

    The repository [2]_ contains two divergent implementations:
    ``notebooks/Exp01_Linear_Classification.ipynb``, which implements the
    4,400-feature ``L * D * P * S`` transform of Equation 19 and produces the
    UCR results in Tables 6–7 of [1]_, and ``src/hitrocket.py``, an incomplete
    refactor with placeholder loops that averages biases across samples (288
    features in ``scripts/run_linear.py``). This class implements the paper
    version from ``Exp01_Linear_Classification.ipynb`` with three bugs fixed:

    1. In ``Exp01_Linear_Classification.ipynb``, ``_fit_biases`` concatenates
       per-dilation quantile tensors along axis 1 into shape
       ``(n_samples, len(dilations) * hadamard_order, n_quantiles)``, but
       ``forward`` indexes ``self.biases[sample_idx, kernel_idx, bias_idx]``
       without ``dilation_idx * hadamard_order``, reusing ``dilation=1`` biases
       for every dilation. Indexing ``dilation_idx * hadamard_order + kernel_idx``
       improves accuracy across tested UCR datasets (for example, ``ACSF1``
       ``0.8500 -> 0.9100``, ``Adiac`` ``0.7928 -> 0.8235``, ``OSULeaf``
       ``0.9132 -> 0.9463``, ``ItalyPowerDemand`` ``0.9534 -> 0.9621``).
    2. ``_fit_biases`` in ``Exp01_Linear_Classification.ipynb`` raises
       ``ValueError`` when ``n_cases < n_samples``; ``n_samples`` is clamped to
       ``n_cases`` (as in ``src/hitrocket.py``).
    3. ``_quantiles_torch`` in ``Exp01_Linear_Classification.ipynb`` hardcodes
       ``device="cuda"`` and ``_fit_biases`` draws from unseeded
       ``np.random.choice``. Computations run on CPU and instance sampling uses
       ``random_state``.

    References
    ----------
    .. [1] Hao, Wang, Zhang Kuang, Hou Chengyu, Yuan Zhonghao, Tan Chenxing,
       Fu Weifeng, and Yao Xingran. "HITrocket: Hadamard-vector inner-product
       transform of ROCKET for time series classification." Pattern Recognition
       180 (2026): 114220.
    .. [2] https://github.com/Einstein-sworder/HitRocket

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     HITRocketClassifier,
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    >>> clf = HITRocketClassifier(
    ...     hadamard_order=8, dilations=[1, 2], bias_quantiles=2, n_samples=2,
    ...     random_state=0,
    ... )  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    HITRocketClassifier(bias_quantiles=2, dilations=[1, 2], hadamard_order=8,
                        n_samples=2, random_state=0)
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": "torch>=2.0",
    }

    def __init__(
        self,
        hadamard_order=16,
        dilations=None,
        bias_quantiles=5,
        n_samples=5,
        n_jobs=1,
        random_state=None,
    ):
        self.hadamard_order = hadamard_order
        self.dilations = dilations
        self.bias_quantiles = bias_quantiles
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    @staticmethod
    def _preprocess(X):
        X_2d = np.nan_to_num(X[:, 0, :], nan=0.0).astype(np.float32)
        if X_2d.shape[1] < 2:
            X_2d = np.pad(X_2d, ((0, 0), (0, 2 - X_2d.shape[1])))
        return X_2d

    def _transform_features(self, X_2d):
        import torch
        import torch.nn.functional as F

        n_cases, sig_len = X_2d.shape
        k_size = self.hadamard_order
        n_dil = len(self._dilations)
        n_actual_samples, _, n_q = self.biases_.shape
        feat_per_dil = k_size * n_q * n_actual_samples
        features = np.empty((n_cases, n_dil * feat_per_dil), dtype=np.float32)

        weights_t = torch.from_numpy(self._weights).unsqueeze(1)
        biases_t = torch.from_numpy(self.biases_)
        batch_size = max(1, int(2e7 // max(1, feat_per_dil * sig_len)))

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self._n_jobs)
        try:
            with torch.no_grad():
                for start in range(0, n_cases, batch_size):
                    end = min(n_cases, start + batch_size)
                    xb = torch.from_numpy(X_2d[start:end]).unsqueeze(1)
                    col = 0
                    for d_idx, d in enumerate(self._dilations):
                        pad = ((k_size - 1) * d) // 2
                        co = F.conv1d(F.pad(xb, (pad, pad)), weights_t, dilation=d)
                        # Match (kernel, quantile, sample) feature order of Exp01
                        b = biases_t[
                            :, d_idx * k_size : (d_idx + 1) * k_size, :
                        ].permute(1, 2, 0)
                        co_exp = co.unsqueeze(-1).unsqueeze(-1)
                        b_exp = b.unsqueeze(0).unsqueeze(2)
                        ppv = (co_exp > b_exp).float().mean(dim=2)
                        features[start:end, col : col + feat_per_dil] = (
                            ppv.reshape(end - start, feat_per_dil).numpy()
                        )
                        col += feat_per_dil
        finally:
            torch.set_num_threads(prev_threads)

        return features

    def _fit(self, X, y):
        import torch
        import torch.nn.functional as F

        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}.")

        self._weights = _hadamard_matrix(int(self.hadamard_order)).T
        if self.dilations is None:
            self._dilations = list(range(1, 12))
        else:
            self._dilations = [int(d) for d in self.dilations]
            if len(self._dilations) < 1 or any(d < 1 for d in self._dilations):
                raise ValueError("dilations must be a non-empty sequence of ints >= 1.")

        quantiles_list = _resolve_quantiles(self.bias_quantiles)
        self._n_jobs = check_n_jobs(self.n_jobs)

        X_2d = self._preprocess(X)
        n_cases = X_2d.shape[0]
        actual_samples = min(int(self.n_samples), n_cases)

        rng = check_random_state(self.random_state)
        sample_indices = rng.choice(n_cases, size=actual_samples, replace=False)

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self._n_jobs)
        try:
            with torch.no_grad():
                x_samples = torch.from_numpy(X_2d[sample_indices]).unsqueeze(1)
                weights_t = torch.from_numpy(self._weights).unsqueeze(1)
                q_t = torch.tensor(quantiles_list, dtype=torch.float32)
                k_size = int(self.hadamard_order)

                biases_blocks = []
                for d in self._dilations:
                    pad = ((k_size - 1) * d) // 2
                    co = F.conv1d(
                        F.pad(x_samples, (pad, pad)), weights_t, dilation=d
                    ).view(actual_samples, k_size, -1)
                    q = (
                        torch.quantile(co, q_t, dim=2)
                        .permute(1, 2, 0)
                        .reshape(actual_samples, k_size, len(quantiles_list))
                    )
                    biases_blocks.append(q)
                self.biases_ = torch.cat(biases_blocks, dim=1).numpy()
        finally:
            torch.set_num_threads(prev_threads)

        if self.n_classes_ == 1:
            self.classifier_ = None
            return self

        Xt = self._transform_features(X_2d)
        _, y_ind = np.unique(y, return_inverse=True)
        with _quiet_ridge_warnings():
            self.classifier_ = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(
                    alphas=[0.01, 0.1, 1.0, 10.0],
                    class_weight="balanced",
                ),
            ).fit(Xt, y_ind)

        return self

    def _predict(self, X):
        if self.n_classes_ == 1:
            return np.repeat(self.classes_[0], len(X))

        Xt = self._transform_features(self._preprocess(X))
        preds_ind = self.classifier_.predict(Xt)
        return self.classes_[preds_ind]

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
            "hadamard_order": 8,
            "dilations": [1, 2],
            "bias_quantiles": 2,
            "n_samples": 2,
        }
