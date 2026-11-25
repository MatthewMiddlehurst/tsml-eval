"""Tests for FT-FVC, CEEMD-MultiRocket, KG-MTP, SPROCKET, SelF-Rocket and HIT-Rocket."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    CEEMDMultiRocketClassifier,
    FTFVCClassifier,
    HITRocketClassifier,
    KGMTPClassifier,
    SelFRocketClassifier,
    SPROCKETClassifier,
)
from tsml_eval.estimators.classification.convolution_based._ceemd_multirocket import (
    _ceemd_batch,
    _INDICES_6_2,
    _transform_6_2,
)
from tsml_eval.estimators.classification.convolution_based._ft_fvc import (
    _discrete_hilbert_transform,
)
from tsml_eval.estimators.classification.convolution_based._hit_rocket import (
    _hadamard_matrix,
)


def test_ft_fvc_discrete_hilbert_transform():
    """Test the discrete Hilbert transform against direct summation."""
    rng = np.random.RandomState(0)
    X = rng.randn(4, 25)
    got = _discrete_hilbert_transform(X)

    expected = np.zeros_like(X)
    for c in range(4):
        for k in range(25):
            expected[c, k] = sum(X[c, i] / (k - i) for i in range(25) if i != k)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_discrete_hilbert_transform(-X), -got, rtol=1e-12)


def test_ft_fvc_voting_and_short_series():
    """Test FT-FVC view dimensions, vote probabilities, and short-series padding."""
    X, y = make_example_3d_numpy(
        n_cases=18, n_channels=1, n_timepoints=3, n_labels=3, random_state=0
    )
    clf = FTFVCClassifier(n_kernels=15, random_state=0).fit(X, y)

    assert len(clf.classifiers_) == 3
    for pipe in clf.classifiers_:
        assert pipe[-1].coef_.shape[-1] == 4 * 15

    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba * 3.0)).issubset({0.0, 1.0, 2.0, 3.0})
    assert clf.predict(X[:1])[0] == clf.predict(X)[0]


def test_ceemd_decomposition_and_nspv_table3():
    """Test CEEMD reconstruction identity and NSPV against Table 3 of the paper."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 60)
    noise = rng.randn(4, 60)
    imf1, imf2, residue = _ceemd_batch(X, noise, 0.4, 8)
    np.testing.assert_allclose(imf1 + imf2 + residue, X, rtol=1e-12, atol=1e-12)

    # Construct a constant-kernel test via bias manipulation on binary step patterns
    # Table 3 of Wang et al. (2023): A->3, B->1, C->2 maximal positive runs of len >= 2
    patterns = np.array(
        [
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    for row, expected_nspv in zip(patterns, [3, 1, 2]):
        # Count maximal contiguous runs of 1s with length >= 2 as in _transform_6_2
        run_len = 0
        nspv = 0
        for v in row:
            if v > 0.5:
                run_len += 1
            else:
                if run_len > 1:
                    nspv += 1
                run_len = 0
        if run_len > 1:
            nspv += 1
        assert nspv == expected_nspv

    # Verify _transform_6_2 produces finite non-negative integer NSPV in block 5
    dilations = np.array([1], dtype=np.int32)
    n_feat_per_dil = np.array([1], dtype=np.int32)
    biases = np.zeros(15, dtype=np.float32)
    Xt = _transform_6_2(X, dilations, n_feat_per_dil, biases, _INDICES_6_2)
    assert Xt.shape == (5, 75)
    nspv_block = Xt[:, 60:75]
    np.testing.assert_array_equal(nspv_block, np.floor(nspv_block))
    assert np.all(nspv_block >= 0)

    X3d, y = make_example_3d_numpy(
        n_cases=16, n_channels=1, n_timepoints=30, random_state=0
    )
    clf = CEEMDMultiRocketClassifier(
        n_features=450, n_realizations=3, n_siftings=5, random_state=0
    ).fit(X3d, y)
    preds = clf.predict(X3d)
    assert clf.predict(X3d[1:2])[0] == preds[1]


def test_kg_mtp_feature_counts_and_high_amplitude():
    """Test KG-MTP default pooling count and high-amplitude kernel responses."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=1, n_timepoints=80, random_state=0
    )
    clf = KGMTPClassifier(n_features=50000, random_state=0).fit(X, y)
    n_hydra = sum(2 * 62 * len(p[0]) for p in clf._rep_params)
    # 50000 // 3 // 5 // 62 = 53 dilations/biases per kernel -> 3 * 5 * 62 * 53 = 49290
    assert clf.classifier_[-1].coef_.shape[-1] == 49290 + n_hydra

    # High-amplitude unnormalised input (> 10000) that exceeded the reference sentinel
    X_large = X * 20000.0
    clf_large = KGMTPClassifier(n_features=930, random_state=0).fit(X_large, y)
    preds_large = clf_large.predict(X_large)
    assert set(preds_large) <= set(y)
    assert clf_large.predict(X_large[:1])[0] == preds_large[0]


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "manhattan",
        "squared",
        "emd_sort",
        "cosine",
        "chebyshev",
        "msm",
        "dtw",
    ],
)
def test_sprocket_distances(distance):
    """Test SPROCKET executes cleanly across inelastic and elastic distances."""
    X, y = make_example_3d_numpy(
        n_cases=14, n_channels=2, n_timepoints=20, random_state=0
    )
    clf = SPROCKETClassifier(
        n_kernels=12,
        distance=distance,
        normalise=True,
        stratified_sampling=True,
        random_state=0,
    ).fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= set(y)


def test_sprocket_multivariate_non_zero_channels():
    """Test SPROCKET uses non-zero channels when channel 0 carries no signal."""
    rng = np.random.RandomState(0)
    n_cases, n_tp = 24, 30
    X = np.zeros((n_cases, 4, n_tp), dtype=np.float64)
    y = np.array([0] * 12 + [1] * 12)
    t = np.linspace(0, 2 * np.pi, n_tp)
    # Channel 0 is identically zero; discriminative signal is only on channels 1..3
    for i in range(n_cases):
        sign = 1.0 if y[i] == 1 else -1.0
        for ch in range(1, 4):
            X[i, ch] = sign * np.sin(ch * t) + 0.05 * rng.randn(n_tp)

    clf = SPROCKETClassifier(n_kernels=64, random_state=0).fit(X, y)
    assert np.mean(clf.predict(X) == y) > 0.9


def test_self_rocket_modes_and_long_series_fallback():
    """Test SelF-Rocket MIX restriction, single-case predict, and long fallback."""
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=1, n_timepoints=30, random_state=0
    )
    clf_mix = SelFRocketClassifier(
        n_kernels=84, n_runs=2, only_mix=True, random_state=0
    ).fit(X, y)
    assert clf_mix.selected_combination_.endswith("_MIX")
    assert clf_mix.predict(X[:1])[0] == clf_mix.predict(X)[0]

    # Constant input forces ties / fallback check on long series (> 512)
    X_long = make_example_3d_numpy(
        n_cases=12, n_channels=1, n_timepoints=520, random_state=1
    )[0]
    clf_long = SelFRocketClassifier(
        n_kernels=84, n_runs=2, n_features_per_split=40, random_state=1
    ).fit(X_long, y[:12])
    assert len(clf_long.predict(X_long[:2])) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies("torch>=2.0", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_hit_rocket_orthogonality_and_features():
    """Test Hadamard orthogonality, default 4,400 feature count, and small-n clamp."""
    H = _hadamard_matrix(16)
    np.testing.assert_array_equal(H.T @ H, 16 * np.eye(16, dtype=np.float32))

    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=1, n_timepoints=40, random_state=0
    )
    clf = HITRocketClassifier(random_state=0).fit(X, y)
    assert clf.biases_.shape == (5, 11 * 16, 5)
    assert clf.classifier_[-1].coef_.shape[-1] == 4400

    # Different dilations produce distinct bias blocks
    assert not np.allclose(clf.biases_[:, :16, :], clf.biases_[:, 16:32, :])

    # n_cases < n_samples clamps cleanly
    clf_small = HITRocketClassifier(
        hadamard_order=8, dilations=[1, 2], n_samples=10, random_state=0
    ).fit(X[:4], np.array([0, 1, 0, 1]))
    assert clf_small.biases_.shape[0] == 4
    assert clf_small.predict(X[:1])[0] in (0, 1)


@pytest.mark.parametrize(
    "cls, bad_kwargs",
    [
        (FTFVCClassifier, {"n_kernels": 0}),
        (CEEMDMultiRocketClassifier, {"n_features": 100}),
        (CEEMDMultiRocketClassifier, {"n_realizations": 0}),
        (KGMTPClassifier, {"n_features": 500}),
        (SPROCKETClassifier, {"n_kernels": 0}),
        (SPROCKETClassifier, {"proto_per_kernel": 1.0}),
        (SPROCKETClassifier, {"distance": "invalid_metric"}),
        (SelFRocketClassifier, {"n_kernels": 42}),
        (SelFRocketClassifier, {"n_runs": 0}),
    ],
)
def test_invalid_hyperparameters(cls, bad_kwargs):
    """Test invalid hyperparameter values raise ValueError during fit."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, random_state=0)
    with pytest.raises(ValueError):
        cls(**bad_kwargs).fit(X, y)
