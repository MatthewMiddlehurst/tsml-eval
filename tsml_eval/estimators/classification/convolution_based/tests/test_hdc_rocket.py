"""Tests for the HDC-ROCKET classifiers."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from aeon.transformations.collection.convolution_based._minirocket import (
    _static_transform_multi,
)
from aeon.transformations.collection.convolution_based._multirocket import (
    _transform_uni,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    HDCHydraClassifier,
    HDCMiniRocketClassifier,
    HDCMultiRocketClassifier,
    HDCMultiRocketHydraClassifier,
)
from tsml_eval.estimators.classification.convolution_based._hdc_hydra import (
    _hdc_hydra_transform,
)
from tsml_eval.estimators.classification.convolution_based._hdc_minirocket import (
    _hdc_minirocket_transform,
)
from tsml_eval.estimators.classification.convolution_based._hdc_multirocket import (
    _transform_multirocket,
)
from tsml_eval.estimators.classification.convolution_based._hdc_utils import (
    HDC_BETAS,
    _fpe_poses,
)


def test_fpe_poses():
    """Test the structure of the FPE time encodings."""
    poses = _fpe_poses(50, 3, 20, 0)

    assert poses.shape == (20, 50)
    # the first phase is zero, so its cosine dimension is constant
    np.testing.assert_allclose(poses[0], 1)
    # each time point vector is cos(p * e) followed by the reversed sin(p * e)
    np.testing.assert_allclose(poses[:10] ** 2 + poses[10:][::-1] ** 2, 1, rtol=1e-5)
    # with beta=0 every time point has the same encoding
    np.testing.assert_allclose(_fpe_poses(50, 0, 20, 0), poses[:, :1].repeat(50, 1))
    # a prefix of the dimensions matches the full encoding
    np.testing.assert_array_equal(_fpe_poses(50, 3, 20, 0, n_columns=13), poses[:13])

    with pytest.raises(ValueError, match="HDC dimension"):
        _fpe_poses(50, 3, 20, 0, n_columns=21)


def _ones_poses(n_columns, n_timepoints):
    return np.ones((n_columns, n_timepoints), dtype=np.float32)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_hdc_minirocket_transform_counts(n_channels):
    """Test HDC-MiniRocket with constant encodings counts the MiniRocket PPV."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=n_channels, n_timepoints=40, random_state=0
    )
    clf = HDCMiniRocketClassifier(n_kernels=168, beta=0, random_state=0).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))
    parameters = clf._parameters

    counts = _hdc_minirocket_transform(
        Xp, parameters, MiniRocket._indices, _ones_poses(168, 40)
    )
    ppv = _static_transform_multi(Xp, parameters, MiniRocket._indices)

    # alternate kernels only use the outputs without padding
    n_positions = []
    for j, (dilation, n_features) in enumerate(zip(parameters[2], parameters[3])):
        for k in range(84):
            padded = (j + k) % 2 == 1
            n_positions += [40 - 8 * dilation if padded else 40] * n_features

    np.testing.assert_allclose(counts, ppv * np.array(n_positions), atol=1e-4)


def test_hdc_multirocket_transform_matches_multirocket():
    """Test HDC-MultiRocket with constant encodings gives the MultiRocket features."""
    X, y = make_example_3d_numpy(n_cases=8, n_timepoints=40, random_state=0)
    clf = HDCMultiRocketClassifier(
        n_kernels=168, hdc_dim=400, beta=0, random_state=0
    ).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))

    hdc = _transform_multirocket(
        Xp, clf._parameters, clf._parameters1, _ones_poses(168, 40)
    )
    # the univariate MultiRocket parameters have no channel information
    X2 = Xp[:, 0]
    mr = _transform_uni(
        X2,
        np.diff(X2, 1),
        clf._parameters[2:],
        clf._parameters1[2:],
        4,
        MultiRocket._indices,
        0,
    )

    # MultiRocket has PPV, LSPV, MPV and MIPV blocks for each input
    nf = 168
    mr = np.hstack([mr[:, i * nf : (i + 1) * nf] for i in [0, 1, 2, 4, 5, 6]])
    np.testing.assert_allclose(hdc, mr, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_hdc_hydra_transform_matches_hydra():
    """Test HDC-Hydra with constant encodings gives the multivariate Hydra features."""
    import torch
    from aeon.transformations.collection.convolution_based._hydra import (
        _HydraInternal,
    )

    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=3, n_timepoints=40, random_state=0
    )
    clf = HDCHydraClassifier(n_groups=8, hdc_dim=500, beta=0, random_state=0).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))
    hydra = clf._hydra

    hdc = _hdc_hydra_transform(Xp, hydra, _ones_poses(clf._n_hdc_columns(), 40))

    aeon_hydra = _HydraInternal(40, 3, k=8, g=8)
    aeon_hydra.W = torch.stack(hydra["weights"])
    aeon_hydra.idx = hydra["channels"]
    with torch.no_grad():
        Z = aeon_hydra(torch.from_numpy(Xp)).numpy()

    # aeon gives the max then min counts for each block, HDC-Hydra min then max
    Z = Z.reshape(len(Z), -1, 2, 32)[:, :, ::-1].reshape(len(Z), -1)
    np.testing.assert_allclose(hdc, Z, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "classifier, params",
    [
        (HDCMiniRocketClassifier, {"n_kernels": 84}),
        (HDCMultiRocketClassifier, {"n_kernels": 84, "hdc_dim": 200}),
        (HDCHydraClassifier, {"n_groups": 8, "hdc_dim": 500}),
        (
            HDCMultiRocketHydraClassifier,
            {"n_kernels": 84, "hydra_n_groups": 8, "hdc_dim": 500},
        ),
    ],
)
def test_beta_selection(classifier, params):
    """Test beta is selected from the reference grid on short series."""
    deps = classifier.get_class_tag("python_dependencies")
    if deps is not None and not _check_soft_dependencies(deps, severity="none"):
        pytest.skip("skip test if required soft dependency not available")

    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=8, random_state=0)

    classifier = classifier(random_state=0, **params).fit(X, y)

    assert classifier.beta_ in HDC_BETAS
    assert classifier.beta_scores_.max() == classifier.classifier_.best_score_
    assert set(classifier.predict(X)) <= set(y)
