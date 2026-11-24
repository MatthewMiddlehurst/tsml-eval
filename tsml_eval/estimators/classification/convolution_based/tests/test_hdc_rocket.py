"""Tests for the HDC-ROCKET classifiers."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import (
    HydraTransformer,
    MiniRocket,
    MultiRocket,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    HDCHydraClassifier,
    HDCMiniRocketClassifier,
    HDCMultiRocketClassifier,
    HDCMultiRocketHydraClassifier,
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
def test_hdc_minirocket_matches_minirocket(n_channels):
    """Test HDC-MiniRocket with constant encodings counts the MiniRocket PPV."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=n_channels, n_timepoints=40, random_state=0
    )
    clf = HDCMiniRocketClassifier(n_kernels=168, beta=0, random_state=0).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))

    counts = clf._transform_hdc(Xp, _ones_poses(168, 40))
    minirocket = MiniRocket(n_kernels=168, random_state=0).fit(Xp)
    ppv = minirocket.transform(Xp)

    # alternate kernels only use the outputs without padding
    n_positions = []
    dilations, n_features_per_dilation = minirocket.parameters[2:4]
    for j, (dilation, n_features) in enumerate(zip(dilations, n_features_per_dilation)):
        for k in range(84):
            padded = (j + k) % 2 == 1
            n_positions += [40 - 8 * dilation if padded else 40] * n_features

    np.testing.assert_allclose(counts, ppv * np.array(n_positions), atol=1e-4)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_hdc_multirocket_matches_multirocket(n_channels):
    """Test HDC-MultiRocket with constant encodings gives the MultiRocket features."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=n_channels, n_timepoints=40, random_state=0
    )
    clf = HDCMultiRocketClassifier(
        n_kernels=168, hdc_dim=400, beta=0, random_state=0
    ).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))

    hdc = clf._transform_hdc(Xp, _ones_poses(168, 40))
    mr = MultiRocket(n_kernels=168, normalise=False, random_state=0).fit_transform(Xp)

    # MultiRocket has PPV, LSPV, MPV and MIPV blocks for each input, HDC-MultiRocket
    # drops MIPV
    nf = 168
    mr = np.hstack([mr[:, i * nf : (i + 1) * nf] for i in [0, 1, 2, 4, 5, 6]])
    if n_channels > 1:
        # aeon MultiRocket draws new channels to fit the difference biases
        hdc, mr = hdc[:, : 3 * nf], mr[:, : 3 * nf]
    # aeon MultiRocket takes the difference after casting to single precision
    np.testing.assert_allclose(hdc, mr, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_hdc_hydra_matches_hydra(n_channels):
    """Test HDC-Hydra with constant encodings gives the Hydra features."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=n_channels, n_timepoints=40, random_state=0
    )
    clf = HDCHydraClassifier(n_groups=8, hdc_dim=500, beta=0, random_state=0).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))

    hdc = clf._transform_hdc(Xp, _ones_poses(clf._n_hdc_columns(), 40))
    hydra = HydraTransformer(n_groups=8, random_state=0).fit_transform(Xp)

    # each block of aeon Hydra has the maximum values then the minimum counts,
    # HDC-Hydra has the counts first, and adds univariate series to themselves
    n = 8 * 4
    blocks = hydra.reshape(len(X), -1, 2, n)
    scale = 2 if n_channels == 1 else 1
    expected = np.stack((blocks[:, :, 1], scale * blocks[:, :, 0]), 2)
    np.testing.assert_allclose(hdc, expected.reshape(len(X), -1), rtol=1e-3, atol=1e-3)
    # the transform is stored in half precision, as in the reference
    assert hdc.dtype == np.float16


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_hdc_multirocket_hydra_features():
    """Test HDC-MultiRocket-Hydra concatenates the two transforms."""
    X, y = make_example_3d_numpy(n_cases=8, n_timepoints=40, random_state=0)
    params = {"beta": 1, "random_state": 0}
    clf = HDCMultiRocketHydraClassifier(
        n_kernels=168, hydra_n_groups=8, hdc_dim=500, **params
    ).fit(X, y)
    mr = HDCMultiRocketClassifier(n_kernels=168, hdc_dim=500, **params).fit(X, y)
    hydra = HDCHydraClassifier(n_groups=8, hdc_dim=500, **params).fit(X, y)
    Xp = clf._preprocess(X.astype(np.float64))

    np.testing.assert_allclose(
        clf._transform_hdc(Xp, clf._poses),
        np.hstack(
            (mr._transform_hdc(Xp, mr._poses), hydra._transform_hdc(Xp, hydra._poses))
        ),
    )
    assert clf._n_multirocket_features == 6 * 168


def test_default_feature_counts():
    """Test the default number of features matches Table 3 of the paper."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=100, random_state=0)

    mini = HDCMiniRocketClassifier(beta=1, random_state=0).fit(X, y)
    assert mini.classifier_.coef_.shape[-1] == 9996

    multi = HDCMultiRocketClassifier(beta=1, random_state=0).fit(X, y)
    assert multi.classifier_.coef_.shape[-1] == 37296


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
