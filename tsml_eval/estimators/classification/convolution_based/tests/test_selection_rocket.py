"""Tests for the kernel and feature selection ROCKET classifiers."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    ClassWiseMiniRocketClassifier,
    DetachRocketClassifier,
    DetachRocketEnsembleClassifier,
    POCKETClassifier,
    SRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._cw_minirocket import (
    _detect_knee_channels,
)
from tsml_eval.estimators.classification.convolution_based._selection_utils import (
    _prune_rocket_kernels,
    _rocket_transform,
)


def test_prune_rocket_kernels_matches_subset():
    """Test pruned Rocket kernel tuple produces identical features to slicing."""
    X, _ = make_example_3d_numpy(
        n_cases=8, n_channels=3, n_timepoints=30, random_state=0
    )
    rocket = Rocket(n_kernels=15, random_state=0).fit(X)
    full_out = rocket.transform(X)

    mask = np.zeros(15, dtype=bool)
    mask[[1, 4, 7, 13]] = True
    pruned = _prune_rocket_kernels(rocket.kernels, mask)
    pruned_out = _rocket_transform(rocket, X, pruned)

    feat_mask = np.repeat(mask, 2)
    np.testing.assert_array_equal(pruned_out, full_out[:, feat_mask])
    np.testing.assert_array_equal(rocket.transform(X), full_out)


@pytest.mark.parametrize("rocket_transform", ["rocket", "rocket-ppvmax", "minirocket"])
@pytest.mark.parametrize("init_population", ["half_bernoulli", "ones"])
def test_srocket_variants(rocket_transform, init_population):
    """Test SRocketClassifier across transform and initialisation modes."""
    n_kernels = 84 if rocket_transform == "minirocket" else 30
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=25, n_labels=3, random_state=0
    )
    clf = SRocketClassifier(
        n_kernels=n_kernels,
        n_epochs=15,
        pop_size=4,
        init_population=init_population,
        rocket_transform=rocket_transform,
        random_state=0,
    ).fit(X, y)

    assert 1 <= clf.n_selected_kernels_ <= n_kernels
    assert clf.n_selected_kernels_ == int(clf.kernel_mask_.sum())
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= set(y)


@pytest.mark.parametrize("rocket_transform", ["rocket", "rocket-ppv", "minirocket"])
@pytest.mark.parametrize("stage2", [True, False])
def test_pocket_variants(rocket_transform, stage2):
    """Test POCKETClassifier across transform modes and Stage 1/2 settings."""
    n_kernels = 84 if rocket_transform == "minirocket" else 30
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=25, n_labels=3, random_state=0
    )
    clf = POCKETClassifier(
        n_kernels=n_kernels,
        remain_ratio=0.4,
        n_iter=15,
        k=[0.1, 1.0],
        stage2=stage2,
        rocket_transform=rocket_transform,
        random_state=0,
    ).fit(X, y)

    expected_max = int(n_kernels * 0.4)
    assert 1 <= clf.n_remaining_kernels_ <= expected_max
    assert clf.k_ in (0.1, 1.0)
    assert (clf.classifier_ is not None) is stage2
    preds = clf.predict(X)
    assert set(preds) <= set(y)


@pytest.mark.parametrize("rocket_transform", ["rocket", "minirocket", "multirocket"])
@pytest.mark.parametrize("set_percentage", [None, 20.0])
def test_detach_rocket_variants(rocket_transform, set_percentage):
    """Test DetachRocketClassifier across transforms and pruning modes."""
    n_kernels = 84 if rocket_transform in ("minirocket", "multirocket") else 30
    X, y = make_example_3d_numpy(
        n_cases=24, n_channels=2, n_timepoints=25, n_labels=3, random_state=0
    )
    clf = DetachRocketClassifier(
        n_kernels=n_kernels,
        set_percentage=set_percentage,
        n_steps=20,
        rocket_transform=rocket_transform,
        random_state=0,
    ).fit(X, y)

    assert 0.0 < clf.retained_ratio_ <= 1.0
    assert int(clf.feature_mask_.sum()) >= 1
    assert set(clf.predict(X)) <= set(y)


def test_detach_rocket_small_imbalanced_fallback():
    """Test DetachRocketClassifier falls back to 10% or 5% when split fails."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(7, 1, 20))
    # Class 2 has only 1 instance so stratified validation split is impossible
    y = np.array([0, 0, 0, 1, 1, 1, 2])

    clf = DetachRocketClassifier(n_kernels=50, n_steps=60, random_state=0).fit(X, y)
    assert clf.retained_ratio_ == pytest.approx(0.10, abs=0.02)
    assert set(clf.predict(X)) <= set(y)

    clf_mr = DetachRocketClassifier(
        n_kernels=84, n_steps=70, rocket_transform="multirocket", random_state=0
    ).fit(X, y)
    assert clf_mr.retained_ratio_ == pytest.approx(0.05, abs=0.015)
    assert set(clf_mr.predict(X)) <= set(y)


@pytest.mark.skipif(
    not _check_soft_dependencies("aeon>=1.6.0", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("rocket_transform", ["minirocket", "multirocket", "rocket"])
@pytest.mark.parametrize("voting", ["soft", "hard"])
def test_detach_rocket_ensemble_and_channel_relevance(rocket_transform, voting):
    """Test DetachRocketEnsembleClassifier probabilities and channel relevance."""
    rng = np.random.RandomState(0)
    n_cases, n_channels, n_timepoints = 36, 4, 40
    X = rng.normal(0.0, 0.2, size=(n_cases, n_channels, n_timepoints))
    y = np.repeat([0, 1], n_cases // 2)
    # Concentrate discriminative signal solely on channel 0
    t = np.linspace(0, 4 * np.pi, n_timepoints)
    X[y == 1, 0, :] += 3.0 * np.sin(t)
    X[y == 0, 0, :] -= 3.0 * np.sin(t)

    n_kernels = (
        672
        if rocket_transform == "multirocket"
        else (84 if rocket_transform == "minirocket" else 40)
    )
    clf = DetachRocketEnsembleClassifier(
        n_estimators=4,
        n_kernels=n_kernels,
        n_steps=25,
        voting=voting,
        rocket_transform=rocket_transform,
        random_state=0,
    ).fit(X, y)

    probas = clf.predict_proba(X)
    assert probas.shape == (n_cases, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)
    assert clf.channel_relevance_.shape == (n_channels,)
    assert clf.channel_relevance_.sum() == pytest.approx(1.0)
    # Channel 0 carries the class signal and should receive the largest relevance
    assert int(np.argmax(clf.channel_relevance_)) == 0


@pytest.mark.parametrize("channel_selection", ["ecp", "ecs", None])
@pytest.mark.parametrize("distance", ["euclidean", "dtw"])
def test_classwise_minirocket_channel_selection(channel_selection, distance):
    """Test ClassWiseMiniRocketClassifier local per-class channel selection."""
    rng = np.random.RandomState(0)
    n_cases, n_channels, n_timepoints = 30, 5, 30
    X = rng.normal(0.0, 0.05, size=(n_cases, n_channels, n_timepoints))
    y = np.repeat([0, 1, 2], 10)
    # Make channel k strongly distinguish class k from the others
    X[y == 0, 0, :] += 5.0
    X[y == 1, 1, :] += 5.0
    X[y == 2, 2, :] += 5.0

    clf = ClassWiseMiniRocketClassifier(
        n_kernels=84,
        channel_selection=channel_selection,
        distance=distance,
        random_state=0,
    ).fit(X, y)

    assert len(clf.channels_selected_) == 3
    if channel_selection is None:
        assert all(len(ch) == n_channels for ch in clf.channels_selected_)
    else:
        for k in range(3):
            assert k in clf.channels_selected_[k]
            assert len(clf.channels_selected_[k]) < n_channels

    probas = clf.predict_proba(X)
    assert probas.shape == (n_cases, 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)
    assert np.mean(clf.predict(X) == y) == 1.0


def test_detect_knee_two_channel_fallback():
    """Test elbow detection selects the top channel on two-channel inputs."""
    selected = _detect_knee_channels(np.array([4.2, 0.5]), np.array([1, 0]))
    assert selected == [1]


def test_parameter_validation_errors():
    """Test invalid hyperparameters raise clear ValueErrors."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=2, random_state=0)

    with pytest.raises(ValueError, match="n_kernels"):
        SRocketClassifier(n_kernels=0).fit(X, y)
    with pytest.raises(ValueError, match="n_epochs"):
        SRocketClassifier(n_epochs=0).fit(X, y)
    with pytest.raises(ValueError, match="pop_size"):
        SRocketClassifier(pop_size=2).fit(X, y)

    with pytest.raises(ValueError, match="n_kernels"):
        POCKETClassifier(n_kernels=0).fit(X, y)
    with pytest.raises(ValueError, match="n_iter"):
        POCKETClassifier(n_iter=0).fit(X, y)
    with pytest.raises(ValueError, match="remain_ratio"):
        POCKETClassifier(remain_ratio=0.0).fit(X, y)

    with pytest.raises(ValueError, match="n_kernels"):
        DetachRocketClassifier(n_kernels=0).fit(X, y)
    with pytest.raises(ValueError, match="set_percentage"):
        DetachRocketClassifier(set_percentage=120).fit(X, y)
    with pytest.raises(ValueError, match="smoothing_points"):
        DetachRocketClassifier(smoothing_points=0).fit(X, y)
    with pytest.raises(ValueError, match="multiclass_type"):
        DetachRocketClassifier(multiclass_type="invalid").fit(X, y)

    if _check_soft_dependencies("aeon>=1.6.0", severity="none"):
        with pytest.raises(ValueError, match="voting"):
            DetachRocketEnsembleClassifier(voting="invalid").fit(X, y)
        with pytest.raises(ValueError, match="n_kernels"):
            DetachRocketEnsembleClassifier(n_kernels=0).fit(X, y)

    with pytest.raises(ValueError, match="n_kernels"):
        ClassWiseMiniRocketClassifier(n_kernels=0).fit(X, y)
    with pytest.raises(ValueError, match="max_dilations_per_kernel"):
        ClassWiseMiniRocketClassifier(max_dilations_per_kernel=0).fit(X, y)
    with pytest.raises(ValueError, match="channel_selection"):
        ClassWiseMiniRocketClassifier(channel_selection="invalid").fit(X, y)


def test_numba_thread_cap_respected(monkeypatch):
    """Test n_jobs=-1 does not exceed a capped NUMBA_NUM_THREADS limit."""
    import numba

    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 1)
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=2, n_timepoints=15, random_state=0
    )
    for cls in [
        SRocketClassifier,
        POCKETClassifier,
        DetachRocketClassifier,
        DetachRocketEnsembleClassifier,
        ClassWiseMiniRocketClassifier,
    ]:
        deps = cls.get_class_tag("python_dependencies")
        if deps is not None and not _check_soft_dependencies(deps, severity="none"):
            continue
        clf = cls(**cls._get_test_params(), n_jobs=-1, random_state=0).fit(X, y)
        assert clf._n_jobs == 1
        assert len(clf.predict(X)) == len(y)
