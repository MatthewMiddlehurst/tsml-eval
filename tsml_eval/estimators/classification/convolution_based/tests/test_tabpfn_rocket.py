"""Tests for the RocketPFN and MASHT TabPFN-based ROCKET classifiers."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    MASHTClassifier,
    RocketPFNClassifier,
)
from tsml_eval.estimators.classification.convolution_based._masht import (
    _adaptive_feature_budget,
)

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies(["tabpfn>=8.1.0", "torch>=2.5"], severity="none"),
    reason="skip test if required soft dependency not available",
)


@pytest.mark.parametrize("n_channels, n_timepoints", [(1, 6), (3, 25)])
def test_rocket_pfn_group_averaging_and_short_series(n_channels, n_timepoints):
    """Test RocketPFN averages group probabilities and pads series shorter than 11."""
    X, y = make_example_3d_numpy(
        n_cases=18,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=3,
        random_state=0,
    )
    # Non-zero-indexed string labels verify class ordering matches classes_
    y = np.array([f"cls_{v + 2}" for v in y])

    clf = RocketPFNClassifier(
        n_groups=3,
        n_kernels=15,
        n_estimators=2,
        model_path="_tiny",
        device="cpu",
        n_jobs=1,
        random_state=0,
    ).fit(X, y)

    assert len(clf.transformers_) == 3
    assert len(clf.classifiers_) == 3

    import torch

    proba = clf.predict_proba(X)
    X_prep = clf._preprocess(X)
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        manual_proba = np.mean(
            [
                c.predict_proba(t.transform(X_prep))
                for t, c in zip(clf.transformers_, clf.classifiers_)
            ],
            axis=0,
        )
    finally:
        torch.set_num_threads(prev_threads)
    np.testing.assert_array_equal(proba, manual_proba)
    np.testing.assert_array_equal(
        clf.predict(X), clf.classes_[np.argmax(proba, axis=1)]
    )
    assert clf.predict(X[:1])[0] == clf.predict(X)[0]


def test_rocket_pfn_exceeding_ten_classes_raises():
    """Test RocketPFN rejects datasets with more than 10 classes."""
    X, _ = make_example_3d_numpy(
        n_cases=22, n_channels=1, n_timepoints=15, random_state=0
    )
    y_eleven = np.arange(22) % 11
    with pytest.raises(ValueError, match="classes"):
        RocketPFNClassifier(
            n_groups=2, n_kernels=10, n_estimators=2, model_path="_tiny", device="cpu"
        ).fit(X, y_eleven)


def test_masht_adaptive_budget_and_bugfixes():
    """Test MASHT adaptive schedule, L=65 dilation fix, small budget, and L<10 pad."""
    assert _adaptive_feature_budget(50) == 10_000
    assert _adaptive_feature_budget(999) == 10_000
    assert _adaptive_feature_budget(1_000) == 2_000
    assert _adaptive_feature_budget(99_999) == 2_000
    assert _adaptive_feature_budget(100_000) == 200

    # At L=65 ((L - 1) / 8 == 8, exact power of two), Hydra uses 4 dilations.
    # Fixed group count at budget 10,000 (component_budget=5,000) is 5000 // 64 = 78,
    # keeping total features at 9,696 <= 10,000 (reference ceil bug gave 104 -> 11,360).
    X65, y65 = make_example_3d_numpy(
        n_cases=16, n_channels=2, n_timepoints=65, random_state=0
    )
    clf65 = MASHTClassifier(
        max_features=10_000,
        n_estimators=2,
        model_path="_tiny",
        device="cpu",
        random_state=0,
    ).fit(X65, y65)
    assert clf65.feature_budget_ == 10_000
    assert clf65.n_hydra_groups_ == 78
    assert clf65.n_transformed_features_ == 9_696
    assert clf65.n_transformed_features_ <= clf65.feature_budget_

    # Small budget (200, as on >= 100,000 cases) on long series (L=300) and short
    # series (L=6 < 10) both clamp cleanly without ZeroDivisionError or ValueError.
    X_long, y_long = make_example_3d_numpy(
        n_cases=14, n_channels=1, n_timepoints=300, random_state=1
    )
    clf_small = MASHTClassifier(
        max_features=200,
        n_estimators=2,
        model_path="_tiny",
        device="cpu",
        random_state=1,
    ).fit(X_long, y_long)
    assert clf_small.n_hydra_groups_ == 1
    assert set(clf_small.predict(X_long[:3])) <= set(y_long)

    X_short, y_short = make_example_3d_numpy(
        n_cases=14, n_channels=2, n_timepoints=6, random_state=2
    )
    clf_short = MASHTClassifier(
        max_features=400,
        n_estimators=2,
        model_path="_tiny",
        device="cpu",
        random_state=2,
    ).fit(X_short, y_short)
    assert clf_short.predict(X_short[:1])[0] == clf_short.predict(X_short)[0]


@pytest.mark.parametrize(
    "cls, bad_kwargs",
    [
        (RocketPFNClassifier, {"n_groups": 0}),
        (RocketPFNClassifier, {"n_kernels": 0}),
        (RocketPFNClassifier, {"n_estimators": 0}),
        (MASHTClassifier, {"max_features": 0}),
        (MASHTClassifier, {"n_estimators": 0}),
    ],
)
def test_tabpfn_rocket_invalid_hyperparameters(cls, bad_kwargs):
    """Test invalid hyperparameter values raise ValueError during fit."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, random_state=0)
    with pytest.raises(ValueError):
        cls(model_path="_tiny", **bad_kwargs).fit(X, y)
