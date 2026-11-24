"""Test the selection ROCKET classifiers against outputs of the reference code.

The expected values come from running the reference implementations
(https://github.com/salehinejad/srocket, https://github.com/ShaowuChen/POCKET,
https://github.com/gon-uri/detach_rocket, and
https://github.com/mlgig/ChannelSelectionMTSC) with aeon's Rocket, MiniRocket,
and MultiRocket on datasets distributed with aeon, with labels encoded in the
order of ``classes_``. Kernel counts and iteration budgets are reduced to keep
the test suite fast.
"""

import numpy as np
import pytest
from aeon.datasets import load_classification

from tsml_eval.estimators.classification.convolution_based import (
    ClassWiseMiniRocketClassifier,
    DetachRocketClassifier,
    DetachRocketEnsembleClassifier,
    POCKETClassifier,
    SRocketClassifier,
)

# classifier, params, dataset, seed, expected_acc, expected_attrs
REFERENCE_RESULTS = [
    (
        SRocketClassifier,
        {"n_kernels": 100, "n_epochs": 50, "pop_size": 8, "init_population": "ones"},
        "GunPoint",
        1,
        1.0,
        {"n_selected_kernels_": 38, "best_cost_": 0.44},
    ),
    (
        SRocketClassifier,
        {"n_kernels": 100, "n_epochs": 50, "pop_size": 8, "init_population": "ones"},
        "ArrowHead",
        2,
        0.6685714285714286,
        {"n_selected_kernels_": 22, "best_cost_": 0.6088888888888888},
    ),
    (
        SRocketClassifier,
        {
            "n_kernels": 100,
            "n_epochs": 50,
            "pop_size": 8,
            "init_population": "half_bernoulli",
        },
        "GunPoint",
        2,
        1.0,
        {"n_selected_kernels_": 37, "best_cost_": 0.41},
    ),
    (
        POCKETClassifier,
        {"n_kernels": 100, "remain_ratio": 0.5, "n_iter": 25, "stage2": False},
        "GunPoint",
        2,
        0.98,
        {"k_": 0.01, "n_remaining_kernels_": 50},
    ),
    (
        POCKETClassifier,
        {"n_kernels": 100, "remain_ratio": 0.5, "n_iter": 25, "stage2": True},
        "ArrowHead",
        2,
        0.8,
        {"k_": 1.0, "n_remaining_kernels_": 50},
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 100, "trade_off": 0.1, "n_steps": 50},
        "GunPoint",
        1,
        1.0,
        {"optimal_alpha_": 3.359818286283774, "n_features_selected": 36},
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 100, "trade_off": 0.1, "n_steps": 50},
        "ArrowHead",
        2,
        0.7714285714285715,
        {"optimal_alpha_": 37.92690190732238, "n_features_selected": 23},
    ),
    (
        DetachRocketEnsembleClassifier,
        {
            "n_estimators": 3,
            "n_kernels": 168,
            "n_steps": 30,
            "rocket_transform": "minirocket",
        },
        "BasicMotions",
        1,
        1.0,
        {
            "channel_relevance_": [
                0.26505706606849455,
                0.1292565945531882,
                0.11790710701988004,
                0.1429053271068189,
                0.16551493847456045,
                0.1793589667770579,
            ]
        },
    ),
    (
        DetachRocketEnsembleClassifier,
        {
            "n_estimators": 3,
            "n_kernels": 672,
            "n_steps": 30,
            "rocket_transform": "multirocket",
        },
        "BasicMotions",
        1,
        1.0,
        {
            "channel_relevance_": [
                0.22388312916337794,
                0.19843667870873513,
                0.17180879850009784,
                0.09946413565649476,
                0.14386587570871764,
                0.16254138226257678,
            ]
        },
    ),
    (
        ClassWiseMiniRocketClassifier,
        {"n_kernels": 168, "channel_selection": "ecp"},
        "BasicMotions",
        1,
        1.0,
        {"channels_selected_": [[0, 1], [0, 1], [0, 1], [0, 1]]},
    ),
    (
        ClassWiseMiniRocketClassifier,
        {"n_kernels": 168, "channel_selection": "ecs"},
        "BasicMotions",
        1,
        1.0,
        {"channels_selected_": [[0, 1], [0, 1], [0, 1], [0, 1]]},
    ),
]


@pytest.mark.parametrize(
    "classifier, params, dataset, seed, expected_acc, expected_attrs",
    REFERENCE_RESULTS,
)
def test_reference_results(
    classifier, params, dataset, seed, expected_acc, expected_attrs
):
    """Test a fit matches the reference code's accuracy and learned attributes."""
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    clf = classifier(random_state=seed, **params).fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) == pytest.approx(expected_acc)

    for attr, expected_val in expected_attrs.items():
        if attr == "n_features_selected":
            assert int(clf.feature_mask_.sum()) == expected_val
        elif attr == "channel_relevance_":
            np.testing.assert_allclose(clf.channel_relevance_, expected_val, rtol=1e-6)
        elif attr == "channels_selected_":
            assert [c.tolist() for c in clf.channels_selected_] == expected_val
        elif isinstance(expected_val, float):
            assert getattr(clf, attr) == pytest.approx(expected_val)
        else:
            assert getattr(clf, attr) == expected_val
