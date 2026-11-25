"""Test the convolution-based ROCKET classifiers against reference code outputs.

The expected values come from running the reference implementations with aeon's
Rocket, MiniRocket, and MultiRocket on datasets distributed with aeon, with
labels encoded in the order of ``classes_``. Kernel counts and iteration
budgets are reduced where needed so the test suite runs quickly. For FROCKS,
DROCKS, SPROCKET, and HIT-ROCKET, the reference code was run with the same bug
fixes as the classifiers apply: the intercepts, not dealing cases to clients
twice, and for DROCKS the seed list aliasing; the dilated convolution midpoint
and single-channel multivariate indexing for SPROCKET; and the per-dilation
bias offset for HIT-ROCKET.
"""

import numpy as np
import pytest
from aeon.datasets import load_classification
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    ClassWiseMiniRocketClassifier,
    DetachRocketClassifier,
    DetachRocketEnsembleClassifier,
    DROCKSClassifier,
    Fed2RCClassifier,
    FROCKSClassifier,
    HITRocketClassifier,
    KGMTPClassifier,
    MiniFed2RCClassifier,
    POCKETClassifier,
    SelFRocketClassifier,
    SPROCKETClassifier,
    SRocketClassifier,
)

# Federated ROCKET reference results:
# classifier, params, dataset, seed, accuracy, rounds, kernels, lambda
FEDERATED_REFERENCE_RESULTS = [
    (
        FROCKSClassifier,
        {"n_kernels": 100},
        "GunPoint",
        1,
        0.8266666666666667,
        5,
        52,
        None,
    ),
    (
        FROCKSClassifier,
        {"n_kernels": 100},
        "GunPoint",
        2,
        0.8533333333333334,
        5,
        55,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "GunPoint",
        1,
        0.49333333333333335,
        5,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "GunPoint",
        2,
        0.5066666666666667,
        6,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "ArrowHead",
        1,
        0.6114285714285714,
        5,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "ArrowHead",
        2,
        0.3028571428571429,
        6,
        40,
        None,
    ),
    (
        Fed2RCClassifier,
        {"n_kernels": 100},
        "ArrowHead",
        1,
        0.8342857142857143,
        3,
        58,
        0.6733641100085501,
    ),
    (
        Fed2RCClassifier,
        {"n_kernels": 100},
        "ArrowHead",
        2,
        0.7885714285714286,
        4,
        54,
        0.17491190145594143,
    ),
    (
        MiniFed2RCClassifier,
        {"n_kernels": 84},
        "ArrowHead",
        1,
        0.6914285714285714,
        None,
        None,
        0.4615627690341306,
    ),
    (
        MiniFed2RCClassifier,
        {"n_kernels": 84},
        "ArrowHead",
        2,
        0.7028571428571428,
        None,
        None,
        0.3832004090536262,
    ),
]

# Selection ROCKET reference results:
# https://github.com/salehinejad/srocket, https://github.com/ShaowuChen/POCKET,
# https://github.com/gon-uri/detach_rocket, and
# https://github.com/mlgig/ChannelSelectionMTSC
# classifier, params, dataset, seed, expected_acc, expected_attrs
SELECTION_REFERENCE_RESULTS = [
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
        1.0,
        {"k_": 1.0, "n_remaining_kernels_": 50},
    ),
    (
        POCKETClassifier,
        {"n_kernels": 100, "remain_ratio": 0.5, "n_iter": 25, "stage2": True},
        "ArrowHead",
        2,
        0.8114285714285714,
        {"k_": 10.0, "n_remaining_kernels_": 50},
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 100, "trade_off": 0.1, "n_steps": 50},
        "GunPoint",
        1,
        1.0,
        {"optimal_alpha_": 3.359818286283774, "n_features_selected": 31},
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
        34,
        1.0,
        {
            "channel_relevance_": [
                0.2340997490079166,
                0.1350585295934327,
                0.17855752727521426,
                0.15376726381331068,
                0.16753206559119344,
                0.13098486471893225,
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
        34,
        1.0,
        {
            "channel_relevance_": [
                0.19457995733629577,
                0.1839925111039145,
                0.15052062887272294,
                0.15296864121777232,
                0.1407195360019964,
                0.17721872546729794,
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
    "classifier, params, dataset, seed, accuracy, rounds, kernels, ridge_lambda",
    FEDERATED_REFERENCE_RESULTS,
)
def test_federated_reference_results(
    classifier, params, dataset, seed, accuracy, rounds, kernels, ridge_lambda
):
    """Test federated fits match reference accuracy, rounds, kernels and lambda."""
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    clf = classifier(random_state=seed, **params).fit(X_train, y_train)

    assert np.mean(clf.predict(X_test) == y_test) == pytest.approx(accuracy)
    if rounds is not None:
        assert clf.n_rounds_ == rounds
        assert len(clf.seeds_) == kernels
    if ridge_lambda is not None:
        assert clf.lambda_ == pytest.approx(ridge_lambda)


@pytest.mark.parametrize(
    "classifier, params, dataset, seed, expected_acc, expected_attrs",
    SELECTION_REFERENCE_RESULTS,
)
def test_selection_reference_results(
    classifier, params, dataset, seed, expected_acc, expected_attrs
):
    """Test selection fits match reference accuracy and learned attributes."""
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


# Representation and pooling ROCKET reference results:
# https://github.com/WangPanJie2024/KG-MTP,
# https://github.com/username76543/sprocket_public,
# https://github.com/ANR-MYEL/SelF-Rocket, and
# https://github.com/Einstein-sworder/HitRocket
REPRESENTATION_REFERENCE_RESULTS = [
    (
        KGMTPClassifier,
        {"n_features": 930},
        "GunPoint",
        1,
        1.0,
        {},
    ),
    (
        KGMTPClassifier,
        {"n_features": 930},
        "ArrowHead",
        2,
        0.7771428571428571,
        {},
    ),
    (
        SPROCKETClassifier,
        {"n_kernels": 50, "distance": "euclidean"},
        "GunPoint",
        1,
        0.98,
        {},
    ),
    (
        SPROCKETClassifier,
        {"n_kernels": 50, "distance": "euclidean"},
        "ArrowHead",
        2,
        0.7714285714285715,
        {},
    ),
    (
        SPROCKETClassifier,
        {"n_kernels": 50, "distance": "msm"},
        "BasicMotions",
        1,
        1.0,
        {},
    ),
    (
        SelFRocketClassifier,
        {"n_kernels": 200, "n_features_per_split": 200, "n_runs": 3},
        "GunPoint",
        1,
        0.9933333333333333,
        {"selected_combination_": "PPV_MIX"},
    ),
    (
        SelFRocketClassifier,
        {"n_kernels": 200, "n_features_per_split": 200, "n_runs": 3},
        "ArrowHead",
        2,
        0.76,
        {"selected_combination_": "LSPV_MIX"},
    ),
    (
        SelFRocketClassifier,
        {
            "n_kernels": 200,
            "n_features_per_split": 200,
            "n_runs": 3,
            "only_mix": True,
        },
        "ArrowHead",
        1,
        0.8114285714285714,
        {"selected_combination_": "PPV_MIX"},
    ),
    (
        HITRocketClassifier,
        {"n_samples": 4, "dilations": (1, 2, 3, 4)},
        "GunPoint",
        1,
        0.9933333333333333,
        {},
    ),
    (
        HITRocketClassifier,
        {"n_samples": 4, "dilations": (1, 2, 3, 4)},
        "ArrowHead",
        2,
        0.8685714285714285,
        {},
    ),
]


@pytest.mark.parametrize(
    "classifier, params, dataset, seed, expected_acc, expected_attrs",
    REPRESENTATION_REFERENCE_RESULTS,
)
def test_representation_reference_results(
    classifier, params, dataset, seed, expected_acc, expected_attrs
):
    """Test representation fits match reference accuracy and learned attributes."""
    if classifier is HITRocketClassifier and not _check_soft_dependencies(
        "torch>=2.0", severity="none"
    ):
        pytest.skip("torch is required for HITRocketClassifier")

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    clf = classifier(random_state=seed, **params).fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) == pytest.approx(expected_acc)

    for attr, expected_val in expected_attrs.items():
        assert getattr(clf, attr) == expected_val
