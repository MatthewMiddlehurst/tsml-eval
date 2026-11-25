"""Test the convolution-based ROCKET classifiers against their published results.

The expected values are published mean test accuracies from the papers and
reference result files on datasets distributed with aeon, evaluated with
reduced kernel counts so the test suite runs quickly. FROCKS, DROCKS, SPROCKET,
and HIT-ROCKET are not included, as they fix bugs in the reference code that
produced the published results.
"""

import numpy as np
import pytest
from aeon.datasets import load_classification
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.classification.convolution_based import (
    ClassWiseMiniRocketClassifier,
    DetachRocketClassifier,
    DetachRocketEnsembleClassifier,
    Fed2RCClassifier,
    FTFVCClassifier,
    KGMTPClassifier,
    MiniFed2RCClassifier,
    POCKETClassifier,
    SelFRocketClassifier,
    SRocketClassifier,
)

# Published accuracy references:
# - Federated ROCKET: https://github.com/CasellaJr/Fed2RC
#   results/Fed2RC/IID/1000kernels_results.csv and
#   results/DAMI/MiniFed2RC/IID/miniIID924.csv
# - S-ROCKET & POCKET: Chen et al. (2024) Table 9 (GunPoint 1.0000,
#   ItalyPowerDemand 0.9682 / 0.9688, ArrowHead 0.8177 / 0.8183)
# - Detach-ROCKET: Uribarri et al. (2024) Table 2 & Table 3 (ArrowHead 0.8291,
#   GunPoint 0.9997, ItalyPowerDemand 0.9679)
# - Detach-ROCKET Ensemble: Uribarri/Solana et al. (2024) BasicMotions 1.0000
# - Class-wise MiniROCKET: Lo et al. (2026) Table 1 & Table 2 (BasicMotions
#   1.0000 for CW, ECS CW+L, and ECP CW+L)
# - FT-FVC: He et al. (2023) Table 1 (GunPoint 1.0000, ItalyPowerDemand 0.9687,
#   ArrowHead 0.8526)
# - KG-MTP: https://github.com/WangPanJie2024/KG-MTP
#   repository UCR results (GunPoint 0.9978, ItalyPowerDemand 0.9610,
#   ArrowHead 0.8990)
# - SelF-ROCKET: https://github.com/ANR-MYEL/SelF-Rocket
#   results/Mean_perf_SR_UCR112_fpc5000_nr10.csv and
#   results/Mean_perf_HSR_UCR112_fpc5000_nr10.csv (GunPoint 0.9938 / 0.9982,
#   ItalyPowerDemand 0.9637 / 0.9615)
PUBLISHED_RESULTS = [
    (Fed2RCClassifier, {"n_kernels": 1000}, "GunPoint", 0.98, 0.002),
    (Fed2RCClassifier, {"n_kernels": 1000}, "ArrowHead", 0.8971428571428571, 0.002),
    (
        Fed2RCClassifier,
        {"n_kernels": 1000},
        "ItalyPowerDemand",
        0.9634596695821186,
        0.002,
    ),
    (MiniFed2RCClassifier, {"n_kernels": 924}, "GunPoint", 0.98, 0.002),
    (SRocketClassifier, {"n_kernels": 1000, "n_epochs": 300}, "GunPoint", 1.0, 0.015),
    (
        SRocketClassifier,
        {"n_kernels": 1000, "n_epochs": 300},
        "ItalyPowerDemand",
        0.9682,
        0.015,
    ),
    (
        POCKETClassifier,
        {"n_kernels": 1000, "remain_ratio": 0.5, "k": 1.0},
        "GunPoint",
        1.0,
        0.01,
    ),
    (
        POCKETClassifier,
        {"n_kernels": 1000, "remain_ratio": 0.5, "k": 1.0},
        "ItalyPowerDemand",
        0.9688,
        0.01,
    ),
    (
        POCKETClassifier,
        {"n_kernels": 1000, "remain_ratio": 0.2447},
        "ArrowHead",
        0.8183,
        0.02,
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 1000, "set_percentage": 10},
        "GunPoint",
        0.9997,
        0.01,
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 1000, "set_percentage": 10},
        "ItalyPowerDemand",
        0.9679,
        0.01,
    ),
    (
        DetachRocketClassifier,
        {"n_kernels": 1000, "set_percentage": 10},
        "ArrowHead",
        0.8291,
        0.025,
    ),
    (
        DetachRocketEnsembleClassifier,
        {"n_estimators": 5, "n_kernels": 840},
        "BasicMotions",
        1.0,
        0.01,
    ),
    (
        ClassWiseMiniRocketClassifier,
        {"n_kernels": 1680, "channel_selection": "ecp"},
        "BasicMotions",
        1.0,
        0.01,
    ),
    (
        ClassWiseMiniRocketClassifier,
        {"n_kernels": 1680, "channel_selection": "ecs"},
        "BasicMotions",
        1.0,
        0.01,
    ),
    (
        ClassWiseMiniRocketClassifier,
        {"n_kernels": 1680, "channel_selection": None},
        "BasicMotions",
        1.0,
        0.01,
    ),
    (FTFVCClassifier, {"n_kernels": 1000}, "GunPoint", 1.0, 0.01),
    (FTFVCClassifier, {"n_kernels": 1000}, "ItalyPowerDemand", 0.9687, 0.01),
    (FTFVCClassifier, {"n_kernels": 1000}, "ArrowHead", 0.8526, 0.02),
    (
        KGMTPClassifier,
        {"n_features": 5000},
        "GunPoint",
        0.9977777777777778,
        0.01,
    ),
    (
        KGMTPClassifier,
        {"n_features": 5000},
        "ItalyPowerDemand",
        0.9609977324263039,
        0.01,
    ),
    (
        KGMTPClassifier,
        {"n_features": 5000},
        "ArrowHead",
        0.8990476190476191,
        0.02,
    ),
    (
        SelFRocketClassifier,
        {"n_kernels": 1000, "n_features_per_split": 1000, "n_runs": 5},
        "GunPoint",
        0.9937777777777778,
        0.01,
    ),
    (
        SelFRocketClassifier,
        {"n_kernels": 1000, "n_features_per_split": 1000, "n_runs": 5},
        "ItalyPowerDemand",
        0.9636540330417882,
        0.01,
    ),
    (
        SelFRocketClassifier,
        {
            "n_kernels": 1000,
            "n_features_per_split": 1000,
            "n_runs": 5,
            "only_mix": True,
        },
        "GunPoint",
        0.9982222222222222,
        0.01,
    ),
    (
        SelFRocketClassifier,
        {
            "n_kernels": 1000,
            "n_features_per_split": 1000,
            "n_runs": 5,
            "only_mix": True,
        },
        "ItalyPowerDemand",
        0.9614512471655328,
        0.01,
    ),
]


@pytest.mark.parametrize(
    "classifier, params, dataset, expected, tolerance", PUBLISHED_RESULTS
)
def test_published_results(classifier, params, dataset, expected, tolerance):
    """Test the mean accuracy over seeds 1 to 5 is close to the published value."""
    deps = classifier.get_class_tag("python_dependencies")
    if deps is not None and not _check_soft_dependencies(deps, severity="none"):
        pytest.skip("skip test if required soft dependency not available")
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    accuracies = [
        np.mean(
            classifier(random_state=seed, **params)
            .fit(X_train, y_train)
            .predict(X_test)
            == y_test
        )
        for seed in range(1, 6)
    ]

    assert np.mean(accuracies) == pytest.approx(expected, abs=tolerance)
