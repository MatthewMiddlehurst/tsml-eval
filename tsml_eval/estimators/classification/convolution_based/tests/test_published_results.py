"""Test the federated ROCKET classifiers against their published results.

The expected values are mean test accuracies over seeds 1 to 5 from the result files
of the reference implementations, for datasets that are distributed with aeon.
"""

import numpy as np
import pytest
from aeon.datasets import load_classification

from tsml_eval.estimators.classification.convolution_based import (
    Fed2RCClassifier,
    FROCKSClassifier,
    MiniFed2RCClassifier,
)

# https://github.com/CasellaJr/Fed2RC results/Fed2RC/IID/1000kernels_results.csv
# and results/DAMI/MiniFed2RC/IID/miniIID924.csv, https://github.com/MatthiasJakobs/FROCKS
# results/frocks_results.csv
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
    (FROCKSClassifier, {"n_kernels": 1000}, "GunPoint", 0.9386666666666666, 0.002),
    (
        FROCKSClassifier,
        {"n_kernels": 1000},
        "ItalyPowerDemand",
        0.9529640427599612,
        0.002,
    ),
]


@pytest.mark.parametrize(
    "classifier, params, dataset, expected, tolerance", PUBLISHED_RESULTS
)
def test_published_results(classifier, params, dataset, expected, tolerance):
    """Test the mean accuracy over seeds 1 to 5 is close to the published value."""
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
