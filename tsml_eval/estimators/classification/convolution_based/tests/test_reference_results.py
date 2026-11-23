"""Test the federated ROCKET classifiers against outputs of the reference code.

The expected values come from running the reference implementations with aeon's
Rocket and MiniRocket in place of sktime's, on datasets distributed with aeon, with
labels encoded in the order of ``classes_``. The client test split only used for
logging was removed, as in DROCKS it draws from the same random state as the MLP.
DROCKS was run with ``max_iter`` lowered from 5000 to keep the test fast.
"""

import numpy as np
import pytest
from aeon.datasets import load_classification

from tsml_eval.estimators.classification.convolution_based import (
    DROCKSClassifier,
    Fed2RCClassifier,
    FROCKSClassifier,
    MiniFed2RCClassifier,
)

# classifier, params, dataset, seed, accuracy, rounds, kernels, lambda
REFERENCE_RESULTS = [
    (FROCKSClassifier, {"n_kernels": 100}, "GunPoint", 1, 0.8, 5, 55, None),
    (
        FROCKSClassifier,
        {"n_kernels": 100},
        "GunPoint",
        2,
        0.7733333333333333,
        5,
        51,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "GunPoint",
        1,
        0.7266666666666667,
        5,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "GunPoint",
        2,
        0.49333333333333335,
        4,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "ArrowHead",
        1,
        0.6228571428571429,
        5,
        40,
        None,
    ),
    (
        DROCKSClassifier,
        {"n_kernels": 40, "max_iter": 100},
        "ArrowHead",
        2,
        0.52,
        7,
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


@pytest.mark.parametrize(
    "classifier, params, dataset, seed, accuracy, rounds, kernels, ridge_lambda",
    REFERENCE_RESULTS,
)
def test_reference_results(
    classifier, params, dataset, seed, accuracy, rounds, kernels, ridge_lambda
):
    """Test a fit matches the reference code's accuracy, rounds, kernels and lambda."""
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    clf = classifier(random_state=seed, **params).fit(X_train, y_train)

    assert np.mean(clf.predict(X_test) == y_test) == pytest.approx(accuracy)
    if rounds is not None:
        assert clf.n_rounds_ == rounds
        assert len(clf.seeds_) == kernels
    if ridge_lambda is not None:
        assert clf.lambda_ == pytest.approx(ridge_lambda)
