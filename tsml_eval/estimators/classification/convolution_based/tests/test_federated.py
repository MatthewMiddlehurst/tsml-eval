"""Tests for the federated ROCKET classifiers."""

import warnings

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import Rocket
from sklearn.neural_network import MLPClassifier

from tsml_eval.estimators.classification.convolution_based import (
    DROCKSClassifier,
    Fed2RCClassifier,
    FROCKSClassifier,
    MiniFed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._drocks import _fit_mlp
from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _normalise,
    _seeded_rocket_ppv,
    _split_clients,
)


def test_split_clients_repeat_initial():
    """Test the FROCKS and DROCKS split gives every client every class."""
    X = np.random.RandomState(0).random_sample((18, 1, 10))
    y = np.array([0] * 10 + [1] * 6 + [2] * 2)
    clients = _split_clients(X, y, 4, np.random.RandomState(0), repeat_initial=True)

    # one case of each class per client is dealt twice
    assert sum(len(c) for c in clients) == len(y) + 3 * 4
    assert all(set(y[c]) == {0, 1, 2} for c in clients)
    assert set(np.concatenate(clients)) == set(range(len(y)))


def test_split_clients_no_repeat():
    """Test the Fed2RC split uses each case once."""
    X = np.random.RandomState(0).random_sample((18, 1, 10))
    y = np.array([0] * 10 + [1] * 6 + [2] * 2)
    clients = _split_clients(X, y, 4, np.random.RandomState(0), repeat_initial=False)

    assert sorted(np.concatenate(clients)) == list(range(len(y)))
    assert all({0, 1} <= set(y[c]) for c in clients)
    assert max(len(c) for c in clients) - min(len(c) for c in clients) <= 1


def test_split_clients_no_repeat_skips_duplicates():
    """Test the Fed2RC split deals a duplicated case and label only once."""
    X = np.random.RandomState(0).random_sample((18, 1, 10))
    X[5] = X[3]
    X[12] = -0.0 * X[11]
    X[11] = 0.0 * X[11]
    y = np.array([0] * 10 + [1] * 6 + [2] * 2)
    # with one client only one case per class is handed out before the check
    clients = _split_clients(X, y, 1, np.random.RandomState(0), repeat_initial=False)

    used = np.concatenate(clients)
    assert len(used) == len(y) - 2
    assert len(set(used) & {3, 5}) == 1
    assert len(set(used) & {11, 12}) == 1


def test_seeded_rocket_ppv_matches_rocket():
    """Test each seeded kernel is the kernel of a single kernel Rocket transform."""
    X, _ = make_example_3d_numpy(n_cases=5, n_channels=2, random_state=0)
    seeds = [3, 10, 7]

    ppv = _seeded_rocket_ppv(_normalise(X), seeds, 1)

    for i, seed in enumerate(seeds):
        rocket = Rocket(n_kernels=1, random_state=seed).fit(X)
        np.testing.assert_allclose(ppv[:, i], rocket.transform(X)[:, 0])


@pytest.mark.parametrize(
    "classifier",
    [
        DROCKSClassifier(n_kernels=20, n_rounds=3, max_iter=20, random_state=0),
        Fed2RCClassifier(n_kernels=20, n_rounds=3, random_state=0),
        MiniFed2RCClassifier(n_kernels=84, random_state=0),
    ],
)
def test_multiclass(classifier):
    """Test the multiclass classifiers on three classes."""
    X, y = make_example_3d_numpy(n_cases=30, n_labels=3, random_state=0)

    classifier.fit(X, y)

    assert set(classifier.predict(X)) <= set(y)


def test_frocks_multiclass_raises():
    """Test FROCKS only accepts binary problems."""
    X, y = make_example_3d_numpy(n_cases=30, n_labels=3, random_state=0)

    with pytest.raises(ValueError, match="binary"):
        FROCKSClassifier(n_kernels=20, random_state=0).fit(X, y)


def test_split_clients_too_few_cases():
    """Test the split raises a clear error with fewer cases than clients."""
    X = np.random.RandomState(0).random_sample((3, 1, 10))
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="at least n_clients"):
        _split_clients(X, y, 4, np.random.RandomState(0), repeat_initial=False)


@pytest.mark.parametrize("classifier", [FROCKSClassifier, DROCKSClassifier])
def test_no_rounds_raises(classifier):
    """Test FROCKS and DROCKS reject n_rounds below one."""
    X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    with pytest.raises(ValueError, match="n_rounds"):
        classifier(n_kernels=20, n_rounds=0).fit(X, y)


@pytest.mark.parametrize("n_cases, n_classes", [(15, 2), (14, 3)])
def test_fit_mlp_matches_sklearn(n_cases, n_classes):
    """Test the DROCKS local model fit is identical to the reference MLPClassifier."""
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_cases, 30)).astype(np.float32)
    y = np.arange(n_cases) % n_classes

    sklearn_rng = np.random.RandomState(1)
    ours_rng = np.random.RandomState(1)
    for _ in range(2):
        mlp = MLPClassifier(
            hidden_layer_sizes=(),
            activation="logistic",
            alpha=0,
            batch_size=2,
            learning_rate_init=0.001,
            max_iter=200,
            random_state=sklearn_rng,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X, y)
        coef, intercept = _fit_mlp(X, y, ours_rng, 200)

        np.testing.assert_array_equal(coef, mlp.coefs_[0])
        np.testing.assert_array_equal(intercept, mlp.intercepts_[0])
    assert sklearn_rng.randint(1000) == ours_rng.randint(1000)
