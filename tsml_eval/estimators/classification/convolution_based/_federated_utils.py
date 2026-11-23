"""Shared utilities for the federated ROCKET classifiers.

The federated classifiers simulate a federation on a single machine. The training
data is split between clients as in the reference implementations, and each client
only uses its own cases to fit its local models.
"""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based._rocket import (
    _apply_kernels,
    _generate_kernels,
)
from numba import get_num_threads, set_num_threads


def _split_clients(y, n_clients, rng, repeat_initial):
    """Split case indices between clients using the reference uniform split.

    Every client first receives one case of each class. The remaining cases are then
    dealt in a random order, each to the client with the fewest cases.

    Parameters
    ----------
    y : np.ndarray
        Integer class labels.
    n_clients : int
        Number of clients.
    rng : np.random.RandomState
        Random number generator.
    repeat_initial : bool
        If True, as in the FROCKS and DROCKS code, the cases handed out first are
        dealt again along with all other cases, so they appear twice, and a class with
        fewer cases than clients reuses its cases. If False, as in the Fed2RC code,
        each case is used once, and a class with fewer cases than clients only goes to
        the first clients.

    Returns
    -------
    clients : list of np.ndarray
        The case indices of each client.
    """
    shuffled = rng.permutation(len(y))

    clients = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)

        n = n_clients if repeat_initial else min(n_clients, len(indices))
        for i in range(n):
            clients[i].append(indices[i % len(indices)])

    dealt = set() if repeat_initial else {i for client in clients for i in client}
    sizes = [len(client) for client in clients]
    for i in shuffled:
        if i in dealt:
            continue
        smallest = int(np.argmin(sizes))
        clients[smallest].append(i)
        sizes[smallest] += 1

    return [np.array(client, dtype=int) for client in clients]


def _normalise(X):
    """Normalise each series and channel, as the ROCKET transform does."""
    return np.asarray(Normalizer().fit_transform(X), dtype=np.float32)


def _seeded_rocket_ppv(X, seeds, n_jobs):
    """PPV features of single ROCKET kernels, one kernel per seed.

    Each kernel is the one ``Rocket(n_kernels=1, random_state=seed)`` generates for
    the series length and channels of ``X``, so a seed identifies its kernel.

    Parameters
    ----------
    X : np.ndarray
        Normalised float32 series of shape ``(n_cases, n_channels, n_timepoints)``.
    seeds : list of int
        The kernel seeds.
    n_jobs : int
        The number of threads used to apply the kernels.

    Returns
    -------
    ppv : np.ndarray
        The PPV feature of each kernel, of shape ``(n_cases, len(seeds))``.
    """
    kernels = [_generate_kernels(X.shape[2], 1, X.shape[1], int(s)) for s in seeds]
    kernels = tuple(np.concatenate(parts) for parts in zip(*kernels))

    prev_threads = get_num_threads()
    set_num_threads(n_jobs)
    try:
        Xt = _apply_kernels(X, kernels)
    finally:
        set_num_threads(prev_threads)

    # the transform outputs the PPV and max of each kernel in turn
    return Xt[:, ::2].astype(np.float64)


class _SeededRocketFeatures:
    """Cache of seeded ROCKET kernel PPV features for one client's series."""

    def __init__(self, X, n_jobs):
        self.X = _normalise(X)
        self.n_jobs = n_jobs
        self._features = {}

    def transform(self, seeds):
        seeds = [int(s) for s in seeds]
        missing = [s for s in dict.fromkeys(seeds) if s not in self._features]
        if missing:
            ppv = _seeded_rocket_ppv(self.X, missing, self.n_jobs)
            self._features.update(zip(missing, ppv.T))
        return np.column_stack([self._features[s] for s in seeds])


def _normalised_ridge_weights(eigenvalues, eigenvectors, b, ridge_lambda):
    """Solve ``(A + lambda I) W = b`` from an eigendecomposition of ``A``."""
    W = eigenvectors @ ((eigenvectors.T @ b) / (eigenvalues + ridge_lambda)[:, None])
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)


def _federated_ridge(client_features, client_y, n_classes):
    """Fit the Fed2RC closed-form ridge classifier.

    Each client shares ``X^T X`` and ``X^T Y`` for its features ``X`` and one-hot
    labels ``Y``, and the server sums them. To select lambda, each client fits the
    ridge solution from the other clients' sums for each lambda in a grid, and keeps
    the lambda with the best accuracy on its own data, with ties going to the larger
    lambda. The global lambda is the mean of the client lambdas weighted by their
    accuracies.

    Parameters
    ----------
    client_features : list of np.ndarray
        The features of each client, each of shape ``(n_cases_c, n_features)``.
    client_y : list of np.ndarray
        The integer class labels of each client.
    n_classes : int
        Number of classes.

    Returns
    -------
    W : np.ndarray
        The ridge weights of shape ``(n_features, n_classes)``, with each class column
        scaled to unit norm. Predictions are the argmax of ``X @ W``.
    ridge_lambda : float
        The selected regularisation parameter.
    """
    A_clients = [X.T @ X for X in client_features]
    b_clients = [X.T @ np.eye(n_classes)[y] for X, y in zip(client_features, client_y)]
    A = np.sum(A_clients, axis=0)
    b = np.sum(b_clients, axis=0)

    weighted_lambdas = []
    accuracies = []
    for X, y, A_c, b_c in zip(client_features, client_y, A_clients, b_clients):
        # one eigendecomposition per client gives the solution for every lambda,
        # rather than solving the system again for each one
        eigenvalues, eigenvectors = np.linalg.eigh(A - A_c)
        best_accuracy = 0
        best_lambda = 0
        for ridge_lambda in np.logspace(-3, 0, 100):
            W = _normalised_ridge_weights(
                eigenvalues, eigenvectors, b - b_c, ridge_lambda
            )
            accuracy = np.mean(np.argmax(X @ W, axis=1) == y)
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and ridge_lambda > best_lambda
            ):
                best_accuracy = accuracy
                best_lambda = ridge_lambda

        weighted_lambdas.append(best_lambda * best_accuracy)
        accuracies.append(best_accuracy)

    ridge_lambda = np.sum(weighted_lambdas) / np.sum(accuracies)
    W = np.linalg.solve(A + ridge_lambda * np.eye(A.shape[0]), b)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
    return W, ridge_lambda
