"""Shared utilities for the federated ROCKET classifiers.

The federated classifiers simulate a federation on a single machine. The training
data is split between clients as in the reference implementations, and each client
only uses its own cases to fit its local models.
"""

__maintainer__ = ["MatthewMiddlehurst"]

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based._rocket import (
    _apply_kernels,
    _generate_kernels,
)
from numba import get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits


def _split_clients(X, y, n_clients, rng, repeat_initial):
    """Split case indices between clients using the reference uniform split.

    Every client first receives one case of each class. The remaining cases are then
    dealt in a random order, each to the client with the fewest cases.

    Parameters
    ----------
    X : np.ndarray
        The series, of shape ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray
        Integer class labels.
    n_clients : int
        Number of clients.
    rng : np.random.RandomState
        Random number generator.
    repeat_initial : bool
        If True, as in the FROCKS and DROCKS code, the cases handed out first are
        dealt again along with all other cases, so they appear twice, and a class with
        fewer cases than clients reuses its cases. If False, as in the Fed2RC code, a
        case is skipped if a case with the same series and label has already been
        given out, and a class with fewer cases than clients only goes to the first
        clients. If no class has at least ``n_clients`` cases, some clients would get
        no cases, and the cases are instead shuffled and split into ``n_clients``
        equal parts, dropping any remainder, as the Fed2RC code does for such
        datasets.

    Returns
    -------
    clients : list of np.ndarray
        The case indices of each client.
    """
    if len(y) < n_clients:
        raise ValueError(
            f"There must be at least n_clients={n_clients} training cases, found "
            f"{len(y)}."
        )

    if not repeat_initial and np.bincount(y).max() < n_clients:
        shuffled = rng.permutation(len(y))
        n = len(y) // n_clients
        return [shuffled[i * n : (i + 1) * n] for i in range(n_clients)]

    shuffled = rng.permutation(len(y))

    clients = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)

        n = n_clients if repeat_initial else min(n_clients, len(indices))
        for i in range(n):
            clients[i].append(indices[i % len(indices)])

    # the Fed2RC code compares values, so duplicated cases are only used once.
    # adding 0.0 makes -0.0 and 0.0 equal, as they are for np.array_equal
    if not repeat_initial:
        keys = [(y[i], (X[i] + 0.0).tobytes()) for i in range(len(y))]
        dealt = {keys[i] for client in clients for i in client}

    sizes = [len(client) for client in clients]
    for i in shuffled:
        if not repeat_initial:
            if keys[i] in dealt:
                continue
            dealt.add(keys[i])

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
        The float32 PPV feature of each kernel, of shape ``(n_cases, len(seeds))``.
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
    return np.ascontiguousarray(Xt[:, ::2])


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


def _client_lambda(X, y, A, A_c, b, b_c):
    """Select the ridge lambda for one client, as in the Fed2RC protocol.

    Fits the ridge solution from the other clients' sums for each lambda in a grid,
    and keeps the lambda with the best accuracy on the client's own data, with ties
    going to the larger lambda.

    One eigendecomposition gives the solution for every lambda. With
    ``A - A_c = V diag(e) V^T``, the weights are ``V diag(1 / (e + lambda)) V^T b``,
    and as ``V`` is orthogonal the norm of each class column is the norm of
    ``diag(1 / (e + lambda)) V^T b``, so the client's scores only need ``X V``.

    Returns
    -------
    weighted_lambda : float
        The selected lambda times its accuracy.
    accuracy : float
        The accuracy of the selected lambda.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A - A_c)
    projected_b = eigenvectors.T @ (b - b_c)
    projected_X = X @ eigenvectors

    best_accuracy = 0
    best_lambda = 0
    for ridge_lambda in np.logspace(-3, 0, 100):
        coef = projected_b / (eigenvalues + ridge_lambda)[:, None]
        coef = coef / (np.linalg.norm(coef, axis=0, keepdims=True) + 1e-10)
        accuracy = np.mean(np.argmax(projected_X @ coef, axis=1) == y)
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and ridge_lambda > best_lambda
        ):
            best_accuracy = accuracy
            best_lambda = ridge_lambda

    return best_lambda * best_accuracy, best_accuracy


def _federated_ridge(client_features, client_y, n_classes, n_jobs=1):
    """Fit the Fed2RC closed-form ridge classifier.

    Each client shares ``X^T X`` and ``X^T Y`` for its features ``X`` and one-hot
    labels ``Y``, and the server sums them. Each client then selects a lambda with
    ``_client_lambda``, and the global lambda is the mean of the client lambdas
    weighted by their accuracies.

    Parameters
    ----------
    client_features : list of np.ndarray
        The float32 features of each client, each of shape ``(n_cases_c,
        n_features)``. As in the reference code, ``X^T X`` is computed in float32.
    client_y : list of np.ndarray
        The integer class labels of each client.
    n_classes : int
        Number of classes.
    n_jobs : int, default=1
        The number of clients to select lambda for in parallel.

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
    A = np.zeros(A_clients[0].shape)
    b = np.zeros(b_clients[0].shape)
    for A_c, b_c in zip(A_clients, b_clients):
        A += A_c
        b += b_c

    # eigh results change with the number of BLAS threads, so each decomposition
    # uses one thread, and clients run in parallel instead
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(A_clients))) as pool:
            selected = list(
                pool.map(
                    _client_lambda,
                    client_features,
                    client_y,
                    repeat(A),
                    A_clients,
                    repeat(b),
                    b_clients,
                )
            )

    weighted_lambdas, accuracies = zip(*selected)
    ridge_lambda = np.sum(weighted_lambdas) / np.sum(accuracies)
    W = np.linalg.solve(A + ridge_lambda * np.eye(A.shape[0]), b)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
    return W, ridge_lambda
