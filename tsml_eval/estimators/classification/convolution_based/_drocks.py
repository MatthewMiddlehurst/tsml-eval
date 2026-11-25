"""Decentralised ROCKET features (DROCKS) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DROCKSClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from scipy.special import expit, xlogy
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, gen_batches

from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _normalise,
    _seeded_rocket_ppv,
    _SeededRocketFeatures,
    _split_clients,
)


class DROCKSClassifier(BaseClassifier):
    """Decentralised ROCKET features (DROCKS) classifier.

    DROCKS [1]_ is a serverless federated version of ROCKET. The training data is
    split between ``n_clients`` simulated clients, which are visited in turn along a
    path in each round. Each kernel is a single ROCKET kernel identified by its seed,
    and only its proportion of positive values (PPV) feature is used.

    Each client fits a linear model, a single layer MLP trained with Adam, on the PPV
    features of the kernels it receives, and adds the ``n_kernels // n_clients``
    kernels with the largest absolute weights to the list of selected kernels. It
    passes the selected kernels, topped up to ``n_kernels`` with kernels not used by
    it or in earlier rounds, on to the next client. Training stops when a round
    selects no new kernel, or after ``n_rounds`` rounds. The model of the last client
    is used to predict.

    This follows the reference implementation [2]_, with three of its bugs fixed. As
    in the reference code, every client in the first round uses the first client's
    kernels, and a class with fewer cases than clients reuses its cases so every
    client gets every class. The final intercept is the mean of the client
    intercepts in the last round, where the reference code sums the intercepts of
    every round and adds only the first entry to every output. Later clients in a
    round no longer also skip the first client's kernels when topping up, which comes
    from list aliasing in the reference code. Clients get each case once, where the
    reference split deals the first case of each class given to each client a second
    time, as the authors fixed in their later Fed2RC code. The published results come
    from a version of the code before commit ``fc06626``, which passes kernels on
    only in the first round.

    Parameters
    ----------
    n_kernels : int, default=5000
        Number of kernels each client receives. 5000 is the setting of the main
        results in [1]_.
    n_clients : int, default=4
        Number of simulated clients the training data is split between.
    n_rounds : int, default=100
        Maximum number of federated training rounds.
    max_iter : int, default=5000
        Maximum number of epochs when fitting each local model.
    n_jobs : int, default=1
        The number of jobs to run in parallel when applying kernels. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to split the data between clients and
        to fit the local models.

    Attributes
    ----------
    seeds_ : np.ndarray
        The seeds of the kernels in the final model.
    coef_ : np.ndarray
        The weights of the final model, of shape ``(len(seeds_), 1)`` for binary
        problems and ``(len(seeds_), n_classes_)`` otherwise.
    intercept_ : np.ndarray
        The intercepts of the final model, of shape ``(1,)`` for binary problems and
        ``(n_classes_,)`` otherwise.
    n_rounds_ : int
        The number of rounds run.

    See Also
    --------
    FROCKSClassifier, Fed2RCClassifier

    References
    ----------
    .. [1] Casella, Bruno, Matthias Jakobs, Marco Aldinucci, and Sebastian
       Buschjäger. "Decentralized time series classification with ROCKET
       features." arXiv preprint arXiv:2504.17617, 2025.
    .. [2] https://github.com/CasellaJr/DROCKS

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     DROCKSClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = DROCKSClassifier(n_kernels=40, n_rounds=2, max_iter=50, random_state=0)
    >>> clf.fit(X, y)
    DROCKSClassifier(max_iter=50, n_kernels=40, n_rounds=2, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=5000,
        n_clients=4,
        n_rounds=100,
        max_iter=5000,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_kernels < self.n_clients:
            raise ValueError("n_kernels must be at least n_clients.")
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be at least 1.")

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        y = np.array([self._class_dictionary[label] for label in y])

        n_top = self.n_kernels // self.n_clients
        clients = _split_clients(X, y, self.n_clients, rng, reuse_small_classes=True)
        features = [_SeededRocketFeatures(X[idx], self._n_jobs) for idx in clients]

        seeds = list(range(self.n_kernels))
        selected = []
        selected_set = set()
        previous_selected = None
        used_in_past_rounds = set()
        used_so_far = set()
        for i in range(self.n_rounds):
            intercept = 0
            for c, idx in enumerate(clients):
                used_seeds = seeds

                w, b = _fit_mlp(
                    features[c].transform(used_seeds), y[idx], rng, self.max_iter
                )
                intercept = intercept + b

                # rows of the largest absolute weights over all outputs
                top = np.argsort(-np.abs(w), axis=None)[:n_top] // w.shape[1]
                for t in top:
                    if used_seeds[t] not in selected_set:
                        selected.append(used_seeds[t])
                        selected_set.add(used_seeds[t])

                # every client in the first round uses the first kernels, after that
                # each client passes the selected kernels on to the next, topped up
                # with kernels not used by this client or in earlier rounds
                if i > 0 or c == self.n_clients - 1:
                    used_set = set(used_seeds)
                    new_seeds = []
                    seed = self.n_kernels
                    while len(selected) + len(new_seeds) < self.n_kernels:
                        if seed not in used_set and seed not in used_in_past_rounds:
                            new_seeds.append(seed)
                        seed += 1
                    seeds = selected + new_seeds

                used_so_far.update(used_seeds)

            used_in_past_rounds = set(used_so_far)

            if selected_set == previous_selected:
                break
            previous_selected = set(selected_set)

        self.seeds_ = np.array(used_seeds)
        self.coef_ = w
        self.intercept_ = intercept / self.n_clients
        self.n_rounds_ = i + 1

        return self

    def _predict(self, X):
        Xt = _seeded_rocket_ppv(_normalise(X), self.seeds_, self._n_jobs)
        # as the reference MLP, apply the logistic function in float32 before taking
        # the class, so saturated outputs tie
        proba = expit(Xt @ self.coef_ + self.intercept_)
        if proba.shape[1] == 1:
            return self.classes_[(proba[:, 0] > 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {"n_kernels": 20, "n_rounds": 3, "max_iter": 20}


def _fit_mlp(X, y, rng, max_iter):
    """Fit the reference code's local model, an MLP with no hidden layer.

    This repeats the operations of scikit-learn's ``MLPClassifier.fit`` for the
    reference settings (``hidden_layer_sizes=()``, ``activation="logistic"``,
    ``alpha=0``, Adam with ``learning_rate_init=0.001`` and ``batch_size=2``, other
    parameters at their defaults) in the same order and data types, but without its
    per batch input checks and dispatch, which take most of the time for batches of
    two cases. The fitted weights and the draws from ``rng`` are the same.

    Parameters
    ----------
    X : np.ndarray
        Features of shape ``(n_cases, n_features)``.
    y : np.ndarray
        Class labels.
    rng : np.random.RandomState
        Random number generator for the initial weights and the batch order.
    max_iter : int
        Maximum number of epochs.

    Returns
    -------
    coef : np.ndarray
        The weights, of shape ``(n_features, n_outputs)``.
    intercept : np.ndarray
        The intercepts, of shape ``(n_outputs,)``.
    """
    binarizer = LabelBinarizer().fit(y)
    multiclass = binarizer.y_type_ == "multiclass"
    Y = binarizer.transform(y).astype(bool)
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]

    init_bound = np.sqrt(2.0 / (n_features + n_outputs))
    coef = rng.uniform(-init_bound, init_bound, (n_features, n_outputs))
    intercept = rng.uniform(-init_bound, init_bound, n_outputs)
    params = [coef.astype(X.dtype, copy=False), intercept.astype(X.dtype, copy=False)]

    # the reference learning rate and alpha, and MLPClassifier's defaults otherwise
    learning_rate_init = 0.001
    alpha = 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    tol = 1e-4
    ms = [np.zeros_like(param) for param in params]
    vs = [np.zeros_like(param) for param in params]
    t = 0

    eps = np.finfo(X.dtype).eps
    batches = list(gen_batches(n_samples, np.clip(2, 1, n_samples)))

    sample_idx = np.arange(n_samples, dtype=int)
    best_loss = np.inf
    no_improvement_count = 0
    for _ in range(max_iter):
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        sample_idx = sample_idx[indices]

        accumulated_loss = 0.0
        for batch_slice in batches:
            batch_idx = sample_idx[batch_slice]
            X_batch = X[batch_idx]
            y_batch = Y[batch_idx]
            sw_sum = X_batch.shape[0]

            # forward pass with a logistic or softmax output
            output = X_batch @ params[0]
            output += params[1]
            if multiclass:
                tmp = output - output.max(axis=1)[:, np.newaxis]
                np.exp(tmp, out=output)
                output /= output.sum(axis=1)[:, np.newaxis]
                y_prob = np.clip(output, eps, 1 - eps)
                loss = -xlogy(y_batch, y_prob).mean(axis=0).sum()
            else:
                expit(output, out=output)
                y_prob = np.clip(output, eps, 1 - eps)
                loss = (
                    -(xlogy(y_batch, y_prob) + xlogy(1 - y_batch, 1 - y_prob))
                    .mean(axis=0)
                    .sum()
                )
            s = params[0].ravel()
            loss += (0.5 * alpha) * np.dot(s, s) / sw_sum

            # gradients
            deltas = output - y_batch
            coef_grad = X_batch.T @ deltas
            coef_grad += alpha * params[0]
            coef_grad /= sw_sum
            intercept_grad = np.sum(deltas, axis=0) / sw_sum
            accumulated_loss += loss * (batch_slice.stop - batch_slice.start)

            # Adam update, with the moments updated in place
            t += 1
            learning_rate = (
                learning_rate_init * np.sqrt(1 - beta_2**t) / (1 - beta_1**t)
            )
            for param, m, v, grad in zip(params, ms, vs, (coef_grad, intercept_grad)):
                m *= beta_1
                m += (1 - beta_1) * grad
                v *= beta_2
                v += (1 - beta_2) * (grad**2)
                param += -learning_rate * m / (np.sqrt(v) + epsilon)

        # stop when the training loss has not improved by tol for 10 epochs
        epoch_loss = accumulated_loss / n_samples
        if epoch_loss > best_loss - tol:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        if no_improvement_count > 10:
            break

    if not all(np.isfinite(param).all() for param in params):
        raise ValueError(
            "Solver produced non-finite parameter weights. The input data may"
            " contain large values and need to be preprocessed."
        )
    return params[0], params[1]
