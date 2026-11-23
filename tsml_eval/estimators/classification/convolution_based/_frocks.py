"""Federated ROCKET features (FROCKS) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["FROCKSClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _normalise,
    _seeded_rocket_ppv,
    _SeededRocketFeatures,
    _split_clients,
)


class FROCKSClassifier(BaseClassifier):
    """Federated ROCKET features (FROCKS) classifier.

    FROCKS [1]_ is a server-client federated version of ROCKET for binary
    classification. The training data is split between ``n_clients`` simulated
    clients. Each kernel is a single ROCKET kernel identified by its seed, and only
    its proportion of positive values (PPV) feature is used.

    In the first round each client fits a logistic regression on the PPV features of
    its own ``n_kernels`` kernels. Each client sends the ``n_kernels // n_clients``
    kernels with the largest absolute weights to the server, along with their weights
    and its intercept. The server averages the weights of kernels sent by more than
    one client, and the selected kernels are used by every client in the next round.
    Training stops when the averaged weights do not change between rounds, or after
    ``n_rounds`` rounds. The aggregated linear model of the last round is used to
    predict.

    This follows the reference implementation [2]_, including its split of the data
    between clients. The intercept also carries over between rounds as in the
    reference code: each round it is set to the previous intercept plus the sum of
    the client intercepts, divided by the number of clients.

    Parameters
    ----------
    n_kernels : int, default=1000
        Number of kernels each client starts with.
    n_clients : int, default=4
        Number of simulated clients the training data is split between.
    n_rounds : int, default=100
        Maximum number of federated training rounds.
    n_jobs : int, default=1
        The number of jobs to run in parallel when applying kernels. ``-1`` means
        using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to split the data between clients.

    Attributes
    ----------
    seeds_ : np.ndarray
        The seeds of the kernels in the final model.
    coef_ : np.ndarray
        The averaged weight of each kernel in the final model.
    intercept_ : float
        The intercept of the final model.
    n_rounds_ : int
        The number of rounds run.

    See Also
    --------
    DROCKSClassifier, Fed2RCClassifier

    References
    ----------
    .. [1] Casella, Bruno, Matthias Jakobs, Marco Aldinucci, and Sebastian
       Buschjäger. "Federated time series classification with ROCKET features."
       32nd European Symposium on Artificial Neural Networks, Computational
       Intelligence and Machine Learning (ESANN), 2024.
    .. [2] https://github.com/MatthiasJakobs/FROCKS

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     FROCKSClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = FROCKSClassifier(n_kernels=40, random_state=0)
    >>> clf.fit(X, y)
    FROCKSClassifier(n_kernels=40, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=1000,
        n_clients=4,
        n_rounds=100,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.n_classes_ != 2:
            raise ValueError(
                "FROCKSClassifier only supports binary classification, found "
                f"{self.n_classes_} classes."
            )
        if self.n_kernels < self.n_clients:
            raise ValueError("n_kernels must be at least n_clients.")

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        y = np.array([self._class_dictionary[label] for label in y])

        n_top = self.n_kernels // self.n_clients
        clients = _split_clients(y, self.n_clients, rng, repeat_initial=True)
        features = [_SeededRocketFeatures(X[idx], self._n_jobs) for idx in clients]

        client_seeds = [
            np.arange(c * self.n_kernels, (c + 1) * self.n_kernels)
            for c in range(self.n_clients)
        ]
        intercept = 0
        weights = None
        self.n_rounds_ = 0
        for _ in range(self.n_rounds):
            self.n_rounds_ += 1
            previous_weights = weights

            client_weights = []
            selected_seeds = []
            for c, idx in enumerate(clients):
                lr = LogisticRegression()
                lr.fit(features[c].transform(client_seeds[c]), y[idx])

                w = lr.coef_[0]
                top = np.argsort(-np.abs(w))[:n_top]
                intercept += lr.intercept_[0]
                client_weights.append(w[top])
                selected_seeds.append(client_seeds[c][top])

            client_weights = np.concatenate(client_weights)
            selected_seeds = np.concatenate(selected_seeds)
            intercept = intercept / self.n_clients

            # average the weights of kernels selected by more than one client
            seeds = np.unique(selected_seeds)
            weights = np.array(
                [client_weights[selected_seeds == s].mean() for s in seeds]
            )
            client_seeds = [seeds] * self.n_clients

            if (
                previous_weights is not None
                and len(weights) == len(previous_weights)
                and np.all(np.isclose(previous_weights, weights))
            ):
                break

        self.seeds_ = seeds
        self.coef_ = weights
        self.intercept_ = intercept

        return self

    def _predict(self, X):
        Xt = _seeded_rocket_ppv(_normalise(X), self.seeds_, self._n_jobs)
        return self.classes_[(Xt @ self.coef_ + self.intercept_ > 0).astype(int)]

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
        return {"n_kernels": 20, "n_rounds": 3}
