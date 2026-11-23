"""Federated ROCKET kernels and ridge classifier (Fed2RC)."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Fed2RCClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _federated_ridge,
    _normalise,
    _seeded_rocket_ppv,
    _SeededRocketFeatures,
    _split_clients,
)


class Fed2RCClassifier(BaseClassifier):
    """Federated ROCKET kernels and ridge classifier (Fed2RC).

    Fed2RC [1]_ is a gradient-free federated version of ROCKET. The training data is
    split between ``n_clients`` simulated clients. Each kernel is a single ROCKET
    kernel identified by its seed, and only its proportion of positive values (PPV)
    feature is used.

    In each round every client fits a ``RidgeClassifierCV`` on the PPV features of its
    kernels, and sends the ``n_kernels // n_clients`` kernels with the largest
    absolute weights to the server. For multiclass problems these are the weights of
    the first class. Clients start with
    ``n_kernels`` kernels of their own, and in later rounds all use the union of the
    kernels selected in the previous round. Selection stops when a round selects the
    same kernels as the one before, or after ``n_rounds + 1`` rounds.

    The final ridge classifier is fitted in closed form from the sums of each
    client's ``X^T X`` and ``X^T Y``, with ``Y`` the one-hot class labels and no
    intercept. Lambda is selected with the federated protocol from [1]_: each client
    picks the lambda that fits its own data best when the model is fitted on the other
    clients' sums, and the server takes the mean of these weighted by each client's
    accuracy. Predictions are the argmax of the class scores, with the weights of
    each class scaled to unit norm.

    This follows the reference implementation [2]_, including its split of the data
    between clients.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels each client starts with. 10000 is the setting of the main
        results in [1]_.
    n_clients : int, default=4
        Number of simulated clients the training data is split between.
    n_rounds : int, default=100
        Maximum number of kernel selection rounds after the first.
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
        The weights of the final model, of shape ``(len(seeds_), n_classes_)``.
    lambda_ : float
        The regularisation parameter selected for the final model.
    n_rounds_ : int
        The number of kernel selection rounds run.

    See Also
    --------
    MiniFed2RCClassifier, FROCKSClassifier, DROCKSClassifier

    References
    ----------
    .. [1] Casella, Bruno, Samuele Fonio, Lorenzo Sciandra, Claudio Gallicchio, Marco
       Aldinucci, Mirko Polato, and Roberto Esposito. "Fed2RC: Federated rocket
       kernels and ridge classifier for time series classification." ECAI 2025,
       pp. 3297-3304, 2025.
    .. [2] https://github.com/CasellaJr/Fed2RC

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     Fed2RCClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = Fed2RCClassifier(n_kernels=40, random_state=0)
    >>> clf.fit(X, y)
    Fed2RCClassifier(n_kernels=40, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=10000,
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
        if self.n_kernels < self.n_clients:
            raise ValueError("n_kernels must be at least n_clients.")

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        y = np.array([self._class_dictionary[label] for label in y])

        n_top = self.n_kernels // self.n_clients
        clients = _split_clients(X, y, self.n_clients, rng, repeat_initial=False)
        features = [_SeededRocketFeatures(X[idx], self._n_jobs) for idx in clients]

        client_seeds = [
            np.arange(c * self.n_kernels, (c + 1) * self.n_kernels)
            for c in range(self.n_clients)
        ]
        previous_seeds = None
        self.n_rounds_ = 0
        for _ in range(self.n_rounds + 1):
            self.n_rounds_ += 1
            selected_seeds = []
            for c, idx in enumerate(clients):
                ridge = RidgeClassifierCV(alphas=np.logspace(-3, 0, 100))
                ridge.fit(features[c].transform(client_seeds[c]), y[idx])

                w = np.atleast_2d(ridge.coef_)[0]
                top = np.argsort(-np.abs(w))[:n_top]
                selected_seeds.append(client_seeds[c][top])

            selected_seeds = np.unique(np.concatenate(selected_seeds))
            if np.array_equal(previous_seeds, selected_seeds):
                break
            previous_seeds = selected_seeds
            client_seeds = [selected_seeds] * self.n_clients

        self.seeds_ = selected_seeds
        self.coef_, self.lambda_ = _federated_ridge(
            [f.transform(self.seeds_) for f in features],
            [y[idx] for idx in clients],
            self.n_classes_,
        )

        return self

    def _predict(self, X):
        Xt = _seeded_rocket_ppv(_normalise(X), self.seeds_, self._n_jobs)
        return self.classes_[np.argmax(Xt @ self.coef_, axis=1)]

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
