"""Decentralised ROCKET features (DROCKS) classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DROCKSClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _encode_labels,
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
    kernels with the largest absolute weights to the list of selected kernels. In the
    first round, each client passes the selected kernels, topped up to ``n_kernels``
    with kernels it has not used, on to the next client. In later rounds, every
    client uses the kernels the last client received in the first round. Training
    stops when a round selects no new kernel, or after ``n_rounds`` rounds. The model
    of the last client is used to predict.

    This follows the version of the reference implementation [2]_ that produced the
    published results, before commit ``fc06626``, including its split of the data
    between clients. Later versions of the code keep the first client's kernels for
    the whole first round and pass kernels on in later rounds. As in the reference
    code, the final intercept is the sum of all intercepts fitted during training
    divided by the number of clients. Only the first entry of this intercept is added
    to every output, so it has no effect on multiclass predictions.

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
        The classes are in numeric order if the labels are numbers, as in the
        reference code, and otherwise in the order of ``classes_``.
    intercept_ : float
        The intercept of the final model.
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

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        y, self._ordered_classes = _encode_labels(y, self.classes_)

        n_top = self.n_kernels // self.n_clients
        clients = _split_clients(X, y, self.n_clients, rng, repeat_initial=True)
        features = [_SeededRocketFeatures(X[idx], self._n_jobs) for idx in clients]

        mlp = MLPClassifier(
            hidden_layer_sizes=(),
            activation="logistic",
            alpha=0,
            batch_size=2,
            learning_rate_init=0.001,
            max_iter=self.max_iter,
            random_state=rng,
        )

        seeds = list(range(self.n_kernels))
        intercept = 0
        selected = []
        selected_set = set()
        previous_selected = None
        for i in range(self.n_rounds):
            for c, idx in enumerate(clients):
                mlp.fit(features[c].transform(seeds), y[idx])

                w = mlp.coefs_[0]
                intercept = intercept + mlp.intercepts_[0]

                # rows of the largest absolute weights over all outputs
                top = np.argsort(-np.abs(w), axis=None)[:n_top] // w.shape[1]
                for t in top:
                    if seeds[t] not in selected_set:
                        selected.append(seeds[t])
                        selected_set.add(seeds[t])

                # in the first round each client passes the selected kernels on,
                # topped up with kernels it has not used. later rounds keep the
                # kernels of the last client in the first round
                if i == 0 and c < self.n_clients - 1:
                    used = set(seeds)
                    new_seeds = []
                    seed = self.n_kernels
                    while len(selected) + len(new_seeds) < self.n_kernels:
                        if seed not in used:
                            new_seeds.append(seed)
                        seed += 1
                    seeds = selected + new_seeds

            if selected_set == previous_selected:
                break
            previous_selected = set(selected_set)

        self.seeds_ = np.array(seeds)
        self.coef_ = mlp.coefs_[0]
        self.intercept_ = (intercept / self.n_clients)[0]
        self.n_rounds_ = i + 1

        return self

    def _predict(self, X):
        Xt = _seeded_rocket_ppv(_normalise(X), self.seeds_, self._n_jobs)
        scores = Xt @ self.coef_ + self.intercept_
        if scores.shape[1] == 1:
            return self._ordered_classes[(scores[:, 0] > 0).astype(int)]
        return self._ordered_classes[np.argmax(scores, axis=1)]

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
