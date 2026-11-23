"""MiniRocket federated ridge classifier (MiniFed2RC)."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MiniFed2RCClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.utils.validation import check_n_jobs
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.convolution_based._federated_utils import (
    _federated_ridge,
    _split_clients,
)


class MiniFed2RCClassifier(BaseClassifier):
    """MiniRocket federated ridge classifier (MiniFed2RC).

    MiniFed2RC [1]_ is a version of Fed2RC that uses MiniRocket features in place of
    federated kernel selection. The training data is split between ``n_clients``
    simulated clients. MiniRocket's kernels and dilations depend only on the series
    length, and its biases are fitted from the data. Each client fits MiniRocket on
    its own data, and the server averages the client biases to give one shared
    transform.

    The ridge classifier is then fitted in closed form as in Fed2RC, from the sums of
    each client's ``X^T X`` and ``X^T Y``, with ``Y`` the one-hot class labels and no
    intercept. Lambda is selected with the Fed2RC federated protocol: each client
    picks the lambda that fits its own data best when the model is fitted on the other
    clients' sums, and the server takes the mean of these weighted by each client's
    accuracy. Predictions are the argmax of the class scores, with the weights of
    each class scaled to unit norm.

    This follows the reference implementation [2]_, including its split of the data
    between clients.

    Parameters
    ----------
    n_kernels : int, default=924
        Number of MiniRocket features. Rounded down to a multiple of 84, with a
        minimum of 84.
    n_clients : int, default=4
        Number of simulated clients the training data is split between.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the MiniRocket transform. ``-1``
        means using all processors.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to split the data between clients and
        to fit MiniRocket. Every client fits MiniRocket with the same seed.

    Attributes
    ----------
    transformer_ : MiniRocket
        The MiniRocket transform with the averaged client biases.
    coef_ : np.ndarray
        The weights of the final model, of shape ``(n_features, n_classes_)``.
    lambda_ : float
        The regularisation parameter selected for the final model.

    See Also
    --------
    Fed2RCClassifier

    References
    ----------
    .. [1] Casella, Bruno, Samuele Fonio, Lorenzo Sciandra, Claudio Gallicchio, Marco
       Aldinucci, Mirko Polato, and Roberto Esposito. "Fast federated time series
       classification using Rocket kernels and closed-form classifiers." Data Mining
       and Knowledge Discovery, 2026.
    .. [2] https://github.com/CasellaJr/Fed2RC

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     MiniFed2RCClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=20, random_state=0)
    >>> clf = MiniFed2RCClassifier(n_kernels=84, random_state=0)
    >>> clf.fit(X, y)
    MiniFed2RCClassifier(n_kernels=84, random_state=0)
    >>> y_pred = clf.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=924,
        n_clients=4,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_clients = n_clients
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)
        y = np.array([self._class_dictionary[label] for label in y])

        clients = _split_clients(y, self.n_clients, rng, repeat_initial=False)

        # clients must share a seed so their kernels and channels match
        if self.random_state is None or isinstance(
            self.random_state, np.random.RandomState
        ):
            seed = rng.randint(np.iinfo(np.int32).max)
        else:
            seed = self.random_state

        self.transformer_ = MiniRocket(
            n_kernels=self.n_kernels, n_jobs=self._n_jobs, random_state=seed
        )
        biases = []
        for idx in clients:
            self.transformer_.fit(X[idx])
            biases.append(self.transformer_.parameters[-1])

        self.transformer_.parameters = (
            *self.transformer_.parameters[:-1],
            np.mean(biases, axis=0).astype(np.float32),
        )

        self.coef_, self.lambda_ = _federated_ridge(
            [self.transformer_.transform(X[idx]).astype(np.float64) for idx in clients],
            [y[idx] for idx in clients],
            self.n_classes_,
        )

        return self

    def _predict(self, X):
        Xt = self.transformer_.transform(X).astype(np.float64)
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
        return {"n_kernels": 84}
