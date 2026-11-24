"""Convolution based estimators."""

__all__ = [
    "ClassWiseMiniRocketClassifier",
    "DetachRocketClassifier",
    "DetachRocketEnsembleClassifier",
    "POCKETClassifier",
    "SRocketClassifier",
]

from tsml_eval.estimators.classification.convolution_based._cw_minirocket import (
    ClassWiseMiniRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._detach_ensemble import (
    DetachRocketEnsembleClassifier,
)
from tsml_eval.estimators.classification.convolution_based._detach_rocket import (
    DetachRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._pocket import (
    POCKETClassifier,
)
from tsml_eval.estimators.classification.convolution_based._s_rocket import (
    SRocketClassifier,
)
