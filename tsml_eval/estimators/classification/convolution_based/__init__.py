"""Convolution based estimators."""

__all__ = [
    "ClassWiseMiniRocketClassifier",
    "DetachRocketClassifier",
    "DetachRocketEnsembleClassifier",
    "DROCKSClassifier",
    "Fed2RCClassifier",
    "FROCKSClassifier",
    "MiniFed2RCClassifier",
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
from tsml_eval.estimators.classification.convolution_based._drocks import (
    DROCKSClassifier,
)
from tsml_eval.estimators.classification.convolution_based._fed2rc import (
    Fed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._frocks import (
    FROCKSClassifier,
)
from tsml_eval.estimators.classification.convolution_based._minifed2rc import (
    MiniFed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._pocket import (
    POCKETClassifier,
)
from tsml_eval.estimators.classification.convolution_based._s_rocket import (
    SRocketClassifier,
)
