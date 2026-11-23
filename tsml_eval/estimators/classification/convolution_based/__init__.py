"""Convolution based estimators."""

__all__ = [
    "DROCKSClassifier",
    "DynamicRocketClassifier",
    "Fed2RCClassifier",
    "FROCKSClassifier",
    "HDCHydraClassifier",
    "HDCMiniRocketClassifier",
    "HDCMultiRocketClassifier",
    "HDCMultiRocketHydraClassifier",
    "MiniFed2RCClassifier",
]

from tsml_eval.estimators.classification.convolution_based._drocks import (
    DROCKSClassifier,
)
from tsml_eval.estimators.classification.convolution_based._dynamic_rocket import (
    DynamicRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._fed2rc import (
    Fed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._frocks import (
    FROCKSClassifier,
)
from tsml_eval.estimators.classification.convolution_based._hdc_hydra import (
    HDCHydraClassifier,
)
from tsml_eval.estimators.classification.convolution_based._hdc_minirocket import (
    HDCMiniRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._hdc_mr_hydra import (
    HDCMultiRocketHydraClassifier,
)
from tsml_eval.estimators.classification.convolution_based._hdc_multirocket import (
    HDCMultiRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._minifed2rc import (
    MiniFed2RCClassifier,
)
