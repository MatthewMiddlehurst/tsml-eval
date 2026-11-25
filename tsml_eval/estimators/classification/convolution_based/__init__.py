"""Convolution based estimators."""

__all__ = [
    "CEEMDMultiRocketClassifier",
    "ClassWiseMiniRocketClassifier",
    "DetachRocketClassifier",
    "DetachRocketEnsembleClassifier",
    "DROCKSClassifier",
    "DynamicRocketClassifier",
    "Fed2RCClassifier",
    "FROCKSClassifier",
    "FTFVCClassifier",
    "HDCHydraClassifier",
    "HDCMiniRocketClassifier",
    "HDCMultiRocketClassifier",
    "HDCMultiRocketHydraClassifier",
    "HITRocketClassifier",
    "KGMTPClassifier",
    "MASHTClassifier",
    "MiniFed2RCClassifier",
    "POCKETClassifier",
    "RocketPFNClassifier",
    "SelFRocketClassifier",
    "SPROCKETClassifier",
    "SRocketClassifier",
]

from tsml_eval.estimators.classification.convolution_based._ceemd_multirocket import (
    CEEMDMultiRocketClassifier,
)
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
from tsml_eval.estimators.classification.convolution_based._dynamic_rocket import (
    DynamicRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._fed2rc import (
    Fed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._frocks import (
    FROCKSClassifier,
)
from tsml_eval.estimators.classification.convolution_based._ft_fvc import (
    FTFVCClassifier,
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
from tsml_eval.estimators.classification.convolution_based._hit_rocket import (
    HITRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._kg_mtp import (
    KGMTPClassifier,
)
from tsml_eval.estimators.classification.convolution_based._masht import (
    MASHTClassifier,
)
from tsml_eval.estimators.classification.convolution_based._minifed2rc import (
    MiniFed2RCClassifier,
)
from tsml_eval.estimators.classification.convolution_based._pocket import (
    POCKETClassifier,
)
from tsml_eval.estimators.classification.convolution_based._rocket_pfn import (
    RocketPFNClassifier,
)
from tsml_eval.estimators.classification.convolution_based._s_rocket import (
    SRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._self_rocket import (
    SelFRocketClassifier,
)
from tsml_eval.estimators.classification.convolution_based._sprocket import (
    SPROCKETClassifier,
)
