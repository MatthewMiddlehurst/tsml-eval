"""Shared TabPFN utilities for convolution-based classifiers."""

__maintainer__ = ["MatthewMiddlehurst"]

import copy
import inspect

_TINY_TABPFN_SPECS = None


def _get_tiny_tabpfn_specs():
    """Return a cached lightweight in-memory ``ModelSpecs`` for offline unit tests."""
    global _TINY_TABPFN_SPECS
    if _TINY_TABPFN_SPECS is None:
        import torch
        from tabpfn.architectures.tabpfn_v2 import TabPFNV2, TabPFNV2Config
        from tabpfn.base import ModelSpecs
        from tabpfn.constants import ModelVersion
        from tabpfn.inference_config import InferenceConfig

        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(0)
            cfg = TabPFNV2Config(
                max_num_classes=10,
                num_buckets=10,
                name="TabPFN-tiny",
                emsize=32,
                nlayers=1,
                nhead=2,
                features_per_group=2,
                seed=0,
            )
            model = TabPFNV2(config=cfg, n_out=10)
            model.eval()
        finally:
            torch.random.set_rng_state(rng_state)

        _TINY_TABPFN_SPECS = ModelSpecs(
            model=model,
            architecture_config=cfg,
            inference_config=InferenceConfig.get_default(
                "multiclass", ModelVersion.V2_5
            ),
        )
    return _TINY_TABPFN_SPECS


def _build_tabpfn_classifier(
    *,
    model_version,
    model_path,
    n_estimators,
    **kwargs,
):
    """Instantiate a ``TabPFNClassifier`` across ``tabpfn`` v8 and v9+ APIs."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    sig = inspect.signature(TabPFNClassifier.__init__).parameters
    supports_auto_estimators = sig["n_estimators"].default == "auto"

    options = dict(kwargs)
    if supports_auto_estimators:
        options["n_estimators"] = n_estimators
    elif n_estimators == "auto":
        options["n_estimators"] = 8
        if "auto_scale_n_estimators" in sig:
            options["auto_scale_n_estimators"] = True
    else:
        options["n_estimators"] = int(n_estimators)
        if "auto_scale_n_estimators" in sig:
            options["auto_scale_n_estimators"] = False

    if model_path == "_tiny":
        resolved_path = _get_tiny_tabpfn_specs()
    elif callable(model_path):
        resolved_path = model_path()
    else:
        resolved_path = model_path

    if resolved_path is not None:
        return TabPFNClassifier(model_path=resolved_path, **options)

    return TabPFNClassifier.create_default_for_version(
        ModelVersion(model_version),
        **options,
    )


def _fit_tabpfn_classifier(clf, X, y):
    """Fit ``clf`` and complete any lazy CPU preprocessor step attributes."""
    clf.fit(X, y)
    members = getattr(getattr(clf, "executor_", None), "ensemble_members", None)
    if members:
        sample = X[:1]
        for member in members:
            member.transform_X_test(sample)
    return clf


def _predict_proba_tabpfn(clf, X):
    """Run ``clf.predict_proba(X)`` without mutating fitted state on ``clf``."""
    executor = getattr(clf, "executor_", None)
    prev_speed = (
        dict(executor._speed_metrics)
        if executor is not None and hasattr(executor, "_speed_metrics")
        else None
    )
    models = getattr(clf, "models_", ())
    prev_nan_checks = [getattr(m, "_do_encoder_nan_check", None) for m in models]

    ep = getattr(clf, "ensemble_preprocessor_", None)
    restore_pipelines = (
        getattr(clf, "fit_mode", None) == "low_memory"
        and ep is not None
        and hasattr(ep, "pipelines")
    )
    orig_pipelines = ep.pipelines if restore_pipelines else None
    if restore_pipelines:
        ep.pipelines = copy.deepcopy(orig_pipelines)

    try:
        return clf.predict_proba(X)
    finally:
        if prev_speed is not None:
            executor._speed_metrics = prev_speed
        for m, prev_flag in zip(models, prev_nan_checks):
            if prev_flag is not None:
                m._do_encoder_nan_check = prev_flag
        if restore_pipelines:
            ep.pipelines = orig_pipelines
