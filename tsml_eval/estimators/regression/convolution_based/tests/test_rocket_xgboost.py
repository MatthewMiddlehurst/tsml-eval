"""Tests for the ROCKET-XGBoost regressor."""

import numpy as np
import pytest
from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.estimators.regression.convolution_based import RocketXGBoostRegressor
from tsml_eval.estimators.regression.convolution_based._rocket_xgboost import (
    _apply_rocket_xgboost_kernels,
)


@pytest.mark.parametrize("n_channels", [1, 4])
def test_rocket_xgboost_five_pooling_features(n_channels):
    """Test the 5-feature kernel transform against direct numpy convolution pooling."""
    X, _ = make_example_3d_numpy(
        n_cases=4,
        n_channels=n_channels,
        n_timepoints=45,
        regression_target=True,
        random_state=0,
    )
    X64 = X.astype(np.float64)
    kernels = Rocket(n_kernels=12, normalise=False, random_state=0).fit(X64).kernels
    Xt = _apply_rocket_xgboost_kernels(X64, kernels)

    assert Xt.dtype == np.float32
    assert Xt.shape == (4, 60)

    weights, lengths, biases, dilations, paddings, n_ch, ch_idx = kernels
    a1 = 0
    a2 = 0
    for j in range(12):
        length = int(lengths[j])
        dilation = int(dilations[j])
        padding = int(paddings[j])
        nc = int(n_ch[j])
        w = weights[a1 : a1 + nc * length].reshape(nc, length)
        chs = ch_idx[a2 : a2 + nc]
        a1 += nc * length
        a2 += nc

        X_pad = (
            np.pad(X64[:, chs, :], ((0, 0), (0, 0), (padding, padding)))
            if padding > 0
            else X64[:, chs, :]
        )
        window = (length - 1) * dilation + 1
        out_len = X_pad.shape[2] - window + 1

        for i in range(4):
            conv = np.array(
                [
                    np.float64(biases[j])
                    + np.sum(w * X_pad[i, :, s : s + window : dilation])
                    for s in range(out_len)
                ],
                dtype=np.float64,
            )
            expected = np.array(
                [
                    np.mean(conv > 0),
                    np.max(conv),
                    np.mean(conv),
                    np.min(conv),
                    np.mean(conv < 0),
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                Xt[i, 5 * j : 5 * j + 5], expected, rtol=1e-6, atol=1e-6
            )


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost>=2.0", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rocket_xgboost_short_series_and_bundled_datasets():
    """Test short-series padding, single-case consistency, and bundled TSER fits."""
    X_short, y_short = make_example_3d_numpy(
        n_cases=16,
        n_channels=2,
        n_timepoints=6,
        regression_target=True,
        random_state=0,
    )
    reg_short = RocketXGBoostRegressor(
        n_kernels=25, normalise=True, random_state=0
    ).fit(X_short, y_short)
    preds_short = reg_short.predict(X_short)
    assert preds_short.shape == (16,)
    assert np.isfinite(preds_short).all()
    np.testing.assert_allclose(reg_short.predict(X_short[:1])[0], preds_short[0])

    for loader in (load_covid_3month, load_cardano_sentiment):
        X_tr, y_tr = loader(split="train")
        X_te, y_te = loader(split="test")
        reg = RocketXGBoostRegressor(n_kernels=100, random_state=1).fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        assert preds.dtype == np.float32
        assert preds.shape == y_te.shape
        assert np.isfinite(preds).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost>=2.0", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rocket_xgboost_invalid_n_kernels():
    """Test non-positive n_kernels raises ValueError."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, regression_target=True, random_state=0
    )
    with pytest.raises(ValueError, match="n_kernels"):
        RocketXGBoostRegressor(n_kernels=0).fit(X, y)
