"""ValueBaseline — small MLP + MSE fit on `round_outcome`."""

import numpy as np
import torch

from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline


def test_output_shape_is_batch_1d():
    torch.manual_seed(0)
    b = ValueBaseline(feature_dim=8, hidden=16)
    x = torch.randn(5, 8)
    y = b(x)
    assert y.shape == (5,)
    assert y.dtype == torch.float32


def test_fit_reduces_mse_on_linear_target():
    rng = np.random.default_rng(0)
    feature_dim = 16
    n = 256
    features = rng.standard_normal((n, feature_dim)).astype(np.float32)
    true_w = rng.standard_normal(feature_dim).astype(np.float32)
    outcomes = (features @ true_w + 0.1 * rng.standard_normal(n)).astype(np.float32)

    torch.manual_seed(0)
    b = ValueBaseline(feature_dim=feature_dim, hidden=32)
    initial_mse = float(((b(torch.from_numpy(features)).detach().numpy() - outcomes) ** 2).mean())
    final_mse = fit_value_baseline(
        b, features, outcomes, batch_size=32, epochs=20, lr=1e-2
    )
    assert final_mse < initial_mse * 0.5, (initial_mse, final_mse)


def test_fit_yields_predictions_correlated_with_targets():
    rng = np.random.default_rng(0)
    feature_dim = 8
    n = 200
    features = rng.standard_normal((n, feature_dim)).astype(np.float32)
    true_w = rng.standard_normal(feature_dim).astype(np.float32)
    outcomes = (features @ true_w).astype(np.float32)

    torch.manual_seed(0)
    b = ValueBaseline(feature_dim=feature_dim, hidden=16)
    fit_value_baseline(b, features, outcomes, batch_size=32, epochs=30, lr=1e-2)

    preds = b(torch.from_numpy(features)).detach().numpy()
    pearson = float(np.corrcoef(preds, outcomes)[0, 1])
    assert pearson > 0.5, pearson


def test_fit_returns_final_mse_float():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((32, 4)).astype(np.float32)
    outcomes = rng.standard_normal(32).astype(np.float32)
    torch.manual_seed(0)
    b = ValueBaseline(feature_dim=4, hidden=8)
    result = fit_value_baseline(b, features, outcomes, batch_size=8, epochs=2, lr=1e-2)
    assert isinstance(result, float)
