"""TorchScript export primitive + version embedding."""

import warnings

import pytest
import torch

from tichu_export.torchscript import (
    VersionMismatchError,
    export_torchscript,
    load_exported,
)


def _build_bc_model():
    from tichu_training.bc.heads import BCModel
    torch.manual_seed(0)
    return BCModel(
        feature_dim=32,
        skill_buckets=10,
        skill_dim=4,
        trunk_hidden=16,
        trunk_depth=1,
        trunk_out_dim=16,
        head_hidden=16,
    )


def _example_inputs():
    return (torch.randn(1, 32), torch.tensor([0], dtype=torch.long))


def test_export_and_load_round_trip(tmp_path):
    model = _build_bc_model()
    output = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=_example_inputs(),
        featurizer_version="v1",
        action_space_version="v1",
        output_path=output,
    )
    assert output.exists()
    loaded = load_exported(
        output,
        expected_featurizer_version="v1",
        expected_action_space_version="v1",
    )
    assert loaded is not None


def test_exported_outputs_match_eager(tmp_path):
    model = _build_bc_model()
    inputs = _example_inputs()
    with torch.no_grad():
        eager_out = model(*inputs)
    output = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=inputs,
        featurizer_version="v1",
        action_space_version="v1",
        output_path=output,
    )
    loaded = load_exported(output)
    fresh = (torch.randn(2, 32), torch.tensor([1, 5], dtype=torch.long))
    with torch.no_grad():
        ts_out = loaded(*fresh)
        eager_fresh = model(*fresh)
    for head, expected in eager_fresh.items():
        actual = ts_out[head]
        assert torch.allclose(actual, expected, atol=1e-5), head


def test_load_raises_on_featurizer_version_mismatch(tmp_path):
    model = _build_bc_model()
    output = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=_example_inputs(),
        featurizer_version="v1",
        action_space_version="v1",
        output_path=output,
    )
    with pytest.raises(VersionMismatchError):
        load_exported(output, expected_featurizer_version="v2")


def test_load_raises_on_action_space_version_mismatch(tmp_path):
    model = _build_bc_model()
    output = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=_example_inputs(),
        featurizer_version="v1",
        action_space_version="v1",
        output_path=output,
    )
    with pytest.raises(VersionMismatchError):
        load_exported(output, expected_action_space_version="v2")


def test_empty_action_space_version_is_allowed(tmp_path):
    """Call networks / belief don't use the canonical action space — should still export."""
    model = _build_bc_model()
    output = tmp_path / "model.pt"
    export_torchscript(
        model,
        example_inputs=_example_inputs(),
        featurizer_version="v1",
        action_space_version="",
        output_path=output,
    )
    loaded = load_exported(output, expected_featurizer_version="v1")
    assert loaded is not None


def test_call_network_exports(tmp_path):
    from tichu_training.bc.call_model import TichuCallNetwork
    torch.manual_seed(0)
    net = TichuCallNetwork(feature_dim=8, skill_buckets=10, skill_dim=4, hidden=16)
    output = tmp_path / "tichu_call.pt"
    inputs = (torch.randn(1, 8), torch.tensor([0], dtype=torch.long))
    with torch.no_grad():
        eager = net(*inputs)
    export_torchscript(
        net,
        example_inputs=inputs,
        featurizer_version="v1",
        action_space_version="",
        output_path=output,
    )
    loaded = load_exported(output, expected_featurizer_version="v1")
    with torch.no_grad():
        ts_out = loaded(*inputs)
    assert torch.allclose(ts_out, eager, atol=1e-5)
