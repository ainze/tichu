"""Checkpoint round-trip and version-mismatch behaviour."""

import pytest

from tichu_training.checkpoint import Checkpoint, VersionMismatchError


def test_round_trip_preserves_all_fields(tmp_path):
    cp = Checkpoint(
        featurizer_version="v1",
        action_space_version="v1",
        payload=b"\x00\x01\x02\x03some-weights",
    )
    cp.save(tmp_path / "model.ckpt")
    loaded = Checkpoint.load(tmp_path / "model.ckpt")
    assert loaded == cp


def test_load_with_matching_versions_succeeds(tmp_path):
    cp = Checkpoint(featurizer_version="v1", action_space_version="v1", payload=b"x")
    cp.save(tmp_path / "ok.ckpt")
    loaded = Checkpoint.load(
        tmp_path / "ok.ckpt",
        expected_featurizer_version="v1",
        expected_action_space_version="v1",
    )
    assert loaded.payload == b"x"


def test_load_with_no_expected_versions_does_not_raise(tmp_path):
    cp = Checkpoint(featurizer_version="v1", action_space_version="v1", payload=b"")
    cp.save(tmp_path / "ck.ckpt")
    # No expected_* args → pure inspection mode.
    loaded = Checkpoint.load(tmp_path / "ck.ckpt")
    assert loaded.featurizer_version == "v1"


def test_featurizer_mismatch_raises_with_clear_message(tmp_path):
    Checkpoint(featurizer_version="v1", action_space_version="v1", payload=b"").save(tmp_path / "x")
    with pytest.raises(VersionMismatchError) as ei:
        Checkpoint.load(tmp_path / "x", expected_featurizer_version="v2")
    msg = str(ei.value)
    assert "featurizer" in msg.lower()
    assert "v1" in msg
    assert "v2" in msg


def test_action_space_mismatch_raises_with_clear_message(tmp_path):
    Checkpoint(featurizer_version="v1", action_space_version="v1", payload=b"").save(tmp_path / "x")
    with pytest.raises(VersionMismatchError) as ei:
        Checkpoint.load(tmp_path / "x", expected_action_space_version="v99")
    msg = str(ei.value)
    assert "action" in msg.lower() and "space" in msg.lower()
    assert "v1" in msg
    assert "v99" in msg


def test_payload_round_trip_byte_identical(tmp_path):
    blob = bytes(range(256)) * 4  # 1024 bytes
    cp = Checkpoint(featurizer_version="v1", action_space_version="v1", payload=blob)
    cp.save(tmp_path / "b.ckpt")
    loaded = Checkpoint.load(tmp_path / "b.ckpt")
    assert loaded.payload == blob


def test_version_mismatch_error_is_a_runtime_error():
    assert issubclass(VersionMismatchError, RuntimeError)
