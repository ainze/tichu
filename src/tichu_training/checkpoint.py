"""Versioned checkpoint container.

A Checkpoint pins the featurizer and action-space version strings to a
payload (the model weights, table, etc.). Loading code can pass the
versions it expects; a mismatch raises `VersionMismatchError` immediately
rather than silently producing wrong predictions.

Wire format: 4-byte little-endian header length, JSON header with the
version metadata, then the raw payload bytes.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path


class VersionMismatchError(RuntimeError):
    """Raised when a checkpoint's stored version disagrees with the loader's expectation."""


@dataclass(frozen=True)
class Checkpoint:
    featurizer_version: str
    action_space_version: str
    payload: bytes

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "featurizer_version": self.featurizer_version,
            "action_space_version": self.action_space_version,
        }
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        with path.open("wb") as fh:
            fh.write(struct.pack("<I", len(header_bytes)))
            fh.write(header_bytes)
            fh.write(self.payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_featurizer_version: str | None = None,
        expected_action_space_version: str | None = None,
    ) -> "Checkpoint":
        path = Path(path)
        with path.open("rb") as fh:
            raw = fh.read()
        if len(raw) < 4:
            raise ValueError(f"checkpoint {path} is too short ({len(raw)} bytes)")
        (header_len,) = struct.unpack("<I", raw[:4])
        header_bytes = raw[4 : 4 + header_len]
        payload = raw[4 + header_len :]
        header = json.loads(header_bytes.decode("utf-8"))
        stored_feat = header["featurizer_version"]
        stored_as = header["action_space_version"]

        if expected_featurizer_version is not None and stored_feat != expected_featurizer_version:
            raise VersionMismatchError(
                f"featurizer version mismatch: checkpoint has '{stored_feat}', "
                f"loader expected '{expected_featurizer_version}'"
            )
        if expected_action_space_version is not None and stored_as != expected_action_space_version:
            raise VersionMismatchError(
                f"action_space version mismatch: checkpoint has '{stored_as}', "
                f"loader expected '{expected_action_space_version}'"
            )

        return cls(
            featurizer_version=stored_feat,
            action_space_version=stored_as,
            payload=payload,
        )
