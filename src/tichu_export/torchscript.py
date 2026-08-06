"""TorchScript export primitive with embedded version metadata.

A traced TorchScript artifact carries the featurizer and action-space
version strings as `_extra_files` so the inference service can assert
compatibility at load time without importing the python-side checkpoint
wrapper or any training dependency.

The same `VersionMismatchError` contract as `tichu_training.checkpoint`
is mirrored here so callers see a consistent exception type regardless
of whether they load the python `.bin` or the TorchScript `.pt`.
"""

from pathlib import Path
from typing import Sequence

import torch


_EXTRA_FILE_FEATURIZER = "featurizer_version"
_EXTRA_FILE_ACTION_SPACE = "action_space_version"
# v7 (ADR-0044): whether the policy consumes the legal-Intent mask as a TRUNK
# input. A traced module's arity is not introspectable, so the flag has to ride
# with the artifact — otherwise a loader must guess, and guessing wrong either
# crashes or (worse) silently feeds 1809 zeros where training saw the legal set.
_EXTRA_FILE_LEGAL_MASK = "use_legal_mask"


class VersionMismatchError(RuntimeError):
    """Raised when an exported artifact's embedded versions disagree with the loader's expectation."""


def export_torchscript(
    module: torch.nn.Module,
    *,
    example_inputs: Sequence[torch.Tensor],
    featurizer_version: str,
    action_space_version: str,
    output_path: str | Path,
    use_legal_mask: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    # strict=False permits dict-shaped outputs (e.g. BCModel's per-head logit
    # map). Trace inputs determine the dict's key set, which is fixed for the
    # life of the module — the only thing strict mode is meant to catch.
    traced = torch.jit.trace(module, tuple(example_inputs), strict=False)
    extra_files = {
        _EXTRA_FILE_FEATURIZER: featurizer_version.encode("utf-8"),
        _EXTRA_FILE_ACTION_SPACE: action_space_version.encode("utf-8"),
        _EXTRA_FILE_LEGAL_MASK: (b"1" if use_legal_mask else b"0"),
    }
    torch.jit.save(traced, str(output_path), _extra_files=extra_files)


def load_exported(
    path: str | Path,
    *,
    expected_featurizer_version: str | None = None,
    expected_action_space_version: str | None = None,
) -> torch.jit.ScriptModule:
    path = Path(path)
    extra_files = {_EXTRA_FILE_FEATURIZER: b"", _EXTRA_FILE_ACTION_SPACE: b""}
    module = torch.jit.load(str(path), _extra_files=extra_files)
    stored_feat = extra_files[_EXTRA_FILE_FEATURIZER].decode("utf-8")
    stored_as = extra_files[_EXTRA_FILE_ACTION_SPACE].decode("utf-8")
    if expected_featurizer_version is not None and stored_feat != expected_featurizer_version:
        raise VersionMismatchError(
            f"featurizer version mismatch: artifact has {stored_feat!r}, "
            f"loader expected {expected_featurizer_version!r}"
        )
    if expected_action_space_version is not None and stored_as != expected_action_space_version:
        raise VersionMismatchError(
            f"action_space version mismatch: artifact has {stored_as!r}, "
            f"loader expected {expected_action_space_version!r}"
        )
    return module


def exported_uses_legal_mask(path: str | Path) -> bool:
    """Whether the exported policy at `path` expects a legal-mask argument.

    Absent stamp = False, which is correct for every pre-v7 artifact: they were
    traced with a two-argument forward.
    """
    extra_files = {_EXTRA_FILE_LEGAL_MASK: b""}
    torch.jit.load(str(Path(path)), _extra_files=extra_files)
    return extra_files[_EXTRA_FILE_LEGAL_MASK].decode("utf-8") == "1"
