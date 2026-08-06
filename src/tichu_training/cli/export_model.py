"""`export_model` CLI — export trained checkpoints to TorchScript.

ONNX is documented as a follow-up and rejected with a clear message at
this stage. The CLI handles three checkpoint kinds:
  * `--checkpoint`              the BC policy (trunk + 4 heads).
  * `--tichu-checkpoint`        the Tichu call network.
  * `--grand-checkpoint`        the Grand Tichu call network.

Architecture hyperparameters are not stored in checkpoints, so a
`--model-config` YAML provides them. Each section maps to a model kind
and lists the constructor kwargs used at training time.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from tichu_export.benchmark import benchmark_p99
from tichu_export.torchscript import export_torchscript
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.training import load_checkpoint
from tichu_training.featurizer import FEATURIZER_VERSION


log = logging.getLogger("export_model")



def export_policy_module(model, output_path) -> None:
    """Trace + save a BCModel policy, with the right arity for its mask flag.

    v7 (ADR-0044): a mask-consuming policy has a THREE-argument forward that
    raises on a missing mask, so it must be traced with three example inputs and
    stamped so the loader knows to supply one. One helper, so the CLI export and
    the co-train greedy gate's live export cannot disagree about the contract.
    """
    feature_dim = model.trunk.input_proj.in_features - model.skill.embedding.embedding_dim
    use_mask = bool(getattr(model, "use_legal_mask", False))
    if use_mask:
        feature_dim -= HEAD_LOGIT_DIMS["play"]
    inputs = [torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long)]
    if use_mask:
        inputs.append(torch.zeros(1, HEAD_LOGIT_DIMS["play"]))
    export_torchscript(
        model, example_inputs=tuple(inputs),
        featurizer_version=FEATURIZER_VERSION,
        action_space_version=ACTION_SPACE_VERSION,
        output_path=output_path,
        use_legal_mask=use_mask,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export trained checkpoints to TorchScript.")
    p.add_argument("--checkpoint", required=True, metavar="FILE",
                   help="BC policy checkpoint to export.")
    p.add_argument("--tichu-checkpoint", metavar="FILE")
    p.add_argument("--grand-checkpoint", metavar="FILE")
    p.add_argument("--schupfen-checkpoint", metavar="FILE")
    p.add_argument("--model-config", required=True, metavar="FILE",
                   help="YAML with `policy`/`tichu_call`/`grand_tichu_call` arch hyperparameters.")
    p.add_argument("--format", choices=("torchscript", "onnx"), default="torchscript")
    p.add_argument("--output", required=True, metavar="DIR")
    p.add_argument("--benchmark", action="store_true",
                   help="Run a sequential latency benchmark after export.")
    p.add_argument("--benchmark-n", type=int, default=1000)
    p.add_argument("--p99-budget-ms", type=float, default=500.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.format == "onnx":
        print("ONNX export is a follow-up; only TorchScript is supported in this version.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8")) or {}

    # Policy export.
    policy_arch = arch.get("policy", {})
    model = BCModel(**policy_arch)
    load_checkpoint(args.checkpoint, model)
    policy_out = out_dir / "policy.pt"
    export_policy_module(model, policy_out)
    log.info("wrote %s (featurizer=%s, action_space=%s)",
             policy_out, FEATURIZER_VERSION, ACTION_SPACE_VERSION)

    if args.tichu_checkpoint:
        _export_call(
            TichuCallNetwork, args.tichu_checkpoint,
            arch.get("tichu_call", {}), out_dir / "tichu_call.pt",
        )
    if args.grand_checkpoint:
        _export_call(
            GrandTichuCallNetwork, args.grand_checkpoint,
            arch.get("grand_tichu_call", {}), out_dir / "grand_tichu_call.pt",
        )
    if args.schupfen_checkpoint:
        _export_schupfen(
            args.schupfen_checkpoint, arch.get("schupfen", {}), out_dir / "schupfen.pt",
        )

    if args.benchmark:
        stats = benchmark_p99(model, *inputs, n=int(args.benchmark_n))
        print(f"benchmark (n={args.benchmark_n}): "
              f"p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
              f"p99={stats['p99']:.2f}ms mean={stats['mean']:.2f}ms")
        if stats["p99"] > args.p99_budget_ms:
            print(f"p99 EXCEEDED budget of {args.p99_budget_ms:.0f}ms", file=sys.stderr)
            return 3
    return 0


def _export_call(cls, checkpoint_path: str, arch: dict, output: Path) -> None:
    net = cls(**arch)
    load_checkpoint(checkpoint_path, net)
    feature_dim = int(arch["feature_dim"])
    inputs = (torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long))
    export_torchscript(
        net, example_inputs=inputs,
        featurizer_version=FEATURIZER_VERSION,
        action_space_version="",
        output_path=output,
    )
    log.info("wrote %s (featurizer=%s)", output, FEATURIZER_VERSION)


def _export_schupfen(checkpoint_path: str, arch: dict, output: Path) -> None:
    net = SchupfenNetwork(**arch)
    load_checkpoint(checkpoint_path, net)
    feature_dim = int(arch["feature_dim"])
    inputs = (torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long))
    # strict=False inside export_torchscript permits the 3-tuple output the
    # Schupfen Network returns (one 56-way head per direction).
    export_torchscript(
        net, example_inputs=inputs,
        featurizer_version=FEATURIZER_VERSION,
        action_space_version="",
        output_path=output,
    )
    log.info("wrote %s (featurizer=%s)", output, FEATURIZER_VERSION)


if __name__ == "__main__":
    sys.exit(main())
