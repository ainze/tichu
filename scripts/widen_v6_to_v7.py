r"""Widen v6 Checkpoints into v7 shape (ADR-0044 Net Widening).

A `FEATURIZER_VERSION` bump invalidates every Checkpoint because all four nets
take `FEATURIZER_OUTPUT_DIM`. ADR-0044 retrains only `BCModel` and WIDENS the
rest: copy the v6 weights into the wider first projection and zero-initialise the
added input columns, which is EXACTLY function-preserving (proved in
`tests/training/test_net_widening.py`) because v7 is strictly additive and the
Rich History Block is identically zero at Schupfen / Grand Tichu.

Two jobs, both needed to run a v7 co-train with cpfix3328 in the greedy gate:

  --bc      widen the standalone BC nets (schupfen / tichu / grand) so
            `warm_start` can load them into v7-shaped modules.
  --opponent
            widen a rollout-weights `.pt` ({"models": {net: state_dict}, ...}) so
            the gate's `export_opponent` can rebuild it under the LIVE v7 arch.
            This is what keeps the served cpfix3328 champion in the gate across
            the bump: widened, it is behaviourally the same agent — it ignores
            the 233 new dims and the mask because their weights are zero.

Usage (from the worktree root, with this worktree's src FIRST on PYTHONPATH):

  python scripts/widen_v6_to_v7.py --bc `
      --schupfen C:\workbench\tichu\data\runs\schupfen_full_corpus_v6_resid_cpfix\checkpoints\schupfen_final.bin `
      --tichu    C:\workbench\tichu\data\runs\calls_full_corpus_v6_tichu_resid\checkpoints\tichu_final.bin `
      --grand    C:\workbench\tichu\data\runs\calls_full_corpus_v6\checkpoints\grand_final.bin `
      --config   configs\cotrain_v7_gated.yaml `
      --out-dir  C:\workbench\tichu\data\widened_v7

  python scripts/widen_v6_to_v7.py --opponent `
      --weights  C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix\iter03328_opponent.pt `
      --config   configs\cotrain_v7_gated.yaml `
      --out      C:\workbench\tichu\data\widened_v7\cpfix3328_v7_opponent.pt
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.checkpoint import Checkpoint
from tichu_training.featurizer import (
    FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION, V6_FEATURIZER_OUTPUT_DIM,
)
from tichu_training.widen import widen_to_current_featurizer

log = logging.getLogger("widen_v6_to_v7")

_NET_TYPES = ("play", "schupfen", "tichu", "grand")


def _build(config: dict, net: str, feature_dim: int, *, use_legal_mask=None):
    """One net at an explicit feature width, using the live arch blocks.

    `use_legal_mask=False` for the SOURCE net: v6 policies predate the
    Legal-Mask Trunk Input, so their first projection is `features + skill` only.
    The TARGET takes the live config flag, and widening zero-initialises the mask
    columns — which is why a mask-less v6 champion stays function-identical after
    widening into a mask-consuming v7 net.
    """
    from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
    from tichu_training.bc.heads import BCModel
    from tichu_training.bc.schupfen_model import SchupfenNetwork

    m = config.get("model", {})
    sm = config.get("schupfen_model", {})
    cm = config.get("call_model", {})
    gm = config.get("grand_model") or cm
    if net == "play":
        return BCModel(
            feature_dim=feature_dim, skill_buckets=10,
            skill_dim=int(m.get("skill_dim", 64)),
            trunk_hidden=int(m.get("trunk_hidden", 1024)),
            trunk_depth=int(m.get("trunk_depth", 4)),
            trunk_out_dim=int(m.get("trunk_out_dim", 512)),
            head_hidden=int(m.get("head_hidden", 256)),
            use_legal_mask=(bool(m.get("use_legal_mask", False))
                            if use_legal_mask is None else bool(use_legal_mask)),
        )
    if net == "schupfen":
        return SchupfenNetwork(
            feature_dim, skill_dim=int(sm.get("skill_dim", 64)),
            hidden=int(sm.get("hidden", 512)),
            residual=bool(sm.get("residual", False)),
            depth=int(sm.get("depth", 4)),
        )
    arch = cm if net == "tichu" else gm
    cls = TichuCallNetwork if net == "tichu" else GrandTichuCallNetwork
    return cls(
        feature_dim, skill_dim=int(arch.get("skill_dim", 64)),
        hidden=int(arch.get("hidden", 256)),
        residual=bool(arch.get("residual", False)),
        depth=int(arch.get("depth", 4)),
    )


def _widen_bc_checkpoint(src: Path, dest: Path, config: dict, net: str) -> None:
    """Widen one standalone BC `.bin`, re-stamping it for v7."""
    # Version pin BYPASSED on read: the file is legitimately v6 — widening it is
    # the whole point. The output carries the live stamp, so nothing downstream
    # can mistake it for a native v7 Checkpoint.
    cp = Checkpoint.load(src)
    state = torch.load(io.BytesIO(cp.payload), weights_only=False)

    old = _build(config, net, V6_FEATURIZER_OUTPUT_DIM, use_legal_mask=False)
    old.load_state_dict(state["model"])
    new = _build(config, net, FEATURIZER_OUTPUT_DIM)
    widen_to_current_featurizer(old, new, old_feature_dim=V6_FEATURIZER_OUTPUT_DIM)

    buf = io.BytesIO()
    torch.save({"model": new.state_dict(),
                "optimizer": state.get("optimizer", {}),
                "step": int(state.get("step", 0))}, buf)
    Checkpoint(
        featurizer_version=FEATURIZER_VERSION,
        action_space_version=cp.action_space_version or ACTION_SPACE_VERSION,
        payload=buf.getvalue(),
    ).save(dest)
    log.info("widened %s: %s -> %s", net, src.name, dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc", action="store_true", help="widen standalone BC nets")
    p.add_argument("--opponent", action="store_true",
                   help="widen a rollout-weights .pt (gate pool opponent)")
    p.add_argument("--config", required=True, help="cotrain YAML carrying the arch blocks")
    p.add_argument("--schupfen"), p.add_argument("--tichu"), p.add_argument("--grand")
    p.add_argument("--out-dir")
    p.add_argument("--weights"), p.add_argument("--out")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    log.info("live featurizer=%s dim=%d (v6 prefix=%d)",
             FEATURIZER_VERSION, FEATURIZER_OUTPUT_DIM, V6_FEATURIZER_OUTPUT_DIM)

    if args.bc:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for net, src in (("schupfen", args.schupfen), ("tichu", args.tichu),
                         ("grand", args.grand)):
            if not src:
                continue
            _widen_bc_checkpoint(Path(src), out_dir / f"{net}_v7.bin", config, net)

    if args.opponent:
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        widened = {}
        for net in _NET_TYPES:
            old = _build(config, net, V6_FEATURIZER_OUTPUT_DIM,
                         use_legal_mask=False)
            old.load_state_dict(state["models"][net])
            new = _build(config, net, FEATURIZER_OUTPUT_DIM)
            widen_to_current_featurizer(old, new,
                                        old_feature_dim=V6_FEATURIZER_OUTPUT_DIM)
            widened[net] = new.state_dict()
            log.info("widened opponent net %s", net)
        out = {k: v for k, v in state.items() if k != "models"}
        out["models"] = widened
        # The critic is NOT widened: the gate rebuilds only the four policy nets
        # for a pool opponent, and a v6-shaped critic tensor riding along is
        # inert. Dropping it would silently change the file's shape contract.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.out)
        log.info("wrote %s", args.out)

    if not (args.bc or args.opponent):
        p.error("pass --bc and/or --opponent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
