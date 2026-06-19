"""Greedy-margin source for the champion promotion gate (option A, ADR-0034 follow-up).

The gate's original margin was the SAMPLED PPO rollout reward. That signal inflates
as the policy sharpens — a low-entropy learner is barely hurt by sampling while the
higher-entropy BC opponent is, so the *sampled* margin grows even as DEPLOYED (greedy)
strength falls (the 2026-06-18 close: rollout +22->+62 while the tournament fell
+3.29->+1.96; 201 phantom promotions). This module replaces that source with a GREEDY
mini-tournament: it exports the live learner nets and each pool opponent to TorchScript
and runs the deployed `MLAgent` (rank-by-logit play + threshold calls + partner-trick
guard + the learned dragon head) seat-swapped via `collect_pair_deltas` — i.e. exactly
what `check_cotrain` measures. The per-deal margins feed the unchanged `PromotionGate`
(it bootstraps its own CI), so the verdict / promote / reanchor logic is untouched; only
the SOURCE of the margins changes.

Faithful by construction: it reuses the deployed greedy path, so there is no
greedy-rollout-vs-MLAgent gap (the reason option (B) — a greedy mode bolted onto
`BatchedCoTrainPolicy` — was rejected: it would have to re-derive MLAgent semantics
and reconcile the train/eval dragon-give mismatch, reintroducing the very gap this fix
removes).

Cost: an export + two greedy tournaments per gate window (CPU). Run at LOW cadence with
a modest worker count; the persistent rollout pool stays alive during the eval, so peak
memory is rollout_workers + greedy_workers processes (OOM risk, ADR-0034) — keep
`greedy_n_deals` / `greedy_workers` small.
"""

from functools import partial
from pathlib import Path

import numpy as np
import torch

from tichu_eval.tournament import collect_pair_deltas
from tichu_export.torchscript import export_torchscript
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

# Side-effect import: registers the "ml" agent factory so the spawned tournament
# workers can rebuild MLAgents from the builder partials (mirrors check_cotrain).
import tichu_inference.ml_agent  # noqa: F401
from tichu_training.cli.check_cotrain import _EXPORT_NAMES, _build_agent

_NET_TYPES = ("play", "schupfen", "tichu", "grand")


def _agent_kwargs(paths: dict) -> dict:
    """The four TorchScript paths as MLAgent constructor kwargs."""
    return {
        "checkpoint_path": paths["play"],
        "schupfen_path": paths["schupfen"],
        "tichu_call_path": paths["tichu"],
        "grand_call_path": paths["grand"],
    }


def export_live_models(models: dict, out_dir) -> dict:
    """Export the LIVE learner nn.Modules (play/schupfen/tichu/grand) to TorchScript
    under `out_dir`, returning the MLAgent kwargs. Mirrors `check_cotrain.export_nets`
    but skips the `.bin` round-trip — the gate has the modules in memory. Only the play
    policy carries the play Action-Space stamp; the standalone nets carry an empty one
    (matches `export_nets` / `export_model`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    example = (torch.randn(1, FEATURIZER_OUTPUT_DIM), torch.tensor([0], dtype=torch.long))
    paths: dict[str, str] = {}
    for net_key in _NET_TYPES:
        dest = out / _EXPORT_NAMES[net_key]
        export_torchscript(
            models[net_key], example_inputs=example,
            featurizer_version=FEATURIZER_VERSION,
            action_space_version=ACTION_SPACE_VERSION if net_key == "play" else "",
            output_path=dest,
        )
        paths[net_key] = str(dest)
    return _agent_kwargs(paths)


def export_opponent(weights_path: str, arch_cfg: dict, out_dir) -> dict:
    """Rebuild a pool opponent from its rollout-weights `.pt`
    ({"models": {net: state_dict}, "critic": ...}) and export it to TorchScript.
    `arch_cfg` carries the model/schupfen_model/call_model arch so `_build_models`
    reconstructs the exact net shapes."""
    from tichu_training.cli.train_cotrain import _build_models

    models = _build_models(arch_cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    for net_key in _NET_TYPES:
        models[net_key].load_state_dict(state["models"][net_key])
    return export_live_models(models, out_dir)


def greedy_pair_margins(
    learner_kwargs: dict, opp_kwargs: dict, positions, *,
    skill_decile: int, workers: int = 1,
) -> np.ndarray:
    """Seat-swapped greedy learner-minus-opponent total-score margins over `positions`
    (2*len(positions) values, one per paired observation). Builds two deployed greedy
    `MLAgent`s from the exported TorchScript and reuses the tournament's
    `collect_pair_deltas` — the same per-Position observations `check_cotrain`'s
    bootstrap draws from. Builders are module-level `partial`s (spawn-safe) so a
    `workers>1` tournament can rebuild its agents per worker."""
    builder_learner = partial(_build_agent, "ml", skill_decile=skill_decile, **learner_kwargs)
    builder_opp = partial(_build_agent, "ml", skill_decile=skill_decile, **opp_kwargs)
    totals, _call_bonus = collect_pair_deltas(
        builder_learner, builder_opp, positions, workers=workers,
    )
    return totals


def record_greedy_window(
    gate, learner_models: dict, opp_cycle, *,
    arch_cfg: dict, positions, skill_decile: int, workers: int, export_root,
) -> None:
    """Run ONE greedy mini-tournament of the live learner vs EACH pool opponent in
    `opp_cycle` (`[(name, weights_path), ...]`) and record the per-deal margins into
    `gate`. A single call fills every opponent's window (2*len(positions) obs each), so
    — unlike the sampled gate's per-iteration accumulation — the verdict can be drawn
    immediately after. The learner is exported once and reused across opponents."""
    root = Path(export_root)
    learner_kwargs = export_live_models(learner_models, root / "learner")
    for name, weights_path in opp_cycle:
        opp_kwargs = export_opponent(weights_path, arch_cfg, root / f"opp_{name}")
        margins = greedy_pair_margins(
            learner_kwargs, opp_kwargs, positions,
            skill_decile=skill_decile, workers=workers,
        )
        gate.record(name, margins)
