"""piKL — inference-time KL-regularized better-response (ADR-0037).

piKL re-weights a frozen BC Anchor's action distribution by a paired-rollout
advantage, leashed back toward the anchor by λ:

    π(a) ∝ τ(a) · exp( Q(a) / λ )

λ→∞ recovers the anchor (pure BC); λ→0 is unregularized better-response. The
support is τ's: a candidate the anchor never proposes can never be played, which
is both the cost win (only score the top-k the anchor proposes) and the method's
hard ceiling (it re-ranks what BC proposes, it cannot discover what BC misses).

Torch-free by design — the anchored softmax and Q standardization are pure NumPy;
the net-dependent pieces (candidate proposal, the agent) import torch lazily.
"""

import random
from functools import partial
from pathlib import Path

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_ml.agent import Agent
from tichu_training.search.blunder_miner import playout_from, team_relative
from tichu_training.search.determinize import sample_determinized_world


def pikl_q(agents, root_view, candidates, worlds, *, asked_tichu=frozenset(),
           initial_scores=None) -> list[float]:
    """The piKL advantage for each candidate: force it at the root, play every
    Determinized World in ``worlds`` to terminal under the frozen field
    ``agents``, and average the team-relative ``round_outcome`` over worlds.

    ``worlds`` is passed in as a *list* on purpose — every candidate is scored
    over the **same** worlds, so the deal-luck cancels in the relative
    ``Q(a) − Q(a')`` (the shared-world pairing of ADR-0037 G; the blunder-miner
    trick). Reuses ``playout_from`` / ``team_relative`` — the proven Leaf-Rollout
    estimator, frozen field = the BC Anchor.
    """
    actor = root_view.player
    init = root_view.public.scores if initial_scores is None else initial_scores
    return [
        float(np.mean([
            team_relative(
                playout_from(agents, world, forced_action=cand,
                             asked_tichu=asked_tichu, initial_scores=init),
                actor,
            )
            for world in worlds
        ]))
        for cand in candidates
    ]


def anchored_softmax(tau, q, lam: float) -> np.ndarray:
    """The piKL re-weighting ``π(a) ∝ τ(a)·exp(Q(a)/λ)`` over the k candidates.

    ``tau`` is the BC Anchor's probability over the candidates, ``q`` their
    advantages (standardized), ``lam`` the regularization weight.
    """
    tau = np.asarray(tau, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Clamp the exponent to ±20 before exp: keeps a τ=0 candidate at exactly 0
    # (0·inf = nan would otherwise leak mass off-support) and guards the float
    # overflow vine hit at iter 169 (ADR-0035). exp(20) is a ~5e8 reweight —
    # already far past what any sane λ-leashed advantage asks for.
    weights = tau * np.exp(np.clip(q / lam, -20.0, 20.0))
    return weights / weights.sum()


class piKLAgent(Agent):
    """An Agent that runs piKL over a frozen BC Anchor at each Play Decision
    (ADR-0037). Stateless across decisions — no θ_a, no per-seat clone, no
    optimizer (the whole reason it sits outside the co-drift failure family).

    The ``anchor`` supplies τ (``play_action_scores``) for the candidate set and
    serves the non-Play Decisions (Schupfen / Wish / Dragon / calls) unchanged.
    The ``field`` is the frozen BC Anchor on all four seats — the rollout
    continuation for ``pikl_q``; it defaults to the anchor itself, the design's
    frozen-reference field.
    """

    def __init__(self, anchor, *, worlds: int, k: int, lam: float, q_scale: float,
                 field=None, seed: int = 0) -> None:
        self._anchor = anchor
        self._field = field if field is not None else [anchor] * 4
        self._worlds = int(worlds)
        self._k = int(k)
        self._lam = float(lam)
        self._q_scale = float(q_scale)
        self._rng = random.Random(seed)

    def act(self, private_state):
        # Non-Play Decisions ride the anchor unchanged — piKL touches Play only.
        if private_state.public.pending_decision is not None:
            return self._anchor.act(private_state)
        legal = list(legal_actions_for(private_state))
        if len(legal) <= 1:
            return legal[0] if legal else self._anchor.act(private_state)

        scored = self._anchor.play_action_scores(private_state)[: self._k]
        candidates = [a for a, _ in scored]
        tau = np.array([p for _, p in scored], dtype=np.float64)

        rng = random.Random(self._rng.randrange(2**31))
        worlds = [sample_determinized_world(private_state, None, rng)
                  for _ in range(self._worlds)]
        q_raw = np.array(pikl_q(self._field, private_state, candidates, worlds))
        q = standardize_q(q_raw, scale=self._q_scale)
        pi = anchored_softmax(tau, q, self._lam)
        return candidates[int(np.argmax(pi))]

    def should_call(self, private_state, kind: str) -> bool:
        return self._anchor.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._anchor.rank_actions(private_state)


def build_pikl_agent(*, export_dir, skill_decile: int = 9, worlds: int = 20,
                     k: int = 8, lam: float = 0.1, q_scale: float = 22.0,
                     seed: int = 0, partner_trick_guard: bool = True) -> "piKLAgent":
    """Module-level builder (a picklable Pool target, like the pMCPA builders) that
    wires a real MLAgent over the iter_06225 TorchScript export as the BC Anchor τ
    and serves it through piKL. The field defaults to this same anchor on all four
    seats — the frozen-reference rollout continuation (ADR-0037 H)."""
    from tichu_inference.ml_agent import MLAgent

    d = Path(export_dir)
    anchor = MLAgent(
        d / "policy.pt", skill_decile=skill_decile,
        schupfen_path=d / "schupfen.pt", tichu_call_path=d / "tichu_call.pt",
        grand_call_path=d / "grand_tichu_call.pt",
        partner_trick_guard=partner_trick_guard,
    )
    return piKLAgent(anchor, worlds=worlds, k=k, lam=lam, q_scale=q_scale, seed=seed)


def run_pikl_ab_shard(*, export_dir, positions, out_path, workers: int = 1,
                      skill_decile: int = 9, hp: dict | None = None,
                      progress=None) -> bool:
    """One resumable shard of the piKL A/B over `positions`: piKL(τ) vs the
    un-searched export (ADR-0037 F). Because τ **is** the iter_06225 export, the
    mechanism read (with-vs-without piKL) and the ship bar (vs shipped iter_06225)
    are the *same* comparison — one opponent, `build_export_agent`.

    Writes per-Position seat-swap deltas to `out_path` (.npz: totals + bonuses);
    returns False without recomputing if `out_path` already exists — the unit of
    resume. Pool the shards with `scripts.pool_pmcpa_ab` (identical npz schema)."""
    out_path = Path(out_path)
    if out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hp = dict(hp or {})
    seed = hp.pop("seed", 0)
    builder_a = partial(build_pikl_agent, export_dir=export_dir,
                        skill_decile=skill_decile, seed=seed, **hp)

    from tichu_eval.tournament import collect_pair_deltas
    from tichu_training.ppo.pmcpa import build_export_agent

    builder_b = partial(build_export_agent, export_dir=export_dir,
                        skill_decile=skill_decile)
    # Fine chunks (~6 per worker) so progress fires throughout, not in one burst.
    max_chunk = max(1, len(positions) // (workers * 6)) if workers > 1 else None
    totals, bonuses = collect_pair_deltas(builder_a, builder_b, positions,
                                          workers=workers, progress=progress,
                                          max_chunk=max_chunk)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, totals=totals, bonuses=bonuses)
    tmp.replace(out_path)  # atomic — a half-written shard never looks complete
    return True


def intent_rank(logits, legal_idx, target_idx: int) -> int:
    """The 0-based rank of ``target_idx`` among the legal Intents ordered by
    descending anchor logit (0 = the anchor's top pick). The coverage pre-gate
    primitive (ADR-0037 E): piKL can only reach a correction whose better Intent
    the anchor ranks inside top-k. Only the legal set competes."""
    logits = np.asarray(logits)
    legal = list(legal_idx)
    order = sorted(legal, key=lambda i: -logits[i])
    return order.index(target_idx)


def standardize_q(q_raw, *, scale: float) -> np.ndarray:
    """Center the candidates' raw-point advantages and divide by a fixed
    point-scale, so ``Q/λ`` is dimensionless and the paper's λ grid transfers
    (ADR-0037 D). Mean-subtraction is free — ``anchored_softmax`` is invariant to
    an additive constant in ``q`` — but centering keeps the clamp symmetric."""
    q_raw = np.asarray(q_raw, dtype=np.float64)
    return (q_raw - q_raw.mean()) / scale
