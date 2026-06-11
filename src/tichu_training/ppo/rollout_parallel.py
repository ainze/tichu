"""Process-parallel rollout for full-stack co-training (ADR-0034, lever #1).

Profiling showed an iteration is dominated by the per-game rules-engine Python
(legal-action enumeration, card hashing) — GIL-held, so threads can't parallelize
it. The M self-play games are independent, so we fan them out across worker
PROCESSES: each worker holds the net skeletons (built once in the pool initializer),
loads the current weights from a small file each iteration, rolls out its chunk via
the unchanged single-process `collect_rollout`, and returns whole trajectories
(coarse IPC — not the per-decision IPC that hurt AWR). The pool is persistent across
iterations (spawn startup is paid once). Opt-in via `rollout_workers`; workers=1
keeps the validated single-process path.
"""

import multiprocessing as mp

import torch

# Per-worker state: the net skeletons built once by the initializer and reused
# (load_state_dict in place) on every task — avoids rebuilding nets per iteration.
_WORKER: dict = {}


def save_rollout_weights(path: str, models: dict, critic) -> None:
    """Write the current learner weights (four nets + shared critic) the workers
    load each iteration. Small (tens of MB), overwritten per iteration, read by all
    workers (OS file cache), so no large tensors cross the pool boundary."""
    torch.save(
        {"models": {k: m.state_dict() for k, m in models.items()},
         "critic": critic.state_dict()},
        path,
    )


def _init_worker(arch_cfg: dict, critic_hidden: int, perfect_info: bool) -> None:
    from tichu_training.awr.value_baseline import ValueBaseline
    from tichu_training.cli.train_cotrain import _build_models
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    from tichu_training.perfect_info import PERFECT_INFO_DIM

    # Each worker is single-threaded; parallelism is across processes, so per-worker
    # BLAS threads would oversubscribe (W workers x N threads). One thread each.
    torch.set_num_threads(1)
    critic_dim = PERFECT_INFO_DIM if perfect_info else FEATURIZER_OUTPUT_DIM
    _WORKER.update(
        arch_cfg=arch_cfg,
        models=_build_models(arch_cfg),
        critic=ValueBaseline(critic_dim, hidden=int(critic_hidden)),
        perfect_info=bool(perfect_info),
        opp_models=None,  # built lazily on the first league task
    )


def _opponent_models():
    """Lazily build (once per worker) a second net set for a league opponent, so the
    learner and the frozen opponent can hold different weights simultaneously."""
    from tichu_training.cli.train_cotrain import _build_models
    if _WORKER.get("opp_models") is None:
        _WORKER["opp_models"] = _build_models(_WORKER["arch_cfg"])
    return _WORKER["opp_models"]


def _league_opponent(opp_weights_path: str, *, skill_decile, seed: int, train_wish: bool):
    """Build the frozen league opponent for one task: a separate net set loaded from
    `opp_weights_path`. Its critic is unused (opponent choices aren't recorded), so
    reuse the learner's. When the run co-trains the wish, the opponent wishes via its
    own head too — its snapshot's full policy, matching how the eval master plays."""
    from tichu_training.ppo.cotrain import BatchedCoTrainPolicy

    opp = _opponent_models()
    opp_state = torch.load(opp_weights_path, map_location="cpu", weights_only=False)
    for key, module in opp.items():
        module.load_state_dict(opp_state["models"][key])
    return BatchedCoTrainPolicy(
        opp["play"], opp["schupfen"], opp["tichu"], opp["grand"], _WORKER["critic"],
        skill_decile=skill_decile, perfect_info=_WORKER["perfect_info"],
        generator=torch.Generator().manual_seed(int(seed) + 7919),
        train_wish=bool(train_wish),
    )


def _vine_chunk(task):
    """Collect vine play-head rows for a chunk of dedicated vine games (ADR-0035).
    Loads the live weights like `_rollout_chunk`, then runs the deterministic
    branch-and-compare collection on this worker's positions."""
    from tichu_training.ppo.vine import collect_vine_rows

    (positions, weights_path, decisions_per_game, branches, skill_decile, seed) = task
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    models = _WORKER["models"]
    for key, module in models.items():
        module.load_state_dict(state["models"][key])
    return collect_vine_rows(
        models, positions, decisions_per_game=decisions_per_game,
        branches=branches, skill_decile=skill_decile, seed=seed,
    )


def _rollout_chunk(task):
    from tichu_training.ppo.cotrain import BatchedCoTrainPolicy
    from tichu_training.ppo.rollout import collect_rollout

    (positions, weights_path, opp_weights_path, learner_team, skill_decile, seed,
     train_wish) = task
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    models, critic = _WORKER["models"], _WORKER["critic"]
    for key, module in models.items():
        module.load_state_dict(state["models"][key])
    critic.load_state_dict(state["critic"])

    gen = torch.Generator().manual_seed(int(seed))
    learner = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"], critic,
        skill_decile=skill_decile, perfect_info=_WORKER["perfect_info"], generator=gen,
        train_wish=bool(train_wish),
    )
    if opp_weights_path is None:
        opponent = learner  # pure self-play: opponents share the learner's weights
    else:
        opponent = _league_opponent(
            opp_weights_path, skill_decile=skill_decile, seed=int(seed),
            train_wish=bool(train_wish),
        )
    return collect_rollout(
        positions, learner, opponent_policy=opponent, learner_team=learner_team
    )


class ParallelRollout:
    """Persistent spawn pool that rolls out chunks of the M games across processes.

    Build once (before the iteration loop); call `collect` each iteration with the
    path to the freshly-saved weights. `close` at the end."""

    def __init__(self, arch_cfg: dict, *, critic_hidden: int, skill_decile: int,
                 perfect_info: bool, workers: int, train_wish: bool = False) -> None:
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            int(workers), initializer=_init_worker,
            initargs=(arch_cfg, int(critic_hidden), bool(perfect_info)),
        )
        self._workers = int(workers)
        self._skill_decile = int(skill_decile)
        self._train_wish = bool(train_wish)

    def collect(self, positions, weights_path: str, *, learner_team: int, base_seed: int,
                opp_weights_path: str | None = None):
        """Split the positions across workers (round-robin for balance), roll each
        chunk out, and concatenate the trajectories. Each chunk gets its own RNG seed
        so the workers don't sample identically (statistically equivalent to serial,
        not bit-identical). `opp_weights_path` (a league member) makes the opponent
        seats play those frozen weights; `None` is pure self-play."""
        positions = list(positions)
        w = max(1, min(self._workers, len(positions)))
        chunks = [positions[i::w] for i in range(w)]
        tasks = [
            (chunk, weights_path, opp_weights_path, learner_team, self._skill_decile,
             base_seed + i, self._train_wish)
            for i, chunk in enumerate(chunks) if chunk
        ]
        results = self._pool.map(_rollout_chunk, tasks)
        return [traj for chunk_trajs in results for traj in chunk_trajs]

    def collect_vine(self, positions, weights_path: str, *, decisions_per_game: int,
                     branches: int, base_seed: int) -> list[dict]:
        """Vine play-head rows (ADR-0035) for dedicated vine games, fanned across
        the same worker pool. Deterministic per (positions, weights, base_seed)."""
        positions = list(positions)
        w = max(1, min(self._workers, len(positions)))
        chunks = [positions[i::w] for i in range(w)]
        tasks = [
            (chunk, weights_path, int(decisions_per_game), int(branches),
             self._skill_decile, base_seed + i)
            for i, chunk in enumerate(chunks) if chunk
        ]
        results = self._pool.map(_vine_chunk, tasks)
        return [row for chunk_rows in results for row in chunk_rows]

    def close(self) -> None:
        self._pool.close()
        self._pool.join()
