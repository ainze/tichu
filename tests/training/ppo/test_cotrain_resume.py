"""Resume Bundle for full-stack co-training (ADR-0034).

The Resume Bundle is the whole-training-state snapshot that makes a multi-day run
stop-and-continue losslessly: all four policy nets, the shared critic, optimizer
state, the per-net KL coefficients, the iteration counter, the league, and RNG.
Written atomically every iteration (keep last 2) so a Ctrl-C never corrupts it.
"""

import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.ppo.cotrain_resume import (
    PREV_SUFFIX,
    capture_rng,
    load_resume_bundle,
    save_resume_bundle,
)

_F, _FC = 8, 8


def _nets():
    models = {
        "play": BCModel(_F, skill_dim=8, trunk_hidden=16, trunk_depth=1, trunk_out_dim=8, head_hidden=8),
        "schupfen": SchupfenNetwork(_F, skill_dim=8, hidden=8),
        "tichu": TichuCallNetwork(_F, skill_dim=8, hidden=8),
        "grand": GrandTichuCallNetwork(_F, skill_dim=8, hidden=8),
    }
    critic = ValueBaseline(_FC, hidden=8)
    return models, critic


def _optimizer(models, critic):
    params = list(critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    return torch.optim.Adam(params, lr=1e-2)


def _take_a_step(models, critic, optimizer):
    # Populate Adam moment state so the round-trip exercises optimizer restoration.
    loss = critic(torch.randn(3, _FC)).sum()
    for m in models.values():
        loss = loss + sum(p.sum() for p in m.parameters())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_resume_bundle_round_trips_full_training_state(tmp_path):
    torch.manual_seed(0)
    models, critic = _nets()
    optimizer = _optimizer(models, critic)
    _take_a_step(models, critic, optimizer)

    path = str(tmp_path / "train_state.bin")
    save_resume_bundle(
        path, models=models, critic=critic, optimizer=optimizer,
        kl_coefs={"play": 0.3, "schupfen": 0.05, "tichu": 0.2, "grand": 0.2},
        iteration=7, rng_state=capture_rng(), league=[{"tag": "snap0"}],
    )

    # Fresh, differently-initialized nets — load must overwrite them exactly.
    torch.manual_seed(999)
    models2, critic2 = _nets()
    optimizer2 = _optimizer(models2, critic2)
    payload = load_resume_bundle(path, models=models2, critic=critic2, optimizer=optimizer2)

    assert payload["iteration"] == 7
    assert payload["kl_coefs"] == {"play": 0.3, "schupfen": 0.05, "tichu": 0.2, "grand": 0.2}
    assert payload["league"] == [{"tag": "snap0"}]
    assert "rng" in payload
    # Every policy net + the critic now matches the saved weights.
    for dt, m in models.items():
        for k, v in m.state_dict().items():
            assert torch.equal(v, models2[dt].state_dict()[k]), f"{dt}.{k} mismatch"
    for k, v in critic.state_dict().items():
        assert torch.equal(v, critic2.state_dict()[k])
    # Optimizer moment state was restored (non-empty).
    assert optimizer2.state_dict()["state"]


def test_save_is_atomic_and_keeps_the_previous_bundle(tmp_path):
    torch.manual_seed(0)
    models, critic = _nets()
    optimizer = _optimizer(models, critic)
    path = str(tmp_path / "train_state.bin")

    common = dict(models=models, critic=critic, optimizer=optimizer,
                  kl_coefs={"play": 0.1}, rng_state=capture_rng())
    save_resume_bundle(path, iteration=1, **common)
    save_resume_bundle(path, iteration=2, **common)

    prev = path + PREV_SUFFIX
    cur = torch.load(path, weights_only=False)
    old = torch.load(prev, weights_only=False)
    # Latest is live; the one-before is retained for corruption recovery (keep-2).
    assert cur["iteration"] == 2
    assert old["iteration"] == 1
