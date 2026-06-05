"""ForcedPressAgent must eliminate caller passing in exactly the states the telemetry
flags (ADR-0031 premise test). Validated through the REAL detection: a passive baseline
racks up caller_pass_events; wrapping it in ForcedPressAgent drives those to zero, which
only holds if the agent's opportunity test matches play_full's.
"""

from tichu_engine.legality import Pass, legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_training.search.forced_press import ForcedPressAgent


class PassiveStub:
    """Calls Tichu (becomes a caller), leads with a play, but cedes (passes) whenever it
    is following — i.e. deliberately exhibits caller passivity for the telemetry to catch."""

    def act(self, pv):
        legal = list(legal_actions_for(pv))
        leader = pv.public.trick.leader
        following = leader is not None and leader != pv.player
        if following:
            for a in legal:
                if isinstance(a, Pass):
                    return a
        non_pass = [a for a in legal if not isinstance(a, Pass)]
        return min(non_pass, key=repr) if non_pass else legal[0]

    def should_call(self, pv, kind):
        return kind == "tichu"

    def rank_actions(self, pv):
        return sorted(legal_actions_for(pv), key=repr)


def test_forced_press_zeroes_caller_pass_events():
    positions = generate_full_position_pool(seed=0, n=12)
    base_opp = base_ev = fp_ev = 0
    for pos in positions:
        b = play_full_round(tuple(PassiveStub() for _ in range(4)),
                            pos.state, pos.grand_prefixes, collect_telemetry=True)
        base_opp += sum(b.telemetry.caller_pass_opportunities)
        base_ev += sum(b.telemetry.caller_pass_events)
        f = play_full_round(tuple(ForcedPressAgent.from_policy(PassiveStub()) for _ in range(4)),
                            pos.state, pos.grand_prefixes, collect_telemetry=True)
        fp_ev += sum(f.telemetry.caller_pass_events)

    assert base_opp > 0          # caller-following-with-beat states actually arose
    assert base_ev > 0           # the passive baseline cedes in some of them
    assert fp_ev == 0            # forced-press never cedes in a flagged state
