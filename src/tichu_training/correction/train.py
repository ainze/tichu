"""Preference Correction training ([ADR-0042](docs/adr/0042-preference-correction.md)).

Two arms over one Correction Corpus, from one base Checkpoint:

  * **PAIR** — `preference_loss`, the sign-verified pairwise ordering.
  * **CE**   — the 2026-06-11 recipe (legal-masked CE toward the alternative on
    Verified Corrections only, KL-to-anchor on the drift set). **Mandatory
    control.** If CE also comes out fine on this corpus then the June kill was
    about lineage or corpus size and the pairwise story is unsupported — worth
    knowing before a 40k Tournament.

Every dial computed here is **selection only**. ADR-0040 convicted correction
fix-rate as a strength proxy ("rose while strength collapsed"), and that verdict
stands: the only verdict instrument is the Tournament.
"""

import copy

import torch

from tichu_training.correction.loss import preference_loss

HELD_MOD = 4  # round_idx % 4 == 3 -> held out (25%)


def split_by_round(df, *, held_mod: int = HELD_MOD):
    """Split by **Round**, never by row. Corrections from one Round share a deal
    and a policy trajectory, so a row-wise split leaks across the boundary — and
    the held-out number is what selects the variant that reaches the Tournament.
    """
    held = df[df["round_idx"] % held_mod == held_mod - 1]
    train = df[df["round_idx"] % held_mod != held_mod - 1]
    return train, held


def rows_from_frame(df, *, skill_decile: int) -> list[dict]:
    """Materialise tensor rows once. Non-corpus frames (the drift set) have no
    verdict columns, so those default to a non-correction with zero margin —
    drift rows only ever feed the KL term."""
    out = []
    for rec in df.to_dict("records"):
        out.append({
            "features": torch.tensor(rec["features"], dtype=torch.float32),
            "skill": torch.tensor([skill_decile], dtype=torch.long),
            "legal_idx": [int(i) for i in rec["legal_idx"]],
            "chosen_idx": int(rec["chosen_idx"]),
            "alt_idx": int(rec.get("alt_idx", rec["chosen_idx"])),
            "is_correction": bool(rec.get("is_correction", False)),
            "mean_delta": float(rec.get("mean_delta", 0.0)),
            "round_idx": int(rec["round_idx"]),
        })
    return out


def _legal_argmax(logits: torch.Tensor, legal_idx) -> int:
    legal = torch.as_tensor(list(legal_idx), dtype=torch.long, device=logits.device)
    return int(legal[int(torch.argmax(logits[legal]))])


@torch.no_grad()
def corpus_dials(logits: torch.Tensor, rows) -> dict:
    """Selection dials over a batch of corpus rows.

    * `fix_rate` — Verified Corrections whose legal-masked argmax is now the
      alternative. 0.0 by construction on a matching base Checkpoint.
    * `preservation` — Verified Non-Corrections still playing `chosen`. 1.0 by
      construction on a matching base. Kept **separate** from `fix_rate`: pooling
      them lets damage on the 31:1 majority hide behind transfer on the minority,
      which is exactly what the positives-only June corpus could not see.
    * `argmax_is_chosen` — the **provenance guard**. Every `chosen` in a
      Correction Corpus is the mining policy's legal-masked argmax, so this reads
      ~1.0 iff these weights are the weights the corpus was mined against.
      Anything lower means a different experiment is being run.
    """
    fixed = kept = agree = 0
    n_corr = n_non = 0
    for i, r in enumerate(rows):
        top = _legal_argmax(logits[i], r["legal_idx"])
        agree += top == r["chosen_idx"]
        if r["is_correction"]:
            n_corr += 1
            fixed += top == r["alt_idx"]
        else:
            n_non += 1
            kept += top == r["chosen_idx"]
    return {
        "fix_rate": fixed / n_corr if n_corr else 0.0,
        "preservation": kept / n_non if n_non else 1.0,
        "argmax_is_chosen": agree / len(rows) if rows else 1.0,
        "n_corrections": n_corr,
        "n_non_corrections": n_non,
    }


def _batched_logits(model, rows, *, chunk: int = 256) -> torch.Tensor:
    device = next(model.parameters()).device
    outs = []
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        feats = torch.stack([r["features"] for r in part]).to(device)
        skills = torch.cat([r["skill"] for r in part]).to(device)
        outs.append(model(feats, skills)["play"])
    return torch.cat(outs, dim=0)


def _masked_logprob(logits, *, legal_idx, target_idx):
    legal = torch.as_tensor(list(legal_idx), dtype=torch.long, device=logits.device)
    return torch.log_softmax(logits[legal], dim=0)[list(legal_idx).index(target_idx)]


def _drift_kl(model, anchor_logits, drift_rows) -> torch.Tensor:
    """KL to the anchor on the drift set. The June guard sampled uniformly over
    all Play Decisions — 39.1% of them forced — so most of its budget anchored
    no-ops; `corpus.drift_rows` excludes forced Decisions."""
    logits = _batched_logits(model, drift_rows)
    kls = []
    for i, r in enumerate(drift_rows):
        legal = torch.as_tensor(list(r["legal_idx"]), dtype=torch.long,
                                device=logits.device)
        kls.append(torch.nn.functional.kl_div(
            torch.log_softmax(logits[i][legal], dim=0),
            torch.log_softmax(anchor_logits[i][legal], dim=0),
            log_target=True, reduction="sum"))
    return torch.stack(kls).mean()


def train_arm(model, train_rows, drift_rows, *, arm: str, epochs: int, lr: float,
              anchor_coef: float, delta_scale: float, log=None):
    """Fine-tune `model` in place. `arm` is "pair" or "ce"."""
    if arm not in ("pair", "ce"):
        raise ValueError(f"arm must be 'pair' or 'ce', got {arm!r}")
    anchor = copy.deepcopy(model).eval()
    with torch.no_grad():
        anchor_logits = _batched_logits(anchor, drift_rows) if drift_rows else None

    # CE is the June recipe verbatim: Verified Corrections only, hard target.
    ce_rows = [r for r in train_rows if r["is_correction"]] if arm == "ce" else None
    # Row indices are built on CPU from the parquet; the model may be on a GPU, and
    # the loss gathers logits by these indices, so they must live where it does.
    dev = next(model.parameters()).device
    chosen = torch.tensor([r["chosen_idx"] for r in train_rows], device=dev)
    alt = torch.tensor([r["alt_idx"] for r in train_rows], device=dev)
    is_corr = torch.tensor([bool(r["is_correction"]) for r in train_rows], device=dev)
    delta = torch.tensor([float(r["mean_delta"]) for r in train_rows], device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        if arm == "pair":
            logits = _batched_logits(model, train_rows)
            loss = preference_loss(logits, chosen_idx=chosen, alt_idx=alt,
                                   is_correction=is_corr, mean_delta=delta,
                                   delta_scale=delta_scale)
        else:
            logits = _batched_logits(model, ce_rows)
            loss = -torch.stack([
                _masked_logprob(logits[i], legal_idx=r["legal_idx"],
                                target_idx=r["alt_idx"])
                for i, r in enumerate(ce_rows)
            ]).mean()
        if anchor_coef > 0.0 and drift_rows:
            loss = loss + anchor_coef * _drift_kl(model, anchor_logits, drift_rows)
        loss.backward()
        opt.step()
        if log is not None:
            log(ep, float(loss.detach()))
    return model
