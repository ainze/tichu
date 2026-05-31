# Tichu Inference Service — API Contract

Self-contained spec for a client (e.g. the game UI) to drive the Tichu AI.
The service is **stateless**: it holds no game state between requests. The
client owns the game/engine, and asks the service for one decision at a time.

- Source: `src/tichu_inference/app.py` (endpoints), `src/tichu_inference/codec.py` (JSON shapes).
- Launch: `py -3.14 -m tichu_inference.cli.serve --config configs\serve_100k_v5.yaml --port 8000` (with `$env:PYTHONPATH = "src"`).
- Base URL: `http://<host>:8000`.

---

## 1. Card encoding — integer id 0..55

Every card is an integer id. The mapping is fixed and deterministic:

```
Natural cards:  id = suit_index * 13 + (rank - 2)
  suit_index:  jade=0, sword=1, pagoda=2, star=3
  rank:        2..10 = pips, 11=Jack, 12=Queen, 13=King, 14=Ace

  jade   2..14 -> 0..12
  sword  2..14 -> 13..25
  pagoda 2..14 -> 26..38
  star   2..14 -> 39..51

Special cards:
  mahjong = 52   (the "1", forces/holds the lead; can carry a wish)
  dog     = 53
  phoenix = 54   (wild; see as_rank / phoenix_as_rank)
  dragon  = 55   (highest single; trick goes to an opponent)
```

There are exactly 56 cards. (This id scheme is identical to the model's
internal feature-slot order — you only need this one table.)

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`  | Liveness + agent roster |
| POST | `/act`     | One play / wish / dragon / schupfen decision |
| POST | `/call`    | Binary Tichu / Grand-Tichu call decision |
| GET  | `/metrics` | Prometheus text (counters + latency) |

`difficulty` is always one of: `"easy" | "medium" | "hard" | "master"`.
All four are configured at startup; unknown values → HTTP 400.

### 2.1 `GET /health`

Response `200`:
```json
{ "status": "ok", "agents": ["easy", "hard", "master", "medium"] }
```

### 2.2 `POST /act`

Use for any decision the engine models as the current player's turn or a
pending decision (**play, mahjong-wish, dragon-give, schupfen**). The service
infers *which* from `private_state.public.pending_decision` (or, if null, it's a
play decision for `public.current_player`). Set `private_state.player` to the AI
seat you want a move for.

Request:
```json
{
  "difficulty": "master",
  "private_state": { ...PrivateState (see §3)... }
}
```

Response `200`:
```json
{
  "action": { "kind": "...", ... },   // see §4
  "action_index": null,               // always null; ignore
  "fallback_used": false              // true => model failed, a random legal action was returned
}
```

Errors: `400` if `difficulty` unknown, `private_state` missing, or malformed.

### 2.3 `POST /call`

Tichu/Grand-Tichu calls are **not** engine actions, so they have their own
endpoint. Ask at the trained moments (see §5).

Request:
```json
{
  "difficulty": "master",
  "kind": "grand",                    // "tichu" | "grand"
  "private_state": { ...PrivateState... }
}
```

Response `200`:
```json
{ "call": true }                      // true => the AI calls; false => declines
```

Errors: `400` if `difficulty` unknown, `kind` not in {`tichu`,`grand`},
`private_state` missing or malformed. Baseline/`easy` and any tier without a
call network wired always return `{"call": false}`.

### 2.4 `GET /metrics`

Prometheus text exposition: `tichu_requests_total{difficulty=...}`,
`tichu_fallback_total{difficulty=...}`, `tichu_latency_p50_ms` / `p95` / `p99`.

---

## 3. `PrivateState` JSON

The requesting player's full view: their own hand (hidden info) + public state.

```json
{
  "player": 0,                         // seat 0..3 this state belongs to (the AI)
  "hand": [0, 5, 13, 52, ...],         // card ids the AI holds
  "public": {
    "current_player": 0,               // whose turn it is, seat 0..3
    "hand_sizes": [14, 14, 14, 14],    // cards remaining per seat
    "scores": [0, 0],                  // cumulative game score [team0 (seats 0&2), team1 (seats 1&3)]
    "trick": { ...Trick (see §3.1)... },
    "mahjong_wish": null,              // active wished rank 2..14, or null
    "pending_decision": null,          // null for a normal play; else §3.2
    "round_points_by_player": [0,0,0,0],
    "out_order": [],                   // seats that have gone out this round, in order
    "tichu_callers": [],               // seats that called Tichu this round
    "grand_tichu_callers": []          // seats that called Grand Tichu
  }
}
```

Teams are fixed: **team 0 = seats 0 & 2, team 1 = seats 1 & 3** (partners sit across).

### 3.1 `Trick`

```json
{
  "plays": [
    { "player": 2, "combination": { ...Combination (see §4.1)... } }
  ],
  "leader": 2,                         // seat that leads the trick, or null if empty
  "passes": [3]                        // seats that have passed this trick
}
```
An empty trick (new lead): `{ "plays": [], "leader": null, "passes": [] }`.

### 3.2 `pending_decision` (one of three)

```json
{ "kind": "schupfen", "submitted": [null, null, null, null] }
// submitted: 4-tuple, one per seat; each is null (not yet) or [to_next, to_partner, to_previous] card ids

{ "kind": "dragon_give", "winner": 0, "points": 25 }
// the dragon-trick winner (seat) must give the trick to an opponent; points in the trick

{ "kind": "mahjong_wish", "player": 0 }
// the seat that just played the Mahjong must declare a wished rank (or none)
```

---

## 4. Action JSON (response of `/act`)

Every action has a `"kind"`. The non-combination kinds:

```json
{ "kind": "Pass" }
{ "kind": "DragonGive", "target": 1 }                  // give dragon trick to seat `target`
{ "kind": "MahjongWish", "rank": 7 }                   // wish rank 2..14, or "rank": null to decline
{ "kind": "SchupfenPass", "to_next": 5, "to_partner": 12, "to_previous": 40 }  // 3 card ids
{ "kind": "BombInterrupt", "player": 3, "bomb": { ...Combination... } }        // out-of-turn bomb
```

A **play** action is a Combination, whose `"kind"` is the combination type
(e.g. `"Single"`, `"Pair"`). Apply it to the named cards in your engine.

### 4.1 Combination JSON (8 kinds)

```json
{ "kind": "Single", "card": 9, "as_rank": null }       // as_rank 2..14 when card==phoenix(54) played onto a rank, else null
{ "kind": "Pair", "cards": [3, 16] }                   // 2 card ids
{ "kind": "Triple", "cards": [3, 16, 29] }             // 3 card ids
{ "kind": "FullHouse", "triple": { ...Triple... }, "pair": { ...Pair... } }
{ "kind": "PairStep", "pairs": [ { ...Pair... }, { ...Pair... } ] }  // 2+ consecutive pairs
{ "kind": "Straight", "cards": [0,1,2,3,4], "phoenix_as_rank": null }  // 5+ cards; phoenix_as_rank set if phoenix substitutes
{ "kind": "FourOfAKindBomb", "cards": [3,16,29,42] }   // 4 of a rank
{ "kind": "StraightFlushBomb", "cards": [0,1,2,3,4] }  // 5+ same-suit consecutive
```

Combinations also appear *inside* `Trick.plays[].combination` and
`BombInterrupt.bomb` — same shapes.

---

## 5. When to ask for each decision

The model was trained on specific game moments; send the matching state.

| Decision | Endpoint | When | `private_state` to send |
|---|---|---|---|
| Grand Tichu | `/call` kind=`grand` | After the first **8** cards are dealt, before the rest | seat's 8-card hand; `hand_sizes` reflect pre-deal-completion |
| Tichu | `/call` kind=`tichu` | At the seat's **first non-pass play** (before committing it) | seat's 14-card hand, normal play state |
| Schupfen | `/act` | Card-pass phase | `pending_decision.kind == "schupfen"`, `player`=the AI seat |
| Mahjong wish | `/act` | After AI plays the Mahjong | `pending_decision.kind == "mahjong_wish"` |
| Dragon give | `/act` | AI won a dragon-led trick | `pending_decision.kind == "dragon_give"` |
| Normal play | `/act` | AI's turn to play/pass | `pending_decision == null`, `current_player`=AI seat |

The service does not enforce timing — it featurizes whatever state you send, so
a wrong/mistimed state yields a miscalibrated decision.

---

## 6. Guarantees & fallback

- Every `/act` response is a **legal** action for the given state (the agent
  ranks/decodes only over the engine's legal set). If the model errors or emits
  non-finite output, it returns a uniform-random **legal** action and sets
  `fallback_used: true`. The game never stalls.
- `/call` returns `false` (decline) on any model error or when no call network
  is wired for that tier — declining is always legal.
- Difficulty `easy` is a deterministic rule-based baseline (no model); it serves
  legal `/act` moves and always declines `/call` (ADR-0003).

---

## 7. Worked examples

### 7.1 Ask `master` for a normal play

```bash
curl -s http://localhost:8000/act -H 'content-type: application/json' -d '{
  "difficulty": "master",
  "private_state": {
    "player": 0,
    "hand": [0, 1, 2, 14, 27, 40, 52],
    "public": {
      "current_player": 0,
      "hand_sizes": [7, 9, 9, 9],
      "scores": [0, 0],
      "trick": { "plays": [], "leader": null, "passes": [] },
      "mahjong_wish": null,
      "pending_decision": null,
      "round_points_by_player": [0,0,0,0],
      "out_order": [],
      "tichu_callers": [],
      "grand_tichu_callers": []
    }
  }
}'
# -> {"action":{"kind":"Single","card":0,"as_rank":null},"action_index":null,"fallback_used":false}
```

### 7.2 Ask `master` whether to call Grand Tichu (8-card state)

```bash
curl -s http://localhost:8000/call -H 'content-type: application/json' -d '{
  "difficulty": "master",
  "kind": "grand",
  "private_state": {
    "player": 0,
    "hand": [55, 54, 12, 25, 38, 51, 11, 24],
    "public": {
      "current_player": 0,
      "hand_sizes": [8, 8, 8, 8],
      "scores": [0, 0],
      "trick": { "plays": [], "leader": null, "passes": [] },
      "mahjong_wish": null,
      "pending_decision": null,
      "round_points_by_player": [0,0,0,0],
      "out_order": [],
      "tichu_callers": [],
      "grand_tichu_callers": []
    }
  }
}'
# -> {"call": true}
```

### 7.3 Schupfen (card pass) via `/act`

Send `pending_decision.kind == "schupfen"`; the response is a `SchupfenPass`
with three distinct card ids from the AI's hand:

```jsonc
// response
{ "action": { "kind": "SchupfenPass", "to_next": 5, "to_partner": 12, "to_previous": 40 },
  "action_index": null, "fallback_used": false }
```

---

## 8. Notes / current limitations

- The exported policy serves **play / wish / dragon** (BC heads) + **schupfen**
  (standalone net). Tichu/Grand calls come from the standalone call nets via
  `/call` (ADR-0023). All consume the same 224-dim featurizer (v5).
- `action_index` is always `null` in `/act` responses (reserved; unused).
- `medium` currently reuses the full BC policy (no smaller checkpoint exported
  yet) — same strength as `hard`.
