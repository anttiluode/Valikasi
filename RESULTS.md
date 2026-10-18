# Välikäsi — results

## Run 1 — Qwen2.5-0.5B-Instruct as A and C, float32, registered design (2026-10-18)

Run by Antti on his machine. 540 notes, 1000 shuffles. Raw output: `results/`.

| gate | verdict |
|---|---|
| V0 determinism | PASS |
| V4 sanity (explicit, unfiltered, text >= 80%) | **FAIL** (29.2%) |
| V1 text residue | VOID |
| V2 workspace | VOID |
| V3 behaviour | VOID |
| V5 no-B control | PASS (all three at chance) |
| V6 different base | not run |

| set | n | text acc | hidden acc | delta [95% CI] |
|---|---|---|---|---|
| L1 all | 240 | 0.292 | - | +0.048 [+0.016, +0.085] |
| L1 clean | 176 | 0.244 | 0.244 | +0.013 [-0.016, +0.040] |
| L2 all | 240 | 0.263 | 0.263 | -0.004 [-0.033, +0.023] |
| L2 clean | 184 | 0.217 | 0.255 | -0.014 [-0.044, +0.014] |
| no-B clean | 40 | 0.175 | 0.225 | +0.029 [-0.038, +0.094] |

Chance is 0.25. Leave-script-out: every set at 0.24–0.26.

### Reading, as registered

V4 failed, so V1–V3 are VOID by the rule written before the run. The headline is
**the 0.5B middleman does not relay**: the experiment cannot test residue because
almost nothing gets through even when B states the colour outright.

### Why V4 failed (post-hoc, `analyse_divergence.py`)

V4 was meant to catch a blind instrument. The instrument isn't blind: it recovered
planted signals in `smoke_test.py`. What failed is the channel.

- When B says "my favourite colour is blue", A's note names B's colour in **13 of 240**
  cases (5.4%). In the rest, A writes a generic note ("I had a great time at the
  restaurant last night…").
- On those few notes, C's behaviour does move: L1-all delta +0.048, CI excluding 0.
  It is the only non-null number in the table, and it comes entirely from explicit
  mentions. Once they are filtered out (L1 clean), it goes to +0.013, CI through 0.
- **Influence without information.** Changing Z changed A's note in 80% of L1 groups
  and 95% of L2 groups (greedy decoding: a different word in B's chat sends the
  continuation down a different path). But every readout is at chance on those
  changed notes. Z perturbs what A writes without the perturbation carrying Z. In
  L2 the cue objects change more of A's input than the single colour word does in
  L1, so L2 diverges more and carries no more.

So for this model: B's hidden variable shakes the middleman's output, but what comes
out is noise with respect to Z. Zero bits detected at text, workspace or behaviour
once explicit words are gone.

### Weak points

- The lexicon is broader than needed: "water" and "clear" hit in no-B notes too, so
  the filter dropped some notes for generic words. This costs sample size only; it
  can't create a leak.
- V4 conflated "instrument blind" with "channel closed". The rule was followed. The
  divergence script is what separates the two, and it is post-hoc.
- 0.5B models are known to follow "do not mention B" loosely and to ignore context.
  A closed channel at 0.5B says nothing about larger models.

### Next

The same registered design with a larger A, which is more likely to let B colour the
note. No threshold changes:

```
python3.13 run.py --model Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 --out results_3b
python3.13 analyse_divergence.py results_3b
```

If V4 passes at 3B, V1–V3 become testable for the first time.
