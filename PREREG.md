# Välikäsi ("the middleman") — pre-registration

Committed before any code. Written 2026-10-18.

## Question

B holds a hidden variable Z. A talks with B, then writes to C about something
unrelated. C never meets B. How much of Z reaches C, at which depth, and does
it survive when every explicit mention is removed?

Formally the chain is Z -> (A's state) -> (A's note to C) -> C. By the
data-processing inequality, I(Z; C) <= I(Z; A's note). Everything that leaks
must ride in the text A writes. There is no hidden pipe: separate conversations
share no KV cache.

## Setup (fixed now)

- Z = one of four colours: red, blue, green, yellow. Balanced.
- B is scripted (no model), so B's side is identical across Z except for Z.
- A and C are the same frozen model by default (default Qwen/Qwen2.5-0.5B-Instruct),
  greedy decoding, so the counterfactual is exact: rerun with only Z changed
  and any difference downstream is leakage by construction.
- A's message to C is 2 sentences about an unrelated topic (12 topics).
- B scripts come in two levels:
  - **L1 explicit**: B names the colour ("my favourite colour is blue").
  - **L2 implicit**: B never names a colour, only colour-associated things
    (sea, sapphire for blue; tomato, fire engine for red; grass, frog for green;
    banana, lemon for yellow). L2 is the residue case.
- 5 B scripts per (colour, level), 12 topics: 60 episodes per (colour, level),
  480 total, plus 60 no-B control episodes (B talks about nothing coloured).

## Filter (the "strip every explicit mention" rule)

An episode counts as **filtered-clean** if A's note contains no colour word and
no word from the colour-cue lexicon (all four colours' names, their cue words
and the common synonyms; the lexicon is in `valikasi/lexicon.py`, committed
with the code and frozen before the first real run). Episodes are subsampled to
equal counts per colour (the minimum across colours, seed 0) so that survival
rate cannot differ by class. Survival rate per colour is reported.

## Readouts (three depths)

- **R-text**: character n-gram (2-4) logistic regression predicting Z from A's
  note. Grouped CV: folds split by topic, so topic cannot be the signal.
- **R-hidden**: logistic regression on C's hidden state at the last position
  of "note + question", on layers {4, 8, 12, 16, 20} (or the nearest valid
  layers for a smaller model). Layer chosen by inner CV on training topics
  only. Same grouped outer CV by topic.
- **R-behaviour**: C is asked "Pick one colour. Answer with one word." The
  next-token distribution is restricted to the four colour words (first tokens
  of the bare, capitalised and space-prefixed forms, summed). Delta =
  mean over colours of [mean P(colour | Z = colour) - mean P(colour | Z != colour)].
  95% CI by bootstrap over episodes (2000 resamples).

Chance for 4 classes is 25%. Permutation test: 1000 label shuffles within the
grouped CV, p = fraction of shuffles with accuracy >= observed.

Information in bits is reported as the Fano lower bound
I >= log2(4) - H(Pe) - Pe log2(3), with Pe = 1 - accuracy, clipped at 0. It is
a bound from a particular classifier, not an estimate of true capacity.

## Gates

| Gate | Test | Pass |
|---|---|---|
| **V0 determinism** | Rerun 10 episodes twice | identical text, all 10 |
| **V4 sanity (upper bound)** | R-text on L1, unfiltered | accuracy >= 80%. If this fails every other gate is VOID (the instrument cannot see leakage even when it is stated) |
| **V1 text residue** | R-text on L2, filtered-clean | accuracy >= 35% and permutation p < 0.01 |
| **V2 workspace** | R-hidden on L2, filtered-clean | accuracy >= 35% and permutation p < 0.01 |
| **V3 behaviour** | R-behaviour on L2, filtered-clean | Delta >= 0.05 and bootstrap CI lower bound > 0 |
| **V5 no-B control** | R-text, R-hidden, R-behaviour with labels assigned to the no-B episodes at random (balanced) | all three within chance (accuracy < 35% / CI of Delta contains 0). If this fails the pipeline leaks labels |
| **V6 different base** | Only if `--model-c` is given: V2 and V3 repeated with C a different model | reported, no threshold. Prediction (from subliminal learning, Cloud et al. 2025): V2/V3 are weaker than same-base |

## Reporting

- Report the depth profile: text -> hidden -> behaviour. If Z is in the text but
  not in C's hidden state, or in the hidden state but not in behaviour, that is
  a finding, not a failure.
- Report survival rate of the filter per colour and per level.
- Report L1 filtered too (explicit mentions are stripped from A's note, so
  what remains is the residue of the explicit-B condition).
- No threshold changes after seeing results. Fixes to code bugs are logged
  with both runs kept.

## What this does not show

- It does not show the same thing happens between humans. It is the lab where
  the same conversation can be rerun with one hidden variable changed, which
  humans cannot give.
- One 0.5B model, one four-way hidden variable, scripted B. Leakage that rides
  on semantic neighbours the lexicon missed would count as residue here; the
  lexicon is finite, and that is the weakest point of the filter.
- A hidden-state probe shows decodability, not that C uses the information.
  R-behaviour is the check for use.
- The J-lens readout of C's workspace (the original idea) needs the lens from
  RealTimeReadWrite and is left as the next step; the linear probe is the
  lens-free version.

## Addendum 1 — written 2026-10-18, before any model run and before any result

Implementation choices fixed while writing the code. None changes a threshold.

- R-hidden uses a standardised ridge classifier (alpha = 10) instead of logistic
  regression. Reason: the 1000-shuffle null has to repeat the whole procedure,
  including the inner layer selection, and ridge makes that take minutes, not hours.
  R-text keeps logistic regression.
- Folds: outer fold = topic index mod 4. Inner layer selection: 3 folds over the
  training topics only.
- Shuffles permute labels within each topic.
- V5 labels: the 60 no-B episodes get a balanced random label assignment (seed 0)
  before the filter is applied.
- Balancing: per-class subsampling to the minimum count uses seed 0. A set with
  fewer than 8 clean notes per colour is reported NOT EVALUABLE, not FAIL.
- If V4 fails, V1-V3 are reported VOID. V5 is still reported.
- Extra reported lines: how often A's note is identical across all four Z for the
  same (script, topic) (A ignored B), how often A mentions B, empty notes.
