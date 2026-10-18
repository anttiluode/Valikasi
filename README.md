# Välikäsi — the middleman

B holds a hidden variable. A talks with B, then writes to C about something
unrelated. C never meets B. **What does C get?**

This is the lab version of "information leaking from people we never talk to,
through a third party". Humans can't rerun the same conversation with one hidden
variable changed. A model can.

**Status: built, instrument-tested, not yet run on a real model.** Hugging Face is
unreachable from the machine that built this, so nothing here has touched Qwen
weights. Every gate currently reads *not run*. Gates were pre-registered
([PREREG.md](PREREG.md)) before any code existed; two addenda were written before any
model run and are in the same file.

## The chain

```
Z (B's hidden colour)  ->  A reads B's chat  ->  A writes a 2-sentence note to C
                                                  about a topic with nothing to do with B
                                                  ->  C reads the note
```

By the data-processing inequality, what C can learn about Z is capped by what
survives in A's note. Separate conversations share no KV cache, so there is no hidden
pipe: anything that leaks rides in the text. That makes the leak measurable at
three depths:

| depth | readout | question |
|---|---|---|
| text | char n-gram classifier on A's note | is Z recoverable from what A wrote? |
| workspace | linear probe on C's hidden state at the answer position | has Z entered C's representation? |
| behaviour | C is asked "pick one colour"; shift in P(colour) toward Z | does C act on it? |

The counterfactual is exact: B is scripted and decoding is greedy, so two runs
differing only in Z differ downstream only because of Z.

## Two levels of B

- **L1 explicit.** B says "my favourite colour is blue".
- **L2 implicit.** B never names a colour, only things (the sea, sapphires, a clear
  sky). This is the residue case.

Then every note that contains any colour word, cue object or synonym is dropped
(`valikasi/lexicon.py`), classes are balanced by subsampling, and the readouts run on
what's left. A **no-B control** (B talks about trains and staplers, labels assigned
at random) must come out at chance, otherwise the pipeline leaks labels.

## Gates

| gate | pass |
|---|---|
| V0 determinism | 10 reruns give identical notes |
| V4 sanity | explicit mentions (L1, unfiltered) give >= 80% from text. If not, V1-V3 are VOID: the instrument can't see leakage even when it is stated |
| V1 text residue | L2-clean accuracy >= 35% (chance 25%), permutation p < 0.01 |
| V2 workspace | same, on C's hidden state |
| V3 behaviour | mean shift >= 0.05, bootstrap 95% CI lower bound > 0 |
| V5 no-B control | all three readouts at chance |
| V6 different base | with `--model-c`: reported only. Prediction from subliminal learning (Cloud et al. 2025): weaker leak when C is a different base model |

Reported, no threshold: the depth profile, filter survival per colour, how often A
ignored B entirely, bits via a Fano lower bound, and **leave-script-out** accuracy
(does the leak generalise to B conversations the classifier never saw).

## What was tested here

`python smoke_test.py` runs without a GPU or any download:

- L2 and no-B scripts never name a colour; L1 scripts always do; the no-B script
  avoids the whole cue lexicon.
- A's conversation differs across Z only in B's lines.
- The statistics find a planted signal and sit at chance on a shuffled null: planted
  text 0.70 vs null 0.25, planted hidden-state layer recovered exactly (0.92),
  behaviour shift and Fano bound behave.
- `--full` runs all of `run.py` on a miniature random Qwen2 with a fake tokenizer,
  including determinism and a resumed rerun that gives identical numbers. Its gate
  verdicts are meaningless by construction.

Not tested: the real Qwen tokenizer's colour tokens (the code asserts each colour keeps
a unique token id), chat-template behaviour of the real model, and runtime on your GPU.

## Run it

```
git pull
pip install scikit-learn        # if missing; torch and transformers you already have
python3.13 run.py               # Qwen2.5-0.5B-Instruct for A and C, the registered design
```

540 notes are generated one at a time (greedy, 60 new tokens) and cached in
`results/notes.jsonl`; C's readouts are cached per episode. **If it stops, rerun and it
resumes.** I haven't timed generation on a 3060; the shuffle tests (1000 each) took
roughly 4 min for text and 6 min for hidden states on a 2-core CPU. Output:
`results/results.json` and `results/LEDGER.md`.

Options: `--model-c <other model>` for V6, `--perms`, `--max-new`. Using fewer topics or
scripts (`--topics 8 --scripts 3`) is a dry run, not the registered design. Paste
`results/LEDGER.md` back and I'll write the results in as registered.

## How to read the outcomes

- **V4 fails:** the setup can't detect leakage even when B states it. Check that A
  actually uses B's lines (the ignore rate in the ledger) before anything else.
- **V4 passes, V1 fails:** A's note carries no Z once the obvious words are gone. Clean
  negative.
- **V1 passes, V2 fails:** Z is in what A wrote but C doesn't take it into its
  representation.
- **V2 passes, V3 fails:** Z is decodable in C but C doesn't use it.
- **All pass, but leave-script-out is at chance:** the leak is style tied to one
  conversation, not something that generalises across B conversations.
- **V5 fails:** pipeline bug or label leak; nothing else counts.

## Limits

- One 0.5B model, one four-way hidden variable, a scripted B. The 0.5B model may simply
  ignore B, in which case the result is a clean zero.
- The filter is a finite lexicon. Leakage through neighbours it misses (words
  associated with the sea but not on the list) counts as "residue" here, so a pass on
  V1 is an upper bound on style-only leakage.
- A hidden-state probe shows the information is decodable, not that C uses it. V3 is the
  use check.
- The J-lens readout of C's workspace, the original idea, needs the lens from
  RealTimeReadWrite. The linear probe is the lens-free version; the J-lens is the next
  step.
- This says nothing yet about humans. It is the instrument that tells you what to look
  for there.

## Related

Interactive alignment (Pickering & Garrod, 2004); convention spread through pairwise talk
(Centola & Baronchelli, PNAS 2015); contagion vs homophily (Shalizi & Thomas, 2011);
subliminal learning (Cloud et al., 2025); prompt infection (Gu et al., 2024). Cited from
memory, not re-checked this session.
