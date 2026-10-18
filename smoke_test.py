"""Code-path and instrument tests. None of these say anything about Qwen.

  python smoke_test.py            # parts 1-2 (no torch needed)
  python smoke_test.py --full     # also part 3: whole run.py on a tiny random Qwen2
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from valikasi import measure as ms
from valikasi.episodes import TOPICS, a_messages, build_episodes
from valikasi.lexicon import COLOURS, colour_names_only, hits, is_clean


def part1_lexicon_and_episodes():
    eps = build_episodes()
    assert len(eps) == 2 * 4 * 5 * 12 + 5 * 12 == 540, len(eps)
    for e in eps:
        names = colour_names_only(" ".join(e["b"]))
        if e["level"] == "L1":
            assert names == [e["colour"]], (e["id"], names)
        else:  # L2 and NOB: B must never name a colour
            assert names == [], (e["id"], names)
    # the NOB control must also avoid the cue lexicon
    for e in eps:
        if e["level"] == "NOB":
            assert is_clean(" ".join(e["b"])), (e["id"], hits(" ".join(e["b"])))
    # counterfactual: A's conversation differs across Z only in the B lines
    by = {}
    for e in eps:
        if e["level"] == "L2":
            by.setdefault((e["script"], e["topic_idx"]), []).append(e)
    for key, group in by.items():
        assert len(group) == 4
        base = a_messages(group[0])
        for g in group[1:]:
            m = a_messages(g)
            assert m[0] == base[0] and m[-1] == base[-1]  # system prompt and task identical
            assert any(x != y for x, y in zip(m, base))
    assert not is_clean("I love the sea") and not is_clean("two ripe Cherries")
    assert is_clean("Chop the onions and simmer for ten minutes.")
    print("part 1 ok: lexicon, episodes, counterfactual structure")


def _planted(n_topics=12, per=4, strength=0.6, seed=0):
    """Synthetic notes: with probability `strength` a class-specific word is added."""
    rng = np.random.default_rng(seed)
    marks = ["qwx zorb", "plim vask", "drel mott", "fyng hule"]
    texts, y, topics = [], [], []
    for t in range(n_topics):
        for c in range(4):
            for k in range(per):
                s = f"note {t} about thing {rng.integers(100)} here"
                if rng.random() < strength:
                    s += " " + marks[c]
                texts.append(s)
                y.append(c)
                topics.append(t)
    return texts, np.array(y), np.array(topics)


def part2_measure():
    rng = np.random.default_rng(1)
    texts, y, topics = _planted(strength=0.7)
    acc = ms.cv_text(texts, y, topics)
    assert acc > 0.6, acc  # planted signal is found
    null = [ms.cv_text(texts, ms.permute_within(y, topics, rng), topics) for _ in range(20)]
    assert np.mean(null) < 0.35, np.mean(null)  # and the null sits near chance (0.25)
    p = ms.perm_p(lambda yp: ms.cv_text(texts, yp, topics), acc, y, topics, 30, rng)
    assert p < 0.05, p
    # hidden probe: planted direction in one layer only
    n = len(y)
    X = rng.normal(size=(n, 3, 16))
    X[:, 1, :4] += np.eye(4)[y] * 3.0  # ideal 4-way accuracy ~0.95 at this strength
    accH, layers = ms.cv_hidden(X, y, topics, return_layers=True)
    assert accH > 0.7 and set(layers) == {1}, (accH, layers)
    Xn = rng.normal(size=(n, 3, 16))
    assert ms.cv_hidden(Xn, y, topics) < 0.4
    # behaviour delta
    P = np.full((n, 4), 0.25)
    assert abs(ms.delta(P, y)) < 1e-12
    P2 = P.copy()
    P2[np.arange(n), y] += 0.2
    P2 /= P2.sum(1, keepdims=True)
    d = ms.delta(P2, y)
    lo, hi = ms.boot_delta(P2, y, rng, n=200)
    assert d > 0.1 and lo > 0
    assert abs(ms.fano_bits(0.25)) < 1e-9 and abs(ms.fano_bits(1.0) - 2.0) < 1e-9
    assert 0 < ms.fano_bits(0.5) < 2
    # balancing
    idx, counts = ms.balance(np.arange(10), np.array([0] * 6 + [1] * 4), rng)
    assert len(idx) == 8
    print(f"part 2 ok: planted text {acc:.2f}, null {np.mean(null):.2f}, planted hidden {accH:.2f}, "
          f"delta {d:.2f} [{lo:.2f},{hi:.2f}]")


def part3_full():
    out = tempfile.mkdtemp()
    cmd = [sys.executable, "run.py", "--fake", "--topics", "8", "--scripts", "2",
           "--perms", "10", "--max-new", "30", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1800:])
    assert r.returncode == 0, r.stderr[-2000:]
    R = json.load(open(os.path.join(out, "results.json")))
    assert set(R["gates"]) >= {"V0", "V1", "V2", "V3", "V4", "V5", "V6"}
    assert R["v0_determinism"] is True
    # rerun resumes from the caches and gives identical numbers
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-2000:]
    R2 = json.load(open(os.path.join(out, "results.json")))
    assert R2["L2_all"]["text_acc"] == R["L2_all"]["text_acc"]
    print("part 3 ok: end to end on tiny random Qwen2; gate scores are meaningless")


if __name__ == "__main__":
    part1_lexicon_and_episodes()
    part2_measure()
    if "--full" in sys.argv:
        part3_full()
