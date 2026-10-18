"""Post-hoc, NOT pre-registered (written after run 1). Separates two things the gates mix:

  influence    - does changing Z change A's note at all?  (counterfactual divergence)
  information  - does the change carry Z?                 (the gates)

  python analyse_divergence.py [results_dir]
"""
import collections
import json
import sys

from valikasi.lexicon import COLOURS, colour_names_only

d = sys.argv[1] if len(sys.argv) > 1 else "results"
notes = {}
for line in open(f"{d}/notes.jsonl"):
    r = json.loads(line)
    notes[r["id"]] = r["note"]
R = json.load(open(f"{d}/results.json"))
S, T = R["args"]["scripts"], R["args"]["topics"]

for lv in ("L1", "L2"):
    dist = collections.Counter()
    for s in range(S):
        for t in range(T):
            ns = [notes[f"{lv}-{c}-s{s}-t{t}"] for c in COLOURS]
            dist[len(set(ns))] += 1
    changed = 1 - dist[1] / (S * T)
    print(f"{lv}: distinct notes per 4-way counterfactual group {dict(sorted(dist.items()))}; "
          f"Z changed the note in {changed:.0%} of groups")

own = {c: sum(c in colour_names_only(notes[f"L1-{c}-s{s}-t{t}"])
              for s in range(S) for t in range(T)) for c in COLOURS}
n = S * T
print(f"L1: note names B's own colour {sum(own.values())}/{4 * n} "
      f"({sum(own.values()) / (4 * n):.1%}); per colour {own}")
