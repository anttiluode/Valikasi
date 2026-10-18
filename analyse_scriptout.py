"""Post-hoc, NOT pre-registered (written after run 2). Permutation p-values for the
leave-script-out accuracies, which run.py reports without a test.

Rebuilds exactly the balanced sets run.py used (same filter, same seed-0 balancing
order), then shuffles labels within script and reruns the whole leave-script-out CV.

  python analyse_scriptout.py results_3b Qwen_Qwen2_5_3B_Instruct [n_perm_text] [n_perm_hidden]
"""
import json
import sys

import numpy as np

from valikasi import measure as ms
from valikasi.episodes import build_episodes
from valikasi.lexicon import is_clean

d, cache = sys.argv[1], sys.argv[2]
n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
R = json.load(open(f"{d}/results.json"))
eps = build_episodes(R["args"]["topics"], R["args"]["scripts"])
notes = {}
for line in open(f"{d}/notes.jsonl"):
    r = json.loads(line)
    notes[r["id"]] = r["note"]
texts = [notes[e["id"]] for e in eps]
level = np.array([e["level"] for e in eps])
lab = np.array([e["label"] for e in eps])
scripts = np.array([e["script"] for e in eps])
clean = np.array([is_clean(t) for t in texts])
hid = np.stack([np.load(f"{d}/cache_{cache}/{e['id']}.npz")["hid"].astype(np.float32) for e in eps])

rng_bal = np.random.default_rng(0)  # same call order as run.py
sets = {}
for key, lv, co in [("L1_all", "L1", False), ("L1_clean", "L1", True),
                    ("L2_all", "L2", False), ("L2_clean", "L2", True)]:
    m = (level == lv) & (clean if co else True)
    sets[key], _ = ms.balance(np.where(m)[0], lab, rng_bal)

S = R["args"]["scripts"]
rng = np.random.default_rng(123)
n_perm_h = int(sys.argv[4]) if len(sys.argv) > 4 else 200
keys = ("L1_clean", "L2_all", "L2_clean")
data = {k: (lab[sets[k]], scripts[sets[k]], [texts[i] for i in sets[k]], hid[sets[k]]) for k in keys}
out = {}
for key in keys:  # text first: fast
    y, sp, t, X = data[key]
    at = ms.cv_text(t, y, sp, nfold=S)
    pt = ms.perm_p(lambda yp: ms.cv_text(t, yp, sp, nfold=S), at, y, sp, n_perm, rng)
    out[key] = dict(n=len(y), text_acc=at, text_p=pt, text_perms=n_perm)
    print(f"{key}: n={len(y)} text {at:.3f} (p={pt:.4f}, {n_perm} shuffles)", flush=True)
    json.dump(out, open(f"{d}/scriptout_perm.json", "w"), indent=2)
for key in keys:  # hidden: slow at 3B width, fewer shuffles
    y, sp, t, X = data[key]
    ah = ms.cv_hidden(X, y, sp, nfold=S)
    ph = ms.perm_p(lambda yp: ms.cv_hidden(X, yp, sp, nfold=S), ah, y, sp, n_perm_h, rng)
    out[key].update(hid_acc=ah, hid_p=ph, hid_perms=n_perm_h)
    print(f"{key}: hidden {ah:.3f} (p={ph:.4f}, {n_perm_h} shuffles)", flush=True)
    json.dump(out, open(f"{d}/scriptout_perm.json", "w"), indent=2)
