"""A writes a note, C reads it. Greedy, batch size 1, so reruns are exact."""
import json
import os

import numpy as np
import torch

from .episodes import a_messages, c_messages
from .lexicon import COLOURS


class Model:
    def __init__(self, tok, model, device):
        self.tok, self.model, self.device = tok, model, device
        self.model.eval()
        n = self.model.config.num_hidden_layers
        fr = (1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6)
        self.layers = sorted({min(n, max(1, int(round(n * f)))) for f in fr})
        self.colour_ids = self._colour_ids()

    def _colour_ids(self):
        ids = {}
        for c in COLOURS:
            s = set()
            for v in (c, c.capitalize(), " " + c, " " + c.capitalize()):
                e = self.tok.encode(v, add_special_tokens=False)
                if e:
                    s.add(int(e[0]))
            ids[c] = s
        # drop ids shared between colours (can only happen with odd tokenizers)
        for c in COLOURS:
            others = set().union(*(ids[o] for o in COLOURS if o != c))
            ids[c] = sorted(ids[c] - others)
            assert ids[c], f"no unique token id left for {c}"
        return ids

    def _prompt(self, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    @torch.no_grad()
    def write_note(self, ep, max_new=60):
        enc = self._prompt(a_messages(ep))
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  temperature=None, top_p=None, top_k=None,
                                  pad_token_id=getattr(self.tok, "pad_token_id", None)
                                  or getattr(self.tok, "eos_token_id", 0))
        gen = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def read_note(self, note):
        enc = self._prompt(c_messages(note))
        out = self.model(**enc, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        p = torch.softmax(logits, -1)
        probs = np.array([float(p[self.colour_ids[c]].sum()) for c in COLOURS])
        mass = float(probs.sum())
        probs = probs / mass if mass > 0 else np.full(4, 0.25)
        hid = np.stack([out.hidden_states[l][0, -1].float().cpu().numpy()
                        for l in self.layers])
        return probs, hid, mass


def load_model(name, device=None, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    return Model(tok, model, device)


def generate_all(M, episodes, path, max_new=60, log=print):
    """Resumable: notes cached in a JSONL file keyed by episode id."""
    notes = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                notes[r["id"]] = r["note"]
    todo = [e for e in episodes if e["id"] not in notes]
    log(f"notes: {len(notes)} cached, {len(todo)} to generate")
    with open(path, "a") as f:
        for i, e in enumerate(todo):
            notes[e["id"]] = M.write_note(e, max_new)
            f.write(json.dumps({"id": e["id"], "note": notes[e["id"]]}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                log(f"  generated {i + 1}/{len(todo)}")
    return notes


def read_all(M, episodes, notes, cache_dir, log=print):
    """Resumable: one small .npz per episode (probs, hidden, colour mass)."""
    os.makedirs(cache_dir, exist_ok=True)
    probs, hid, mass = [], [], []
    for i, e in enumerate(episodes):
        p = os.path.join(cache_dir, e["id"] + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            pr, h, m = z["probs"], z["hid"], float(z["mass"])
        else:
            pr, h, m = M.read_note(notes[e["id"]])
            np.savez_compressed(p, probs=pr, hid=h.astype(np.float16), mass=m)
        probs.append(pr)
        hid.append(h.astype(np.float32))
        mass.append(m)
        if (i + 1) % 100 == 0:
            log(f"  read {i + 1}/{len(episodes)}")
    return np.array(probs), np.stack(hid), np.array(mass)
