"""Readout statistics. All CV is grouped by topic so topic cannot carry the signal."""
import math
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

NFOLD = 4
RIDGE_ALPHA = 10.0  # see PREREG addendum 1


def permute_within(y, groups, rng):
    y2 = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y2[idx] = y[rng.permutation(idx)]
    return y2


def _majority(ytr, n):
    vals, cnt = np.unique(ytr, return_counts=True)
    return np.full(n, vals[np.argmax(cnt)])


def _text_predict(txt_tr, ytr, txt_te):
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                              sublinear_tf=True)
        Xtr = vec.fit_transform(txt_tr)
        Xte = vec.transform(txt_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
        return clf.predict(Xte)
    except ValueError:  # empty vocabulary (e.g. degenerate text)
        return _majority(ytr, len(txt_te))


def cv_text(texts, y, topics):
    texts = np.asarray(texts, dtype=object)
    folds = topics % NFOLD
    pred = np.empty_like(y)
    for f in range(NFOLD):
        te = folds == f
        if not te.any() or te.all():
            continue
        pred[te] = _text_predict(list(texts[~te]), y[~te], list(texts[te]))
    return float((pred == y).mean())


def _ridge_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def cv_hidden(X, y, topics, return_layers=False):
    """X: (n, n_layers, d). Layer picked by inner CV on training topics only."""
    n, L, _ = X.shape
    folds = topics % NFOLD
    pred = np.empty_like(y)
    chosen = []
    for f in range(NFOLD):
        te = folds == f
        tr = ~te
        if not te.any() or te.all():
            continue
        tr_idx = np.where(tr)[0]
        tr_topics = np.unique(topics[tr_idx])
        rank = {t: i for i, t in enumerate(tr_topics)}
        inner = np.array([rank[t] % 3 for t in topics[tr_idx]])
        best, best_acc = 0, -1.0
        for l in range(L):
            correct = 0
            for g in range(3):
                ite = inner == g
                if not ite.any() or ite.all():
                    continue
                p = _ridge_predict(X[tr_idx[~ite], l], y[tr_idx[~ite]], X[tr_idx[ite], l])
                correct += int((p == y[tr_idx[ite]]).sum())
            acc = correct / len(tr_idx)
            if acc > best_acc + 1e-12:
                best, best_acc = l, acc
        chosen.append(best)
        pred[te] = _ridge_predict(X[tr, best], y[tr], X[te, best])
    acc = float((pred == y).mean())
    return (acc, chosen) if return_layers else acc


def perm_p(stat_fn, observed, y, topics, n_perm, rng):
    """One-sided p: fraction of within-topic label shuffles with stat >= observed.
    The shuffle repeats the whole procedure (including layer selection)."""
    ge = 0
    for _ in range(n_perm):
        ge += stat_fn(permute_within(y, topics, rng)) >= observed - 1e-12
    return (ge + 1) / (n_perm + 1)


def delta(probs, y):
    ds = []
    for c in range(probs.shape[1]):
        m = y == c
        if m.all() or (~m).all():
            continue
        ds.append(probs[m, c].mean() - probs[~m, c].mean())
    return float(np.mean(ds)) if ds else 0.0


def boot_delta(probs, y, rng, n=2000):
    vals = np.empty(n)
    N = len(y)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = delta(probs[idx], y[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def top1(probs, y):
    return float((probs.argmax(1) == y).mean())


def fano_bits(acc, k=4):
    pe = 1.0 - acc
    if pe <= 0:
        return math.log2(k)
    h = 0.0 if pe >= 1 else -(pe * math.log2(pe) + (1 - pe) * math.log2(1 - pe))
    return max(0.0, math.log2(k) - h - pe * math.log2(k - 1))


def balance(indices, labels, rng):
    """Subsample so every label has the same count (the minimum). Returns indices."""
    indices = np.asarray(indices)
    labs = np.asarray(labels)[indices]
    counts = {c: int((labs == c).sum()) for c in np.unique(labs)}
    if not counts:
        return indices[:0], counts
    m = min(counts.values())
    keep = []
    for c in counts:
        pool = indices[labs == c]
        keep.append(rng.permutation(pool)[:m])
    return np.sort(np.concatenate(keep)) if keep else indices[:0], counts
