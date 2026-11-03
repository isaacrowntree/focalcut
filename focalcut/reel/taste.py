"""Taste: a score learned from the editor's own picks, for ranking windows the way they would.

Every unit's frames are embedded twice a second by the expression step (scores/taste_<unit>.npz: the whole frame's
SigLIP 2 embedding). The labels come from picks.json: frames inside an "add" are what the editor wants (single-time
adds: a second either side), frames inside a "ban" are not, and frames drawn at random from the rest of the footage
stand for what they passed over. Each project's labelled frames go into a pool shared by every project (POOL), so a
shoot with few picks borrows the taste learnt on others; a logistic regression (balanced classes) over the pool then
scores any frame 0..1. Cross-validated by clip (held-out clips are never trained on), the AUC on this project's own
labels says how far to trust it. The model is retrained whenever picks.json or the pool changes.
"""
from __future__ import annotations

import os

import numpy as np

POOL = os.path.expanduser("~/.config/focalcut/taste.npz")
RANDOM_PER_UNIT = 4                               # random frames per unit as negatives
BAN_ALL_MAX = 12                                  # frames taken from a clip banned whole


def track(project, u):
    f = project.path("scores", f'taste_{u["id"].replace("#", "-")}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


def spans(entry, x0):
    """A picks entry's (adds, bans) as source-second ranges (export time + x0); "all" bans the whole unit."""
    from .plan import _secs

    def rng(txt, pad):
        if txt == "all":
            return (-1e9, 1e9)
        if isinstance(txt, str) and "-" in txt.strip()[1:]:
            a, b = txt.split("-", 1)
            return x0 + _secs(a), x0 + _secs(b)
        t = x0 + _secs(txt)
        return t - pad, t + pad
    return [rng(v, 1.0) for v in entry.get("add", [])], [rng(v, 0.0) for v in entry.get("ban", [])]


def inside(t, rs):
    """Which times fall in any of these ranges."""
    return np.any([(t >= a) & (t <= b) for a, b in rs], 0) if rs else np.zeros(len(t), bool)


def labelled(project, seed=0):
    """This project's labelled frames: X (n, D) float16, y (n,) 1 = picked, kind (n,) "add" / "ban" / "random",
    group (n,) the camera clip (for cross-validation)."""
    rng = np.random.default_rng(seed)
    picks = {k: v for k, v in project.picks().items() if not k.startswith("_") and isinstance(v, dict)}
    X, y, kind, grp = [], [], [], []
    for u in project.load("units.json"):
        tr = track(project, u)
        if tr is None or not len(tr["t"]):
            continue
        e = picks.get(u["id"]) or picks.get(u["id"].split("#")[0]) or {}
        adds, bans = spans(e, float(u.get("export_t0", 0.0)))
        t = tr["t"]
        pos, neg = inside(t, adds), inside(t, bans) & ~inside(t, adds)
        if (-1e9, 1e9) in bans and neg.sum() > BAN_ALL_MAX:
            keep = rng.choice(np.nonzero(neg)[0], BAN_ALL_MAX, replace=False)
            neg = np.isin(np.arange(len(t)), keep)
        free = np.nonzero(~pos & ~neg)[0]
        rnd = rng.choice(free, min(RANDOM_PER_UNIT, len(free)), replace=False) if len(free) else []
        for idx, lab, k in ((np.nonzero(pos)[0], 1, "add"), (np.nonzero(neg)[0], 0, "ban"), (rnd, 0, "random")):
            for i in idx:
                X.append(tr["emb"][i]); y.append(lab); kind.append(k); grp.append(u["clip"])
    if not X:
        return None
    return dict(X=np.array(X, np.float16), y=np.array(y), kind=np.array(kind), group=np.array(grp))


def pool_add(name, lab):
    """Put a project's labelled frames into the shared pool (replacing what it had there)."""
    pool = dict(np.load(POOL, allow_pickle=False)) if os.path.exists(POOL) else {}
    for k in ("X", "y", "kind", "group"):
        pool[f"{name}|{k}"] = lab[k]
    os.makedirs(os.path.dirname(POOL), exist_ok=True)
    tmp = POOL + ".tmp.npz"
    np.savez_compressed(tmp, **pool)
    os.replace(tmp, POOL)


def pooled(exclude=None):
    """Every project's labelled frames in the pool but exclude's: X, y."""
    if not os.path.exists(POOL):
        return np.zeros((0, 768), np.float16), np.zeros(0, int)
    pool = np.load(POOL, allow_pickle=False)
    names = sorted({k.split("|")[0] for k in pool.files} - {exclude})
    if not names:
        return np.zeros((0, 768), np.float16), np.zeros(0, int)
    return np.concatenate([pool[f"{n}|X"] for n in names]), np.concatenate([pool[f"{n}|y"] for n in names])


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X.astype(np.float32), y)


def cv_auc(lab, Xo, yo, folds=5):
    """AUC on this project's labels, by held-out clip: each fold trains on the other projects plus this project's
    other clips."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    g = lab["group"]
    n = min(folds, len(set(g)))
    if n < 2 or lab["y"].sum() < 3 or (lab["y"] == 0).sum() < 3:
        return None
    pred = np.zeros(len(g))
    for tr_i, te_i in GroupKFold(n).split(lab["X"], lab["y"], g):
        X = np.concatenate([Xo, lab["X"][tr_i]]); y = np.concatenate([yo, lab["y"][tr_i]])
        if len(set(y)) < 2:
            return None
        pred[te_i] = fit(X, y).predict_proba(lab["X"][te_i].astype(np.float32))[:, 1]
    return float(roc_auc_score(lab["y"], pred))


def model(project):
    """(w, b, auc, n) for this project: weights over the embedding, the cross-validated AUC (None when there are too
    few labels to tell) and the number of labelled frames trained on; None with no labels anywhere. Cached in
    taste_model.npz until picks.json or the pool changes."""
    cache = project.path("taste_model.npz")
    stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else 0.0 for f in (project.path("picks.json"), POOL))
    if os.path.exists(cache):
        z = np.load(cache)
        if tuple(z["stamp"]) == stamp:
            auc = float(z["auc"])
            return z["w"], float(z["b"]), (None if np.isnan(auc) else auc), int(z["n"])
    name = project.name
    lab = labelled(project)
    if lab is not None:
        pool_add(name, lab)
    Xo, yo = pooled(exclude=name)
    X = np.concatenate([Xo, lab["X"]]) if lab is not None else Xo
    y = np.concatenate([yo, lab["y"]]) if lab is not None else yo
    if len(set(y.tolist())) < 2:
        return None
    clf = fit(X, y)
    auc = cv_auc(lab, Xo, yo) if lab is not None else None
    w, b = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else 0.0 for f in (project.path("picks.json"), POOL))
    np.savez(cache, w=w, b=b, auc=np.nan if auc is None else auc, n=len(y), stamp=np.array(stamp))
    return w, b, auc, len(y)


def series(tr, w, b):
    """The taste score (0..1) of every frame in a unit's track."""
    return 1 / (1 + np.exp(-(tr["emb"].astype(np.float32) @ w + b)))


def window(tr, s, t0, t1):
    """Mean taste over a window (NaN: no frame in it)."""
    if tr is None:
        return np.nan
    m = (tr["t"] >= t0) & (tr["t"] <= t1)
    return float(s[m].mean()) if m.any() else np.nan
