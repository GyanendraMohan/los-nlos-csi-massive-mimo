"""Re-implementation of the NLOS detector of Zheng et al. (Remote Sensing 15(10), 2676, 2023) for the 80-split evaluation."""
import json
import os
import time

import numpy as np
from scipy import stats as sstats
from sklearn.svm import SVC

MAIN_KERNEL = "rbf"
TABLE3_KERNELS = ["rbf", "linear", "poly", "sigmoid"]
THRESHOLD_METHODS = ["rician_k", "skewness"]
UNSTATED = {
    "hampel": {"half_width": 3, "n_sigmas": 3.0, "mad_scale": 1.4826, "axis": "time, per subcarrier"},
    "svc": {"C": 1.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
}
DIAGNOSTICS = ["gain_only"]
MAIN_ANTENNA = 0
SENS_ANTENNAS = {7: "corner", 63: "corner", 32: "edge", 27: "centre"}
SAME_ENV = {"chrono": {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}}

cm = None


def setup(common_dir, *_):
    """Import robustness/common.py."""
    global cm
    if cm is not None:
        return
    import importlib
    import sys
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
    cm = importlib.import_module("common")


def svm_split_worker(paths, *args, **kw):
    setup(*paths)
    return run_svm_split(*args, **kw)


def load_inputs(inputs_dir, condition="original"):
    """-> amplitudes {cluster: (frames, n_antennas, 52) float32 raw |H|}, labels {cluster: 0/1}, antenna list."""
    with open(os.path.join(inputs_dir, "index.json")) as f:
        idx = json.load(f)
    cond = idx["conditions"][condition]
    A = {c: np.load(os.path.join(inputs_dir, v["file"])) for c, v in cond["clusters"].items()}
    y = {c: int(idx["labels"][c]) for c in idx["labels"]}
    return A, y, cond["antennas"], idx


def hampel(x, half_width=None, n_sigmas=None, mad_scale=None):
    """Hampel filter along axis 0 (time) of x (frames, subcarriers), window 2*half_width + 1 frames, shrinking at..."""
    h = UNSTATED["hampel"]
    half_width = h["half_width"] if half_width is None else half_width
    n_sigmas = h["n_sigmas"] if n_sigmas is None else n_sigmas
    mad_scale = h["mad_scale"] if mad_scale is None else mad_scale
    x = np.asarray(x, dtype=np.float64)
    pad = np.full((half_width,) + x.shape[1:], np.nan)
    win = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x, pad]), 2 * half_width + 1, axis=0)
    med = np.nanmedian(win, axis=-1)
    mad = mad_scale * np.nanmedian(np.abs(win - med[..., None]), axis=-1)
    out = np.abs(x - med) > n_sigmas * mad
    return np.where(out, med, x).astype(np.float32), out


def filter_clusters(A, antenna_pos):
    """Hampel-filter the amplitudes of one antenna, cluster by cluster."""
    F, share = {}, {}
    for c, a in A.items():
        f, mask = hampel(a[:, antenna_pos, :])
        assert len(f) == len(a)
        F[c], share[c] = f, float(mask.mean())
    return F, share


def minmax_per_vector(f):
    """Algorithm 1, step 4: x_ij = (x_ij - x_min,j) / (x_max,j - x_min,j) over the I subcarriers of each CSI..."""
    f = np.asarray(f, dtype=np.float64)
    lo, hi = f.min(axis=1, keepdims=True), f.max(axis=1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return ((f - lo) / rng).astype(np.float32)


def svm_features(F):
    return {c: minmax_per_vector(f) for c, f in F.items()}


def frame_statistic(f, method):
    """Per-frame statistic over the 52 subcarriers of one antenna (f: frames x 52, filtered raw amplitudes)."""
    f = np.asarray(f, dtype=np.float64)
    if method == "rician_k":
        p = f ** 2
        gamma = p.var(axis=1) / p.mean(axis=1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(np.clip(1 - gamma, 0, None))
            k = np.where(gamma >= 1, 0.0, r / (1 - r))
        return np.nan_to_num(k, nan=0.0, posinf=1e6)
    if method == "skewness":
        return sstats.skew(f, axis=1, bias=False)
    if method == "gain_only":
        return 10 * np.log10((f ** 2).mean(axis=1) + 1e-30)
    raise ValueError(method)


def stack(F, y, clusters):
    X = np.concatenate([F[c] for c in clusters])
    t = np.concatenate([np.full(len(F[c]), y[c], dtype=np.int8) for c in clusters])
    return X, t


def fit_svm(X, t, kernel, cache_size=200, max_iter=-1):
    """X is already min-max normalised per vector (Algorithm 1), so no scaler is fitted."""
    t0 = time.time()
    clf = SVC(kernel=kernel, cache_size=cache_size, max_iter=max_iter, **UNSTATED["svc"]).fit(X, t)
    fit_s = time.time() - t0
    assert list(clf.classes_) == [0, 1]
    return clf.decision_function, {"fit_seconds": fit_s, "n_support_vectors": int(clf.n_support_.sum())}


def fit_threshold(s, t):
    """Direction d and threshold th maximising training accuracy of the rule 'LOS if d * s > th'."""
    s = np.asarray(s, dtype=np.float64)
    best = (-1.0, 1, 0.0)
    n, n_los = len(t), int((t == 1).sum())
    for d in (1, -1):
        v = d * s
        order = np.argsort(v, kind="mergesort")
        vs, ts = v[order], t[order]
        acc = (np.cumsum(ts == 0) + (n_los - np.cumsum(ts == 1))) / n
        cand = np.flatnonzero(np.r_[vs[1:] != vs[:-1], True])
        i = cand[np.argmax(acc[cand])]
        a = acc[i]
        th = (vs[i] + vs[i + 1]) / 2 if i + 1 < n else vs[i] + 1.0
        if n_los / n > a:
            a, th = n_los / n, vs[0] - 1.0
        if a > best[0]:
            best = (float(a), d, float(th))
    acc, d, th = best
    return (lambda z: d * np.asarray(z, dtype=np.float64) - th), {"direction": int(d), "threshold": th,
                                                                   "train_accuracy": acc}


def evaluate(score_fn, F_test, y, clusters):
    """Per frame and per window (50 frames, stride 10, inside one cluster), with the rules of..."""
    f_true, f_hat, f_score, f_grp = [], [], [], []
    w_true, w_vote, w_mean, w_score, w_grp = [], [], [], [], []
    for c in clusters:
        s = np.asarray(score_fn(F_test[c]), dtype=np.float64)
        p = (s > 0).astype(np.int8)
        yt = np.full(len(p), y[c], dtype=np.int8)
        f_true.append(yt); f_hat.append(p); f_score.append(s); f_grp.append(np.full(len(p), c))
        idx = cm.window_index(len(p))
        if len(idx):
            frac, ms = p[idx].mean(axis=1), s[idx].mean(axis=1)
            by_score = (ms > 0).astype(np.int8)
            vote = np.where(frac > 0.5, 1, np.where(frac < 0.5, 0, by_score)).astype(np.int8)
            w_true.append(np.full(len(idx), y[c], dtype=np.int8)); w_vote.append(vote); w_mean.append(by_score)
            w_score.append(ms); w_grp.append(np.full(len(idx), c))
    cat = np.concatenate
    out = {"frame": cm.metrics(cat(f_true), cat(f_hat), cat(f_score), cat(f_grp))}
    if w_true:
        out["window_vote"] = cm.metrics(cat(w_true), cat(w_vote), cat(w_score), cat(w_grp))
        out["window_mean_score"] = cm.metrics(cat(w_true), cat(w_mean), cat(w_score), cat(w_grp))
    return out


def check_split(split):
    """Leakage check: test clusters are never used for training or thresholds."""
    tr, te = set(split["train"]), set(split["test"])
    assert not tr & te, split["name"]
    assert tr | te == set(cm.ALL_CLUSTERS), split["name"]


def row(split, n_train, res, **extra):
    """Per-split row in the format of SVM/results/splits_*.jsonl (frame metrics at the top level)."""
    fr = res["frame"]
    r = {"name": split["name"], "train": split["train"], "test": split["test"], "n_train": int(n_train),
         "n_test": fr["n"], **{k: fr[k] for k in ("accuracy", "macro_f1", "roc_auc", "recall_los", "recall_nlos",
                                                  "per_cluster_accuracy")}}
    r.update({k: v for k, v in res.items() if k != "frame"})
    r.update(extra)
    return r


def run_svm_split(split, F, y, F_shift=None, table3=True, **svm_kw):
    """RBF SVM (the kernel of the paper) trained on all training clusters, tested per frame and per window, and..."""
    check_split(split)
    Xtr, ttr = stack(F, y, split["train"])
    kernels = TABLE3_KERNELS if table3 else [MAIN_KERNEL]
    models, info = {}, {}
    for k in kernels:
        models[k], info[k] = fit_svm(Xtr, ttr, k, **svm_kw)
    res = evaluate(models[MAIN_KERNEL], F, y, split["test"])
    extra = {"method": "svm", "kernel": MAIN_KERNEL, "seed": cm.SEED, **info[MAIN_KERNEL]}
    if table3:
        extra["table3_kernels"] = {}
        for k in kernels:
            r = evaluate(models[k], F, y, split["test"])
            extra["table3_kernels"][k] = {"accuracy": r["frame"]["accuracy"],
                                          "recall_nlos": r["frame"]["recall_nlos"], "recall_los": r["frame"]["recall_los"],
                                          "roc_auc": r["frame"]["roc_auc"],
                                          "window_vote_accuracy": r.get("window_vote", {}).get("accuracy"), **info[k]}
    if F_shift is not None:
        extra["test_shift"] = evaluate(models[MAIN_KERNEL], F_shift, y, split["test"])
    return row(split, len(ttr), res, **extra)


def run_threshold_split(split, S, y, method, S_shift=None):
    """S: {cluster: per-frame statistic}."""
    check_split(split)
    s, t = stack(S, y, split["train"])
    score, par = fit_threshold(s, t)
    res = evaluate(score, S, y, split["test"])
    extra = {"method": method, **par}
    if S_shift is not None:
        extra["test_shift"] = evaluate(score, S_shift, y, split["test"])
    return row(split, len(t), res, **extra)


def same_env_parts(A_raw, antenna_pos, scheme, seed=None):
    """Frames of every cluster split into training and test frames (test frames from SEEN environments)."""
    parts, idx = {"train": {}, "test": {}}, {"train": {}, "test": {}}
    for c, a in A_raw.items():
        n = len(a)
        if scheme == "chrono":
            for p in ("train", "test"):
                lo, hi = SAME_ENV["chrono"][p]
                i = np.arange(int(round(lo * n)), int(round(hi * n)))
                parts[p][c], _ = hampel(a[i, antenna_pos, :])
                idx[p][c] = i
        elif scheme == "random":
            f, _ = hampel(a[:, antenna_pos, :])
            perm = np.random.default_rng([seed, cm.ALL_CLUSTERS.index(c)]).permutation(n)
            n_tr = int(round(0.70 * n))
            for p, i in (("train", perm[:n_tr]), ("test", perm[n_tr:])):
                i = np.sort(i)
                parts[p][c], idx[p][c] = f[i], i
        else:
            raise ValueError(scheme)
        assert not set(idx["train"][c]) & set(idx["test"][c]), c
    return parts, idx


def same_env_svm(parts, y, **svm_kw):
    C = cm.ALL_CLUSTERS
    feat = {p: svm_features(parts[p]) for p in parts}
    Xtr, ttr = stack(feat["train"], y, C)
    score, info = fit_svm(Xtr, ttr, MAIN_KERNEL, **svm_kw)
    return {"kernel": MAIN_KERNEL, "n_train": int(len(ttr)), **info, **evaluate(score, feat["test"], y, C)}


def same_env_threshold(parts, y, method):
    C = cm.ALL_CLUSTERS
    S = {p: {c: frame_statistic(v, method) for c, v in parts[p].items()} for p in parts}
    s, t = stack(S["train"], y, C)
    score, par = fit_threshold(s, t)
    return {**par, "n_train": int(len(t)), **evaluate(score, S["test"], y, C)}


def same_env_our_svm(X, y, idx):
    """Our SVM (common.fit_frame_model settings: z-score on training frames, RBF, C=1, gamma='scale') on the same..."""
    C = cm.ALL_CLUSTERS
    tr = {c: np.asarray(X[c][idx["train"][c]]) for c in C}
    te = {c: np.asarray(X[c][idx["test"][c]]) for c in C}
    Xtr, ttr = stack(tr, y, C)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    t0 = time.time()
    clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit((Xtr - mu) / sd, ttr)
    fit_s = time.time() - t0
    del Xtr
    res = evaluate(lambda Z: clf.decision_function((Z - mu) / sd), te, y, C)
    return {"kernel": "rbf", "n_train": int(len(ttr)), "fit_seconds": fit_s,
            "n_support_vectors": int(clf.n_support_.sum()), **res}
