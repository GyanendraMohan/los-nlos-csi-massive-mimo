"""SVM and RF with hyperparameters selected by nested, cluster-wise validation on the training clusters."""
import json
import os
import time

import numpy as np

cm = None
_inner_folds = None
_DATA = {}

SEED = 42
SEARCH_STRIDE = 4

SVM_GRID = [{"C": C, "gamma_factor": g} for C in (1.0, 0.1, 10.0, 100.0) for g in (1.0, 0.1, 10.0)]
RF_GRID = [{"max_features": mf, "min_samples_leaf": leaf} for mf in ("sqrt", "log2", 0.05) for leaf in (1, 5, 20)]
GRIDS = {"svm": SVM_GRID, "rf": RF_GRID}
DEFAULT = {"svm": SVM_GRID[0], "rf": RF_GRID[0]}
FIXED = {"svm": {"kernel": "rbf", "gamma": "factor x scale, scale = 1 / (n_features * Var[X]) (scikit-learn 'scale')",
                 "standardisation": "per feature, statistics of the training frames", "threshold": "decision value 0"},
         "rf": {"n_estimators": 100, "criterion": "gini", "random_state": SEED, "threshold": "probability 0.5"}}


def setup(common_dir, cnn_core_path):
    """Import robustness/common.py and inner_folds() of cnn_lstm_core.py (without importing Keras)."""
    global cm, _inner_folds
    if cm is not None:
        return
    import importlib
    import sys
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
    cm = importlib.import_module("common")
    src = open(cnn_core_path).read()
    ns = {}
    exec(src[src.index("def inner_folds"):src.index("def select_epochs")], ns)
    _inner_folds = ns["inner_folds"]


def inner_folds(train_clusters):
    return _inner_folds(train_clusters)


def data(data_dir):
    """Feature files of one dataset, memory-mapped once per process (shared page cache between workers)."""
    if data_dir not in _DATA:
        with open(os.path.join(data_dir, "clusters", "index.json")) as f:
            cidx = json.load(f)
        assert cidx["norm"] == "antenna" and not cidx["zscore_applied"]
        X = {c: np.load(os.path.join(data_dir, "clusters", cidx["clusters"][c]["features_file"]), mmap_mode="r")
             for c in cm.ALL_CLUSTERS}
        y = {c: int(cidx["clusters"][c]["label"]) for c in cm.ALL_CLUSTERS}
        _DATA[data_dir] = (X, y)
    return _DATA[data_dir]


def stack(X, y, clusters, stride=1):
    Xs = np.concatenate([np.asarray(X[c][::stride], dtype=np.float32) for c in clusters])
    ys = np.concatenate([np.full(len(range(0, len(X[c]), stride)), y[c], dtype=np.int8) for c in clusters])
    return Xs, ys


def fit(model, params, Xtr, ytr, seed=SEED):
    """-> scorer(Z) (larger = more LOS), decide(score), threshold, info."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    if model == "svm":
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        Z = (Xtr - mu) / sd
        g = params["gamma_factor"]
        gamma = "scale" if g == 1.0 else g / (Z.shape[1] * Z.var())
        clf = SVC(kernel="rbf", C=params["C"], gamma=gamma, cache_size=1000).fit(Z, ytr)
        scorer = lambda A: clf.decision_function((np.asarray(A, dtype=np.float32) - mu) / sd)
        return scorer, lambda s: (s > 0).astype(np.int8), 0.0, {"n_support_vectors": int(clf.n_support_.sum())}
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_features=params["max_features"],
                                     min_samples_leaf=params["min_samples_leaf"], n_jobs=1,
                                     random_state=seed).fit(Xtr, ytr)
        scorer = lambda A: clf.predict_proba(np.asarray(A, dtype=np.float32))[:, 1]
        depth = int(np.mean([e.tree_.max_depth for e in clf.estimators_]))
        return scorer, lambda s: (s >= 0.5).astype(np.int8), 0.5, {"mean_tree_depth": depth}
    raise ValueError(model)


def balanced_accuracy(y_true, y_hat):
    return float(np.mean([(y_hat[y_true == k] == k).mean() for k in (0, 1)]))


def search_task(paths, data_dir, model, protocol, split, fold_k, grid_k, stride=SEARCH_STRIDE):
    """One (split, inner fold, grid point)."""
    setup(*paths)
    X, y = data(data_dir)
    fold = inner_folds(split["train"])[fold_k]
    assert not set(fold["train"] + fold["val"]) & set(split["test"]), "test cluster inside the inner search"
    assert set(fold["train"]) | set(fold["val"]) == set(split["train"])
    params = GRIDS[model][grid_k]
    t0 = time.time()
    Xtr, ytr = stack(X, y, fold["train"], stride)
    Xva, yva = stack(X, y, fold["val"], stride)
    scorer, decide, _, info = fit(model, params, Xtr, ytr)
    p = decide(scorer(Xva))
    return {"key": f"{split['name']}|{fold_k}|{grid_k}", "split": split["name"], "protocol": protocol, "model": model,
            "fold": fold_k, "grid": grid_k, "params": params, "val": fold["val"],
            "balanced_accuracy": balanced_accuracy(yva, p), "accuracy": float((p == yva).mean()),
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "seconds": time.time() - t0, **info}


def select(model, split, rows):
    """rows: search results of this split -> chosen grid index and the mean score of every grid point."""
    n_folds = len(inner_folds(split["train"]))
    mean = []
    for g in range(len(GRIDS[model])):
        r = [x for x in rows if x["grid"] == g]
        assert len(r) == n_folds, f"{split['name']}: grid point {g} has {len(r)} of {n_folds} folds"
        mean.append(float(np.mean([x["balanced_accuracy"] for x in r])))
    best = int(np.argmax(mean))
    return best, mean


def _evaluate(scorer, decide, clusters, Xs, y):
    """Per frame and per window, exactly as common.evaluate_frame_split."""
    f_true, f_hat, f_score, f_grp = [], [], [], []
    w_true, w_vote, w_mean, w_score, w_grp = [], [], [], [], []
    for c in clusters:
        s = scorer(Xs[c]).astype(np.float64)
        p = decide(s)
        yt = np.full(len(p), y[c], dtype=np.int8)
        f_true.append(yt); f_hat.append(p); f_score.append(s); f_grp.append(np.full(len(p), c))
        idx = cm.window_index(len(p))
        frac = p[idx].mean(axis=1)
        ms = s[idx].mean(axis=1)
        by_score = decide(ms)
        vote = np.where(frac > 0.5, 1, np.where(frac < 0.5, 0, by_score)).astype(np.int8)
        w_true.append(np.full(len(idx), y[c], dtype=np.int8)); w_vote.append(vote); w_mean.append(by_score)
        w_score.append(ms); w_grp.append(np.full(len(idx), c))
    cat = np.concatenate
    return {"frame": cm.metrics(cat(f_true), cat(f_hat), cat(f_score), cat(f_grp)),
            "window_vote": cm.metrics(cat(w_true), cat(w_vote), cat(w_score), cat(w_grp)),
            "window_mean_score": cm.metrics(cat(w_true), cat(w_mean), cat(w_score), cat(w_grp))}


def final_task(paths, data_dir, shift_dir, model, protocol, split, chosen, inner_means, stored=None):
    """Train the selected setting on all training frames, evaluate the test clusters."""
    setup(*paths)
    X, y = data(data_dir)
    assert not set(split["train"]) & set(split["test"])
    params = GRIDS[model][chosen]
    t0 = time.time()
    Xtr, ytr = stack(X, y, split["train"])
    scorer, decide, thr, info = fit(model, params, Xtr, ytr)
    fit_s = time.time() - t0
    del Xtr
    res = _evaluate(scorer, decide, split["test"], X, y)
    row = {"name": split["name"], "train": split["train"], "test": split["test"], "protocol": protocol,
           "model": model + "_tuned", "seed": SEED, "n_train": int(len(ytr)),
           "n_test": res["frame"]["n"], **res["frame"],
           "window_vote": res["window_vote"], "window_mean_score": res["window_mean_score"],
           "params": params, "grid_index": chosen, "default_chosen": chosen == 0,
           "inner_mean_balanced_accuracy": inner_means, "fit_seconds": fit_s, **info}
    if shift_dir:
        Xs, ys = data(shift_dir)
        assert ys == y
        sh = _evaluate(scorer, decide, split["test"], Xs, y)
        row["test_shift"] = {**sh["frame"], "window_vote": sh["window_vote"]}
    if stored is not None:
        row["stored_default"] = {k: stored[k] for k in ("accuracy", "recall_los", "recall_nlos", "n_test")
                                 if k in stored}
        assert stored["n_test"] == row["n_test"], "test frames differ from the stored results"
    return row
