"""Shared code of the SNR-equalisation and frame-vs-window experiments."""
import itertools
import json
import os
import time

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
PV2 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = PV2
ORIG_DATA_DIR = os.path.join(PV2, "output_raw_full")
RAW_DIR = os.path.join(ROOT, "raw_traces")
SNR_DATA_DIR = os.path.join(HERE, "data", "snr_equalised")

LOS_CLUSTERS = ["LOS_cluster1", "LOS_cluster2", "LOS_cluster3", "LOS_cluster4"]
NLOS_CLUSTERS = ["NLOS_cluster1", "NLOS_cluster2", "NLOS_cluster3", "NLOS_cluster4", "NLOS_cluster5"]
ALL_CLUSTERS = LOS_CLUSTERS + NLOS_CLUSTERS
PROTOCOLS = {"lopo": "LOPO", "l2po": "L2PO"}
WINDOW, STRIDE = 50, 10
SEED = 42

short = lambda c: c.replace("_cluster", "")


def make_splits(protocol):
    """Identical to the SVM/RF notebooks: same order, same names."""
    if protocol == "lopo":
        tests = [[l, n] for l in LOS_CLUSTERS for n in NLOS_CLUSTERS]
    elif protocol == "l2po":
        tests = [list(ls) + list(ns) for ls in itertools.combinations(LOS_CLUSTERS, 2)
                 for ns in itertools.combinations(NLOS_CLUSTERS, 2)]
    else:
        raise ValueError(protocol)
    return [{"name": "test=" + "+".join(short(c) for c in test),
             "train": [c for c in ALL_CLUSTERS if c not in test], "test": test} for test in tests]


def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["name"]: r for r in (json.loads(line) for line in f if line.strip())}


def append_result(path, r):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(r) + "\n")


def safe_name(split_name):
    return split_name.replace("=", "_").replace("+", "-")


def metrics(y_true, y_hat, score, groups):
    """Same metrics as the existing runners; `score` is any score where larger = more LOS."""
    y_true, y_hat, score = np.asarray(y_true), np.asarray(y_hat), np.asarray(score)
    return {
        "accuracy": float((y_hat == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "recall_los": float(recall_score(y_true, y_hat, pos_label=1, zero_division=0)),
        "recall_nlos": float(recall_score(y_true, y_hat, pos_label=0, zero_division=0)),
        "per_cluster_accuracy": {c: float((y_hat[groups == c] == y_true[groups == c]).mean())
                                 for c in dict.fromkeys(groups.tolist())},
        "n": int(len(y_true)),
    }


def window_index(n_frames, window=WINDOW, stride=STRIDE):
    """Frame indices of the windows cut inside one cluster, exactly as cnn_lstm_core.make_sequences."""
    n = (n_frames - window) // stride + 1
    return stride * np.arange(n)[:, None] + np.arange(window)[None, :]


SAMPLES_PER_SLOT, LTS_OFFSET, FFT_SIZE, NOISE_SLOT, SKIP_FIRST = 224, 47, 64, 4, 200
LTS_FREQ = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
                     0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    dtype=np.float32)
NULL_BINS = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
DATA_BINS = np.array([b for b in range(FFT_SIZE) if b not in NULL_BINS])
ANT_IDX = np.delete(np.arange(4, 72), [32, 33, 34, 35])


def _lts_estimate(samples):
    """(frames, 72, 64, 2) int16 -> (frames, 64, 52) complex, the same estimator as the pipeline."""
    iq = (samples[..., 0] + 1j * samples[..., 1]).astype(np.complex64) * 2 ** -15
    h = np.fft.fftshift(np.fft.fft(iq, FFT_SIZE, axis=2), axes=2) * LTS_FREQ
    h = h[:, :, DATA_BINS]
    return h[:, ::-1, :][:, ANT_IDX, :]


def noise_floor(cluster, n_frames=None, chunk=1000):
    """Mean |N[a,k]|^2 (64 x 52) of the noise-only slot 4, passed through the same estimator as the pilot."""
    import h5py
    s = NOISE_SLOT * SAMPLES_PER_SLOT + LTS_OFFSET
    acc, count = np.zeros((64, 52)), 0
    with h5py.File(os.path.join(RAW_DIR, f"channe14_{cluster}.hdf5"), "r") as f:
        ds = f["Pilot_Samples"]
        stop = ds.shape[0] - 1 if n_frames is None else min(SKIP_FIRST + n_frames, ds.shape[0] - 1)
        for i in range(SKIP_FIRST, stop, chunk):
            n = _lts_estimate(ds[i:min(i + chunk, stop), :, s:s + FFT_SIZE, :])
            acc += (np.abs(n) ** 2).sum(axis=0)
            count += len(n)
    return acc / count


def csi_path(cluster, data_dir=ORIG_DATA_DIR):
    with open(os.path.join(data_dir, "clusters", "index.json")) as f:
        return os.path.join(data_dir, "clusters", json.load(f)["clusters"][cluster]["csi_file"])


def received_power(cluster, chunk=1000):
    """Mean |H[a,k]|^2 (64 x 52) of the calibrated CSI (signal + noise) over all kept frames."""
    H = np.load(csi_path(cluster), mmap_mode="r")
    acc = np.zeros((64, 52))
    for i in range(0, len(H), chunk):
        acc += (np.abs(np.asarray(H[i:i + chunk])) ** 2).sum(axis=0)
    return acc / len(H)


def measure_snr(clusters=ALL_CLUSTERS, n_noise_frames=None, verbose=True):
    """SNR per cluster: signal power (received minus noise) over noise power, summed over antennas and subcarriers."""
    out = {}
    for c in clusters:
        pn = noise_floor(c, n_noise_frames)
        pr = received_power(c)
        ps = np.clip(pr - pn, 0, None)
        out[c] = {"noise": pn, "signal": ps, "snr_db": float(10 * np.log10(ps.sum() / pn.sum()))}
        if verbose:
            print(f"  {c:14s} SNR {out[c]['snr_db']:5.1f} dB")
    return out


def features_from_csi(H):
    """Per-antenna-normalised amplitude, flattened: exactly preprocess_raw.ipynb with NORM='antenna', FEATURE='full'."""
    amp = np.abs(H).astype(np.float32)
    amp = amp / (amp.mean(axis=2, keepdims=True) + 1e-8)
    return amp.reshape(len(amp), -1)


def check_reproduction(cluster, n=2000):
    """Features recomputed from the stored CSI must equal the stored features (no noise added)."""
    H = np.asarray(np.load(csi_path(cluster), mmap_mode="r")[:n])
    with open(os.path.join(ORIG_DATA_DIR, "clusters", "index.json")) as f:
        ff = json.load(f)["clusters"][cluster]["features_file"]
    X = np.asarray(np.load(os.path.join(ORIG_DATA_DIR, "clusters", ff), mmap_mode="r")[:n])
    return float(np.abs(features_from_csi(H) - X).max())


def build_snr_equalised_dataset(snr, target_db=None, out_dir=SNR_DATA_DIR, seed=0, chunk=500):
    """Writes a copy of output_raw_full/clusters in which the LOS clusters carry extra receiver noise."""
    if target_db is None:
        target_db = float(np.mean([snr[c]["snr_db"] for c in NLOS_CLUSTERS]))
    cdir = os.path.join(out_dir, "clusters")
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(ORIG_DATA_DIR, "clusters", "index.json")) as f:
        index = json.load(f)
    info = {"target_snr_db": target_db, "seed": seed, "clusters": {}}
    for i, c in enumerate(ALL_CLUSTERS):
        ff = index["clusters"][c]["features_file"]
        dst = os.path.join(cdir, ff)
        if os.path.lexists(dst):
            os.remove(dst)
        if c in NLOS_CLUSTERS:
            rel = os.path.relpath(os.path.join(ORIG_DATA_DIR, "clusters", ff), cdir)
            index["clusters"][c]["features_file"] = rel
            info["clusters"][c] = {"noise_added": False, "snr_db": snr[c]["snr_db"]}
            continue
        pn, ps = snr[c]["noise"], snr[c]["signal"]
        g = (ps.sum() / 10 ** (target_db / 10)) / pn.sum()
        sigma = np.sqrt(np.clip(g - 1, 0, None) * pn / 2).astype(np.float32)
        rng = np.random.default_rng([seed, i])
        H = np.load(csi_path(c), mmap_mode="r")
        X = open_memmap(dst, mode="w+", dtype=np.float32, shape=(len(H), 64 * 52))
        for j in range(0, len(H), chunk):
            h = np.asarray(H[j:j + chunk])
            noise = (rng.standard_normal(h.shape, dtype=np.float32)
                     + 1j * rng.standard_normal(h.shape, dtype=np.float32)) * sigma
            X[j:j + len(h)] = features_from_csi(h + noise.astype(np.complex64))
        X.flush()
        del X
        achieved = 10 * np.log10(ps.sum() / (pn.sum() * g))
        info["clusters"][c] = {"noise_added": True, "noise_factor_g": float(g),
                               "snr_before_db": snr[c]["snr_db"], "snr_after_db": float(achieved)}
    index["snr_equalisation"] = info
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return info


def frequency_cv(data_dir, cluster, n=2000):
    """Median coefficient of variation across the 52 subcarriers (per frame and antenna)."""
    with open(os.path.join(data_dir, "clusters", "index.json")) as f:
        ff = json.load(f)["clusters"][cluster]["features_file"]
    A = np.asarray(np.load(os.path.join(data_dir, "clusters", ff), mmap_mode="r")[:n]).reshape(-1, 64, 52)
    return float(np.median(A.std(axis=2) / A.mean(axis=2)))


def load_frame_features(data_dir, clusters=ALL_CLUSTERS):
    with open(os.path.join(data_dir, "clusters", "index.json")) as f:
        cidx = json.load(f)
    assert cidx["norm"] == "antenna" and not cidx["zscore_applied"]
    X = {c: np.load(os.path.join(data_dir, "clusters", cidx["clusters"][c]["features_file"])) for c in clusters}
    y = {c: int(cidx["clusters"][c]["label"]) for c in clusters}
    return X, y


def fit_frame_model(model, X, y, train_clusters, seed=SEED):
    """Same settings as SVM/svm_los_nlos.ipynb and RF/rf_los_nlos.ipynb (evaluate_split)."""
    Xtr = np.concatenate([X[c] for c in train_clusters])
    ytr = np.concatenate([np.full(len(X[c]), y[c], dtype=np.int8) for c in train_clusters])
    t0 = time.time()
    if model == "svm":
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit((Xtr - mu) / sd, ytr)
        scorer = lambda Z: clf.decision_function((Z - mu) / sd)
        decide = lambda s: (s > 0).astype(np.int8)
        threshold = 0.0
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_features="sqrt",
                                     n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        scorer = lambda Z: clf.predict_proba(Z)[:, 1]
        decide = lambda s: (s >= 0.5).astype(np.int8)
        threshold = 0.5
    else:
        raise ValueError(model)
    return scorer, decide, threshold, time.time() - t0, int(len(ytr))


def evaluate_frame_split(model, split, X_train, y, test_sets, seed=SEED, scores_dir=None):
    """Train on `X_train`; evaluate each test set in `test_sets` ({tag: feature dict}) per frame and per window."""
    scorer, decide, thr, fit_s, n_train = fit_frame_model(model, X_train, y, split["train"], seed)
    out = {"name": split["name"], "train": split["train"], "test": split["test"], "model": model,
           "seed": seed, "n_train": n_train, "fit_seconds": fit_s}
    for tag, Xs in test_sets.items():
        f_true, f_hat, f_score, f_grp = [], [], [], []
        w_true, w_vote, w_mean, w_score, w_grp = [], [], [], [], []
        saved = {}
        for c in split["test"]:
            s = scorer(Xs[c]).astype(np.float64)
            p = decide(s)
            yt = np.full(len(p), y[c], dtype=np.int8)
            f_true.append(yt); f_hat.append(p); f_score.append(s); f_grp.append(np.full(len(p), c))
            idx = window_index(len(p))
            frac = p[idx].mean(axis=1)
            ms = s[idx].mean(axis=1)
            by_score = decide(ms)
            vote = np.where(frac > 0.5, 1, np.where(frac < 0.5, 0, by_score)).astype(np.int8)
            w_true.append(np.full(len(idx), y[c], dtype=np.int8)); w_vote.append(vote); w_mean.append(by_score)
            w_score.append(ms); w_grp.append(np.full(len(idx), c))
            saved[c] = s.astype(np.float32)
        cat = np.concatenate
        out[tag] = {
            "frame": metrics(cat(f_true), cat(f_hat), cat(f_score), cat(f_grp)),
            "window_vote": metrics(cat(w_true), cat(w_vote), cat(w_score), cat(w_grp)),
            "window_mean_score": metrics(cat(w_true), cat(w_mean), cat(w_score), cat(w_grp)),
        }
        if scores_dir:
            d = os.path.join(scores_dir, model, tag)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, safe_name(split["name"]) + ".npz"),
                                threshold=thr, **{c: v for c, v in saved.items()})
    return out
