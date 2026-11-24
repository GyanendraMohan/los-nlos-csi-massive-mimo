"""Inputs of the Zheng et al. re-implementation (single-antenna amplitudes and the SNR-condition noise).

Usage: python Zheng2023_SVM/prepare_inputs.py
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PV2 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = PV2
sys.path.insert(0, os.path.join(PV2, "robustness"))
sys.path.insert(0, HERE)
import common as cm
import zheng_core as zc

zc.setup(os.path.join(PV2, "robustness"), os.path.join(PV2, "CNN_LSTM", "cnn_lstm_core.py"))

INPUTS = os.path.join(HERE, "inputs")
ANTENNAS = [zc.MAIN_ANTENNA] + list(zc.SENS_ANTENNAS)
SNR_ANTENNAS = [zc.MAIN_ANTENNA]
CHECKSUMS = os.path.join(HERE, "existing_results.sha256.json")
BUNDLE = os.path.join(ROOT, "kaggle_zheng_bundle")
EXISTING = ["SVM/results", "RF/results", "CNN_LSTM/results", "CNN_Transformer/results", "Nguyen2018_CNNLSTM/results",
            "robustness/01_snr_equalisation/results", "robustness/02_frame_vs_window/results"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def existing_checksums():
    out = {}
    for d in EXISTING:
        for p in sorted(glob.glob(os.path.join(PV2, d, "**", "*"), recursive=True)):
            if os.path.isfile(p):
                out[os.path.relpath(p, PV2)] = sha256(p)
    return out


def main():
    t_all = time.time()
    if not os.path.exists(CHECKSUMS):
        cs = existing_checksums()
        with open(CHECKSUMS, "w") as f:
            json.dump(cs, f, indent=1)
        print(f"1. checksums of {len(cs)} existing result files -> {os.path.basename(CHECKSUMS)}")
    else:
        print("1. checksums already recorded (kept unchanged)")

    with open(os.path.join(cm.ORIG_DATA_DIR, "clusters", "index.json")) as f:
        oidx = json.load(f)
    labels = {c: int(oidx["clusters"][c]["label"]) for c in cm.ALL_CLUSTERS}
    index = {"source": "output_raw_full (calibrated CSI, before any normalisation)",
             "amplitude": "|H_t[a, k]| of the calibrated CSI, 52 data subcarriers, no normalisation, no Hampel filter",
             "labels": labels, "conditions": {}}

    os.makedirs(os.path.join(INPUTS, "original"), exist_ok=True)
    cond = {"antennas": ANTENNAS, "clusters": {}}
    for c in cm.ALL_CLUSTERS:
        H = np.load(cm.csi_path(c), mmap_mode="r")
        A = np.abs(np.asarray(H[:, ANTENNAS, :])).astype(np.float32)
        rel = os.path.join("original", f"{c}.npy")
        np.save(os.path.join(INPUTS, rel), A)
        cond["clusters"][c] = {"file": rel, "frames": int(len(A))}
        assert len(A) == oidx["clusters"][c]["frames"]
        print(f"2. {c:14s} {A.shape}")
    index["conditions"]["original"] = cond

    with open(os.path.join(cm.SNR_DATA_DIR, "clusters", "index.json")) as f:
        sidx = json.load(f)
    info = sidx["snr_equalisation"]
    seed, chunk = info["seed"], 500
    os.makedirs(os.path.join(INPUTS, "snr_equalised"), exist_ok=True)
    cond = {"antennas": SNR_ANTENNAS, "clusters": {}, "target_snr_db": info["target_snr_db"], "seed": seed,
            "replay_check_max_abs_diff": {}}
    for i, c in enumerate(cm.ALL_CLUSTERS):
        if c in cm.NLOS_CLUSTERS:
            cond["clusters"][c] = dict(index["conditions"]["original"]["clusters"][c])
            continue
        t0 = time.time()
        pn = cm.noise_floor(c)
        g = info["clusters"][c]["noise_factor_g"]
        sigma = np.sqrt(np.clip(g - 1, 0, None) * pn / 2).astype(np.float32)
        rng = np.random.default_rng([seed, i])
        H = np.load(cm.csi_path(c), mmap_mode="r")
        Xref = np.load(os.path.join(cm.SNR_DATA_DIR, "clusters", sidx["clusters"][c]["features_file"]), mmap_mode="r")
        A, worst = np.empty((len(H), len(SNR_ANTENNAS), 52), np.float32), 0.0
        for j in range(0, len(H), chunk):
            h = np.asarray(H[j:j + chunk])
            noise = (rng.standard_normal(h.shape, dtype=np.float32)
                     + 1j * rng.standard_normal(h.shape, dtype=np.float32)) * sigma
            hn = h + noise.astype(np.complex64)
            worst = max(worst, float(np.abs(cm.features_from_csi(hn) - np.asarray(Xref[j:j + len(h)])).max()))
            A[j:j + len(h)] = np.abs(hn[:, SNR_ANTENNAS, :])
        assert worst < 1e-4, f"{c}: replayed noise does not reproduce the stored SNR-equalised features ({worst})"
        rel = os.path.join("snr_equalised", f"{c}.npy")
        np.save(os.path.join(INPUTS, rel), A)
        cond["clusters"][c] = {"file": rel, "frames": int(len(A))}
        cond["replay_check_max_abs_diff"][c] = worst
        print(f"3. {c:14s} noise replayed, max |diff| to stored features {worst:.2e} ({time.time() - t0:.0f} s)")
    index["conditions"]["snr_equalised"] = cond
    with open(os.path.join(INPUTS, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    if os.path.isdir(BUNDLE):
        shutil.rmtree(BUNDLE)
    dst = os.path.join(BUNDLE, "Zheng2023_SVM")
    for rel in ["zheng_core.py", "zheng2023_svm.ipynb", "README.md", "inputs/index.json"] + \
               [v["file"] for cnd in index["conditions"].values() for v in cnd["clusters"].values()]:
        rel = rel if rel.startswith(("zheng", "README", "inputs")) else os.path.join("inputs", rel)
        src, out = os.path.join(HERE, rel), os.path.join(dst, rel)
        if not os.path.exists(src) or os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            os.link(src, out)
        except OSError:
            shutil.copyfile(src, out)
    with open(os.path.join(BUNDLE, "dataset-metadata.json"), "w") as f:
        json.dump({"title": "los-nlos-zheng-inputs", "id": "<kaggle-user>/los-nlos-zheng-inputs",
                   "licenses": [{"name": "other"}]}, f, indent=1)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(BUNDLE, "**", "*"), recursive=True) if os.path.isfile(p))
    print(f"4. Kaggle folder {os.path.relpath(BUNDLE, ROOT)} ({size / 1e6:.0f} MB). Done in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
