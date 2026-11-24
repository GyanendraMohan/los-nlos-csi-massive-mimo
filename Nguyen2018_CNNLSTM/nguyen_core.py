"""Re-implementation of the CNN-LSTM of Nguyen et al. (Sensors 18(11), 4057, 2018) for the 80-split evaluation.

Usage: python nguyen_core.py --config results/<variant>/part_b_config.json --split '<split JSON>'
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN_LSTM"))
import cnn_lstm_core as base

inner_folds = base.inner_folds
make_sequences = base.make_sequences
Transform = base.Transform

DEFAULT_CONFIG = {
    "csi": "complex",
    "use_rssi": True,
    "antenna": 0,
    "input": "nguyen",
    "conv_filters": [32, 16, 8],
    "conv_kernels": [8, 2, 2],
    "conv_strides": [2, 1, 1],
    "conv_activations": ["softplus", "softplus", "relu"],
    "lstm_units": 10,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "max_epochs": 100,
    "inner_patience": 10,
    "window": 50,
    "stride": 10,
    "seed": 42,
}


def load_cluster_frames(data_dir, clusters, cfg):
    """Per-cluster frame features (before z-score) for the chosen input variant; RSSI is the last column."""
    if cfg["csi"] == "ours":
        if cfg["use_rssi"]:
            raise ValueError("csi='ours' uses per-antenna normalised amplitudes (no absolute power): set use_rssi=False")
        return base.load_cluster_frames(data_dir, clusters, "subcarrier")
    if cfg["csi"] not in ("complex", "amplitude"):
        raise ValueError(f"csi must be 'complex', 'amplitude' or 'ours', got {cfg['csi']!r}")

    cluster_dir = os.path.join(data_dir, "clusters")
    with open(os.path.join(cluster_dir, "index.json")) as f:
        cidx = json.load(f)
    frames, labels = {}, {}
    for c in clusters:
        info = cidx["clusters"][c]
        if "csi_file" not in info:
            raise FileNotFoundError("complex CSI not saved: re-run preprocess_raw.ipynb with SAVE_CSI = True")
        H = np.asarray(np.load(os.path.join(cluster_dir, info["csi_file"]), mmap_mode="r")[:, cfg["antenna"], :])
        if cfg["csi"] == "complex":
            X = np.empty((len(H), 2 * H.shape[1]), dtype=np.float32)
            X[:, 0::2], X[:, 1::2] = H.real, H.imag
        else:
            X = np.abs(H).astype(np.float32)
        if cfg["use_rssi"]:
            rssi = 10 * np.log10(np.mean(np.abs(H) ** 2, axis=1) + 1e-12)
            X = np.concatenate([X, rssi[:, None].astype(np.float32)], axis=1)
        frames[c] = X
        labels[c] = int(info["label"])
    return frames, labels


def build_model(cfg, window, n_features):
    """The CNN-LSTM of Nguyen et al."""
    n_csi = n_features - (1 if cfg["use_rssi"] else 0)
    inputs = keras.Input(shape=(window, n_features))
    csi = inputs[:, :, :n_csi] if cfg["use_rssi"] else inputs
    x = layers.Reshape((window, n_csi, 1), name="per_packet")(csi)
    for k, (f, ks, st, act) in enumerate(zip(cfg["conv_filters"], cfg["conv_kernels"], cfg["conv_strides"],
                                             cfg["conv_activations"]), 1):
        x = layers.Conv2D(f, (1, ks), strides=(1, st), padding="valid", activation=act, name=f"conv{k}")(x)
        x = layers.BatchNormalization(name=f"bn{k}")(x)
    x = layers.Reshape((window, x.shape[2] * x.shape[3]), name="flatten_per_packet")(x)
    x = layers.Dropout(cfg["dropout"], name="cnn_drop")(x)
    if cfg["use_rssi"]:
        x = layers.Concatenate(name="concat_rssi")([x, inputs[:, :, n_csi:]])
    x = layers.LSTM(cfg["lstm_units"], name="lstm")(x)
    x = layers.Dropout(cfg["dropout"], name="lstm_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_los")(x)
    model = keras.Model(inputs, outputs, name="nguyen2018_cnnlstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tuned_threshold(y, p):
    """Threshold that maximises the average LOS/NLOS detection rate (TPR + TNR) / 2, as in their Table 2."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    j = int(np.argmax((tpr - fpr)[ok]))
    t = float(thr[ok][j])
    pred = (p >= t).astype(np.int8)
    return t, float((pred == y).mean()), float((tpr[ok][j] + 1 - fpr[ok][j]) / 2)


def train_and_evaluate(cfg, frames, labels, split, verbose=False):
    """Epoch selection on held-out training clusters, final training, evaluation (shared procedure)."""
    result, details = base.train_and_evaluate(cfg, frames, labels, split, verbose=verbose, build_fn=build_model)
    y, p = details["y_true"], details["p_los"]
    pred = details["y_pred"]
    result["balanced_accuracy"] = float((result["recall_los"] + result["recall_nlos"]) / 2)
    t, acc_t, bal_t = tuned_threshold(y, p)
    result["tuned_threshold"] = t
    result["accuracy_tuned_threshold"] = acc_t
    result["balanced_accuracy_tuned_threshold"] = bal_t
    result["n_parameters"] = int(details["model"].count_params())
    assert len(pred) == len(y)
    return result, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="JSON with name, train and test cluster lists")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    split = json.loads(args.split)
    frames, labels = load_cluster_frames(cfg["data_dir"], split["train"] + split["test"], cfg)
    result, _ = train_and_evaluate(cfg, frames, labels, split)
    sys.stdout.write("RESULT " + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
