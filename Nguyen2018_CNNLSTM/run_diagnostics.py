"""Diagnostics of the Nguyen et al. re-implementation (fixed-epoch curves and same-environment test).

Usage: python run_diagnostics.py [--variants complex ours] [--skip-a] [--skip-b]
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nguyen_core as core

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "output_raw_full"))
LOS_CLUSTERS = ["LOS_cluster1", "LOS_cluster2", "LOS_cluster3", "LOS_cluster4"]
NLOS_CLUSTERS = ["NLOS_cluster1", "NLOS_cluster2", "NLOS_cluster3", "NLOS_cluster4", "NLOS_cluster5"]
ALL_CLUSTERS = LOS_CLUSTERS + NLOS_CLUSTERS

VARIANTS = {
    "complex":   {"csi": "complex",   "use_rssi": True,  "antenna": 0, "label": "Nguyen CNN-LSTM (complex CSI + RSSI)"},
    "amplitude": {"csi": "amplitude", "use_rssi": True,  "antenna": 0, "label": "Nguyen CNN-LSTM* (amplitude + RSSI)"},
    "ours":      {"csi": "ours",      "use_rssi": False,               "label": "Nguyen CNN-LSTM (our input)"},
}
PARTS = {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}
NGUYEN_REPORTED = 96.53
SURFACE, INK_2 = "#fcfcfb", "#52514e"


def config_of(variant):
    cfg = dict(core.DEFAULT_CONFIG)
    cfg.update({"data_dir": DATA_DIR, "seed": 42, "window": 50, "stride": 10})
    cfg.update({k: v for k, v in VARIANTS[variant].items() if k != "label"})
    return cfg


def unseen_mean(results_dir, variant, protocol):
    """Mean accuracy of the stored 80-split run (Step 7 of the notebook), if available."""
    path = os.path.join(results_dir, variant, f"splits_{protocol}.jsonl")
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        accs = [json.loads(line)["accuracy"] for line in f if line.strip()]
    return 100 * float(np.mean(accs)) if accs else float("nan")


class Monitor(keras.callbacks.Callback):
    """Accuracy on the training windows and on the unseen windows after every epoch (inference mode)."""

    def __init__(self, Xtr, ytr, Xte, yte):
        super().__init__()
        self.d, self.rows = (Xtr, ytr, Xte, yte), []

    def on_epoch_end(self, epoch, logs=None):
        Xtr, ytr, Xte, yte = self.d
        ptr = self.model.predict(Xtr, batch_size=512, verbose=0).ravel()
        pte = self.model.predict(Xte, batch_size=512, verbose=0).ravel()
        self.rows.append({"epoch": epoch + 1,
                          "train_acc": float(((ptr >= 0.5) == ytr).mean()),
                          "unseen_acc": float(((pte >= 0.5) == yte).mean()),
                          "unseen_recall_los": float((pte[yte == 1] >= 0.5).mean()),
                          "unseen_recall_nlos": float((pte[yte == 0] < 0.5).mean()),
                          "unseen_auc": float(roc_auc_score(yte, pte))})


def diagnostic_a(variants, data, split, epochs, W, S):
    out = {}
    for v in variants:
        cfg = config_of(v)
        frames, labels = data[v]
        keras.backend.clear_session()
        keras.utils.set_random_seed(cfg["seed"])
        tf_ = core.Transform(cfg, frames, split["train"])
        Xtr, ytr, _ = core.make_sequences(frames, labels, split["train"], tf_, W, S)
        Xte, yte, _ = core.make_sequences(frames, labels, split["test"], tf_, W, S)
        model = core.build_model(cfg, W, Xtr.shape[2])
        mon = Monitor(Xtr, ytr, Xte, yte)
        t0 = time.time()
        model.fit(Xtr, ytr, epochs=epochs, batch_size=cfg["batch_size"], callbacks=[mon], verbose=0)

        best_train = max(r["train_acc"] for r in mon.rows)
        best_unseen_bal = max((r["unseen_recall_los"] + r["unseen_recall_nlos"]) / 2 for r in mon.rows)
        if best_train < max(0.8, ytr.mean() + 0.1):
            verdict = "does NOT learn the training clusters -> implementation or input problem"
        elif best_unseen_bal < 0.75:
            verdict = ("learns the training clusters but does NOT transfer to unseen clusters "
                       "-> memorisation of the training environments, not a bug")
        else:
            verdict = "learns the training clusters and transfers to unseen clusters"
        out[v] = {"rows": mon.rows, "los_share_train": float(ytr.mean()), "los_share_unseen": float(yte.mean()),
                  "best_train_acc": best_train, "best_unseen_balanced_acc": best_unseen_bal, "verdict": verdict}

        print(f"\n=== A | {VARIANTS[v]['label']}  (split {split['name']}; LOS share: train {ytr.mean():.3f}, "
              f"unseen {yte.mean():.3f}; {time.time() - t0:.0f}s)")
        print(f"  {'epoch':>5s} {'train acc':>10s} {'unseen acc':>11s} {'unseen LOS rec':>15s} "
              f"{'unseen NLOS rec':>16s} {'unseen AUC':>11s}")
        for r in mon.rows:
            if r["epoch"] in (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50) or r["epoch"] == len(mon.rows):
                print(f"  {r['epoch']:5d} {r['train_acc']:10.4f} {r['unseen_acc']:11.4f} {r['unseen_recall_los']:15.4f} "
                      f"{r['unseen_recall_nlos']:16.4f} {r['unseen_auc']:11.4f}")
        print(f"  best training accuracy {best_train:.4f}; best balanced accuracy on unseen clusters {best_unseen_bal:.4f}")
        print(f"  VERDICT: {verdict}")
    return out


def plot_a(diag, split, path):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2})
    fig, axes = plt.subplots(1, len(diag), figsize=(6.2 * len(diag), 4.2), sharey=True, squeeze=False)
    for ax, (v, d) in zip(axes[0], diag.items()):
        ep = [r["epoch"] for r in d["rows"]]
        ax.plot(ep, [r["train_acc"] for r in d["rows"]], color="#2a78d6", label="Training clusters: accuracy")
        ax.plot(ep, [r["unseen_acc"] for r in d["rows"]], color="#eb6834", label="Unseen clusters: accuracy")
        ax.plot(ep, [r["unseen_recall_nlos"] for r in d["rows"]], color="#1baf7a", linestyle="--",
                label="Unseen clusters: NLOS recall")
        ax.axhline(d["los_share_unseen"], color=INK_2, linestyle=":", linewidth=1)
        ax.text(ep[-1], d["los_share_unseen"] + 0.01, "always LOS", ha="right", va="bottom", fontsize=8, color=INK_2)
        ax.set_title(VARIANTS[v]["label"], fontsize=10.5)
        ax.set_xlabel("Epoch (fixed training, no selection)")
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0][0].set_ylabel("Accuracy / recall")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(f"Diagnostic A: learning curves on split {split['name']} (unseen clusters only monitored)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def diagnostic_b(variants, data, max_epochs, W, S):
    out = {}
    for v in variants:
        cfg = config_of(v)
        frames, labels = data[v]
        keras.backend.clear_session()
        keras.utils.set_random_seed(cfg["seed"])
        part = {name: {c: frames[c][int(a * len(frames[c])):int(b * len(frames[c]))] for c in ALL_CLUSTERS}
                for name, (a, b) in PARTS.items()}
        tf_ = core.Transform(cfg, part["train"], ALL_CLUSTERS)
        X, y = {}, {}
        for name in PARTS:
            X[name], y[name], _ = core.make_sequences(part[name], labels, ALL_CLUSTERS, tf_, W, S)
        model = core.build_model(cfg, W, X["train"].shape[2])
        stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
        t0 = time.time()
        h = model.fit(X["train"], y["train"], validation_data=(X["val"], y["val"]), epochs=max_epochs,
                      batch_size=cfg["batch_size"], callbacks=[stop], verbose=0)
        p_val = model.predict(X["val"], batch_size=512, verbose=0).ravel()
        p_te = model.predict(X["test"], batch_size=512, verbose=0).ravel()
        t_val = core.tuned_threshold(y["val"], p_val)[0]
        yt = y["test"]
        s = {"n_windows": {k: int(len(y[k])) for k in PARTS},
             "epochs_run": len(h.history["loss"]),
             "best_epoch": int(np.argmax(h.history["val_accuracy"])) + 1,
             "accuracy": float(((p_te >= 0.5) == yt).mean()),
             "recall_los": float((p_te[yt == 1] >= 0.5).mean()),
             "recall_nlos": float((p_te[yt == 0] < 0.5).mean()),
             "roc_auc": float(roc_auc_score(yt, p_te)),
             "threshold_from_val": float(t_val),
             "accuracy_threshold_from_val": float(((p_te >= t_val) == yt).mean()),
             "los_share_test": float(yt.mean())}
        out[v] = s
        print(f"\n=== B | {VARIANTS[v]['label']}: seen environments (chronological 70/15/15 per cluster; "
              f"{time.time() - t0:.0f}s)")
        print(f"  windows train/val/test {s['n_windows']}, best epoch {s['best_epoch']} of {s['epochs_run']} run")
        print(f"  test accuracy @0.5 {100 * s['accuracy']:.2f}%  (LOS recall {100 * s['recall_los']:.1f}%, "
              f"NLOS recall {100 * s['recall_nlos']:.1f}%), AUC {s['roc_auc']:.4f}")
        print(f"  test accuracy @threshold from validation ({s['threshold_from_val']:.3f}) "
              f"{100 * s['accuracy_threshold_from_val']:.2f}%   (always-LOS: {100 * s['los_share_test']:.1f}%)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["complex", "ours"], choices=list(VARIANTS))
    ap.add_argument("--epochs-a", type=int, default=30)
    ap.add_argument("--max-epochs-b", type=int, default=100)
    ap.add_argument("--test", nargs=2, default=["LOS_cluster2", "NLOS_cluster3"], metavar=("LOS", "NLOS"))
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    W, S = 50, 10
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    print("Python", sys.version.split()[0], "| Keras", keras.__version__, "|", sys.executable)
    print("data:", DATA_DIR, "| output:", os.path.abspath(args.out))

    data = {}
    for v in args.variants:
        data[v] = core.load_cluster_frames(DATA_DIR, ALL_CLUSTERS, config_of(v))
        print(f"loaded {v}: {data[v][0][ALL_CLUSTERS[0]].shape[1]} values per frame")

    result = {"variants": args.variants}
    split = {"name": "test=" + "+".join(c.replace("_cluster", "") for c in args.test), "test": list(args.test),
             "train": [c for c in ALL_CLUSTERS if c not in args.test]}
    if not args.skip_a:
        diag = diagnostic_a(args.variants, data, split, args.epochs_a, W, S)
        fig_path = os.path.join(args.out, "figures", "fig5_diagnostic_learning_curves.png")
        plot_a(diag, split, fig_path)
        result["diagnostic_A"] = {"split": split, "epochs": args.epochs_a, "variants": diag}
        print("\nsaved figure:", fig_path)
    if not args.skip_b:
        seen = diagnostic_b(args.variants, data, args.max_epochs_b, W, S)
        result["diagnostic_B"] = {"parts": PARTS, "max_epochs": args.max_epochs_b, "variants": seen}

        print(f"\n{'variant':42s} {'seen env. (%)':>14s} {'unseen LOPO (%)':>16s} {'unseen L2PO (%)':>16s}")
        print(f"{'Nguyen et al., as reported (their building)':42s} {NGUYEN_REPORTED:14.2f} {'—':>16s} {'—':>16s}")
        for v in args.variants:
            print(f"{VARIANTS[v]['label']:42s} {100 * seen[v]['accuracy']:14.2f} "
                  f"{unseen_mean(args.out, v, 'lopo'):16.2f} {unseen_mean(args.out, v, 'l2po'):16.2f}")

    path = os.path.join(args.out, "diagnostics.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved:", path)


if __name__ == "__main__":
    main()
