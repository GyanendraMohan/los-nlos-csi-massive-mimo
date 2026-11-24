"""SNR equalisation (conditions A and B) and frame against window decisions.

Usage: python analysis/robustness.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import common as C
from common import PROT, ROOT, holm, save, clean_axes

SNR = os.path.join(ROOT, "robustness", "01_snr_equalisation", "results")
FVW = os.path.join(ROOT, "robustness", "02_frame_vs_window", "results")
INK, MUTED = C.INK, C.MUTED
COND = {"orig": ("Original data", "#8a8984"),
        "A": ("(A) Equal SNR, train + test", "#2a78d6"),
        "B": ("(B) Equal SNR at test only", "#eb6834")}
FOUR = {"CNN-Transformer": ("deep", "cnn_transformer", "CNN_Transformer"),
        "CNN-LSTM": ("deep", "cnn_lstm", "CNN_LSTM"),
        "RF": ("frame", "rf", "RF"),
        "SVM": ("frame", "svm", "SVM")}
OUT = {}


def load(path):
    with open(path) as f:
        return {r["name"]: r for r in map(json.loads, f)}


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0


D = {}
for m, (kind, key, stored_dir) in FOUR.items():
    D[m] = {}
    for p in PROT:
        A = load(os.path.join(SNR, f"{kind}_{key}_equalised_{p}.jsonl"))
        B = load(os.path.join(SNR, f"{kind}_{key}_test_shift_{p}.jsonl"))
        st = load(os.path.join(ROOT, stored_dir, "results", f"splits_{p}.jsonl"))
        names = [n for n in st if n in A and n in B]
        assert len(names) == len(st), (m, p, len(names))
        if kind == "frame":
            get = {"orig": lambda n: B[n]["original"]["frame"], "A": lambda n: A[n]["test"]["frame"],
                   "B": lambda n: B[n]["shifted"]["frame"]}
        else:
            get = {"orig": lambda n: B[n], "A": lambda n: A[n], "B": lambda n: B[n]["shifted_test"]}
        D[m][p] = {c: {k: np.array([get[c](n)[k] for n in names]) for k in ("accuracy", "recall_los", "recall_nlos")}
                   for c in get}
        D[m][p]["stored"] = np.array([st[n]["accuracy"] for n in names])

snr_info = json.load(open(os.path.join(SNR, "deep_models_run", "summary.json")))
OUT["snr_db"] = {c: round(v.get("snr_before_db", v.get("snr_db")), 1) for c, v in snr_info["snr"].items()}
OUT["target_snr_db"] = round(snr_info["target_snr_db"], 2)
OUT["noise_factor_db"] = {c: round(10 * np.log10(v["noise_factor_g"]), 1)
                          for c, v in snr_info["snr"].items() if v.get("noise_added")}

tests, pvals = [], []
for m in FOUR:
    for c in ("A", "B"):
        a = np.concatenate([D[m][p][c]["accuracy"] for p in PROT])
        o = np.concatenate([D[m][p]["orig"]["accuracy"] for p in PROT])
        d = a - o
        tests.append({"model": m, "cond": c, "delta": 100 * d.mean(), "median": 100 * np.median(d),
                      "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum()),
                      "p": wilcoxon(a, o)})
        pvals.append(tests[-1]["p"])
for t, ph in zip(tests, holm(pvals)):
    t["p_holm"] = float(ph)
OUT["snr_tests_80"] = tests

OUT["snr"] = {}
for m in FOUR:
    OUT["snr"][m] = {}
    for p, P in PROT.items():
        r = D[m][p]
        OUT["snr"][m][P] = {
            "stored_acc": 100 * r["stored"].mean(),
            "max_abs_rerun_minus_stored": 100 * np.abs(r["orig"]["accuracy"] - r["stored"]).max(),
            **{f"{c}_{k}": 100 * r[c][k].mean() for c in ("orig", "A", "B") for k in ("accuracy", "recall_los", "recall_nlos")},
            **{f"{c}_acc_sd": 100 * r[c]["accuracy"].std() for c in ("orig", "A", "B")},
            **{f"{c}_acc_min": 100 * r[c]["accuracy"].min() for c in ("orig", "A", "B")},
            **{f"{c}_delta": 100 * (r[c]["accuracy"] - r["orig"]["accuracy"]).mean() for c in ("A", "B")},
            **{f"{c}_delta_median": 100 * np.median(r[c]["accuracy"] - r["orig"]["accuracy"]) for c in ("A", "B")},
        }

rows_md = []
for m in FOUR:
    for i, (p, P) in enumerate(PROT.items()):
        r = D[m][p]
        cells = [f"{100 * r[c]['accuracy'].mean():.2f} ± {100 * r[c]['accuracy'].std():.2f}" for c in ("orig", "A", "B")] + \
                [f"{100 * r[c]['recall_los'].mean():.1f}" for c in ("orig", "A", "B")]
        rows_md.append(f"| {m if i == 0 else ''} | {P} | " + " | ".join(cells) + " |")
C.write_md("snr.md", ["| Model | Protocol | Acc. original | Acc. (A) equal SNR | Acc. (B) SNR shift | LOS rec. original | LOS rec. (A) | LOS rec. (B) |",
                      "|---|---|---|---|---|---|---|---|"] + rows_md)

md = ["| Model | Condition | Δ accuracy (pts) | Median Δ | Better / worse / tied | p (Holm) |", "|---|---|---|---|---|---|"]
for t in tests:
    md.append(f"| {t['model']} | {t['cond']} | {t['delta']:+.2f} | {t['median']:+.2f} | {t['better']} / {t['worse']} / {t['tied']} | {C.fmt_p(t['p_holm'])} |")
C.write_md("snr_tests.md", md)

fvw = json.load(open(os.path.join(FVW, "summary.json")))
ROWS = ["SVM, per frame", "SVM, window vote", "SVM, window mean score", "RF, per frame", "RF, window vote",
        "RF, window mean score", "CNN-LSTM (window)", "CNN-Transformer (window)"]
OUT["window"] = {r: {PROT[p]: {"mean": 100 * np.mean(list(fvw["table"][r][p].values())),
                               "sd": 100 * np.std(list(fvw["table"][r][p].values()))} for p in PROT} for r in ROWS}
OUT["window_tests"] = fvw["tests"]
rows_md = []
for r in ROWS:
    c = [f"{OUT['window'][r][P]['mean']:.2f} ± {OUT['window'][r][P]['sd']:.2f}" for P in PROT.values()]
    rows_md.append(f"| {r.replace(' (window)', '')} | " + " | ".join(c) + " |")
C.write_md("frame_vs_window.md", ["| Decision | LOPO | L2PO |", "|---|---|---|"] + rows_md)

with open(os.path.join(C.OUT, "robustness_summary.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(wspace=0.16))
jit = np.random.default_rng(1)
offs = {"orig": -0.3, "A": 0.0, "B": 0.3}
for ax, key, title, ylab in ((axes[0], "accuracy", "(a) Accuracy", "Accuracy on held-out clusters (%)"),
                             (axes[1], "recall_los", "(b) LOS recall", "LOS recall on held-out clusters (%)")):
    clean_axes(ax)
    for i, m in enumerate(FOUR):
        for c, (lab, col) in COND.items():
            v = 100 * np.concatenate([D[m][p][c][key] for p in PROT])
            x = i + offs[c]
            ax.scatter(x + jit.uniform(-0.1, 0.1, len(v)), v, s=5, color=col, alpha=0.45, lw=0, zorder=2)
            ax.plot([x - 0.11, x + 0.11], [v.mean()] * 2, color=INK, lw=1.3, zorder=3)
            ax.text(x, 101.5, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=5.0, color=INK)
    ax.set_xticks(range(len(FOUR)))
    ax.set_xticklabels([m.replace("CNN-Transformer", "CNN-Transf.") for m in FOUR])
    ax.set_xlim(-0.5, len(FOUR) - 0.5)
    ax.set_ylim(0, 106)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylabel(ylab)
handles = [plt.Line2D([], [], marker="o", ls="", color=col, markersize=4, label=lab) for lab, col in COND.values()]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=6.3)
fig.text(0.99, -0.085, "dot = one split (80 per model); bar and number = mean", ha="right", fontsize=5.8, color=MUTED)
save(fig, "snr_robustness")

STRIPS = [("SVM, per frame", "SVM\nframe", C.C_CLASSIC), ("SVM, window vote", "SVM\nvote", C.C_CLASSIC),
          ("RF, per frame", "RF\nframe", C.C_CLASSIC), ("RF, window vote", "RF\nvote", C.C_CLASSIC),
          ("CNN-LSTM (window)", "CNN-\nLSTM", C.C_OURS), ("CNN-Transformer (window)", "CNN-\nTransf.", C.C_OURS)]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, p in zip(axes, PROT):
    clean_axes(ax)
    for i, (key, lab, col) in enumerate(STRIPS):
        v = 100 * np.array(list(fvw["table"][key][p].values()))
        ax.scatter(i + jit.uniform(-0.2, 0.2, len(v)), v, s=8, color=col, alpha=0.55, lw=0, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=1.4, zorder=3)
        ax.text(i, 61.2, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.set_xticks(range(len(STRIPS)))
    ax.set_xticklabels([s[1] for s in STRIPS])
    ax.set_xlim(-0.6, len(STRIPS) - 0.4)
    ax.set_ylim(60, 101)
    ax.set_title(f"{PROT[p]} ({len(fvw['table']['SVM, per frame'][p])} splits)", loc="left")
axes[0].set_ylabel("Accuracy on held-out clusters (%)")
fig.text(0.99, -0.06, "dot = one split; bar and number = mean; frame = one decision per frame; vote = majority over the 50-frame windows of the sequence models",
         ha="right", fontsize=5.8, color=MUTED)
save(fig, "frame_vs_window")

for t in tests:
    print(f"{t['model']:16s} {t['cond']}  delta {t['delta']:+.2f}  W/L/T {t['better']}/{t['worse']}/{t['tied']}  p_holm {t['p_holm']:.2g}")
