"""Accuracy, confidence intervals, paired tests, ranks, per-cluster results and training cost of all models.

Usage: python analysis/overall.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import common as C
from common import PROT, CLUSTERS, R, ROOT, metric, summary, wilcoxon_table, friedman, fmt_p, save, clean_axes

SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f5f8fd", "#c4daf4", "#6fa6e6", "#2a78d6", "#14498d"])
MODELS = {
    "CNN-Transformer": ("CNN_Transformer/results", "ours", "window"),
    "CNN-LSTM": ("CNN_LSTM/results", "ours", "window"),
    "RF": ("RF/results", "classic", "frame"),
    "SVM": ("SVM/results", "classic", "frame"),
    "Nguyen (our input)": ("Nguyen2018_CNNLSTM/results/ours", "nguyen", "window"),
    "Nguyen (complex + RSSI)": ("Nguyen2018_CNNLSTM/results/complex", "nguyen", "window"),
}
FOUR = ["CNN-Transformer", "CNN-LSTM", "RF", "SVM"]
SIX = ["CNN-Transformer", "CNN-LSTM", "Nguyen (our input)", "RF", "SVM", "Nguyen (complex + RSSI)"]
FAM_COLOR = C.FAM_COLOR
INK, INK2, MUTED = C.INK, C.INK2, C.MUTED
short = lambda c: c.replace("_cluster", " ")
S = {}

ref = C.ref_names()
for m, (d, fam, unit) in MODELS.items():
    C.put(m, {p: C.load_jsonl(os.path.join(ROOT, d, f"splits_{p}.jsonl")) for p in PROT}, fam, unit, ref)
with open(os.path.join(ROOT, "output_raw_full/clusters/index.json")) as f:
    FRAMES = {c: v["frames"] for c, v in json.load(f)["clusters"].items()}
WINDOWS = {c: (n - 50) // 10 + 1 for c, n in FRAMES.items()}
with open(os.path.join(ROOT, "Nguyen2018_CNNLSTM/results/diagnostics.json")) as f:
    DIAG = json.load(f)


def pooled_confusion(m, p):
    """Exact pooled confusion matrix over all splits of a protocol, from per-cluster accuracies."""
    n_of = WINDOWS if MODELS[m][2] == "window" else FRAMES
    cm = np.zeros((2, 2), dtype=int)
    for r in R[m][p]:
        for c, acc in r["per_cluster_accuracy"].items():
            n = n_of[c]
            ok = int(round(acc * n))
            if c.startswith("LOS"):
                cm[0] += [ok, n - ok]
            else:
                cm[1] += [n - ok, ok]
        assert sum(n_of[c] for c in r["test"]) == r.get("n_test", r.get("n_test_windows"))
    return cm


S["overall"] = {}
for m in SIX:
    S["overall"][m] = {}
    for p, pn in list(PROT.items()) + [(None, "All")]:
        prots = (p,) if p else ("lopo", "l2po")
        d = summary(metric(m, "accuracy", prots))
        if p:
            d.update({"macro_f1": metric(m, "macro_f1", prots).mean() * 100, "macro_f1_sd": metric(m, "macro_f1", prots).std() * 100,
                      "auc": metric(m, "roc_auc", prots).mean(), "auc_sd": metric(m, "roc_auc", prots).std(),
                      "rec_los": metric(m, "recall_los", prots).mean() * 100, "rec_nlos": metric(m, "recall_nlos", prots).mean() * 100,
                      "n_above_99": int((metric(m, "accuracy", prots) > 0.99).sum())})
        S["overall"][m][pn] = d
S["friedman4"], S["friedman6"] = friedman(FOUR), friedman(SIX)
S["wilcoxon4"] = wilcoxon_table(FOUR)
S["wilcoxon_vs_nguyen"] = [r for r in wilcoxon_table(SIX)
                           if r["a"] in FOUR[:2] and r["b"].startswith("Nguyen")]
S["confusion"] = {m: {PROT[p]: pooled_confusion(m, p).tolist() for p in PROT} for m in SIX}

S["per_cluster"] = {}
for m in SIX:
    S["per_cluster"][m] = {}
    for c in CLUSTERS:
        e = {}
        for p in PROT:
            a = np.array([r["per_cluster_accuracy"][c] for r in R[m][p] if c in r["per_cluster_accuracy"]])
            e[PROT[p]] = {"mean": a.mean() * 100, "worst": a.min() * 100, "n": len(a)}
        S["per_cluster"][m][c] = e

grp = np.array([{"LOS_cluster1", "LOS_cluster2"} <= set(r["test"]) for r in R["SVM"]["l2po"]])
S["failure"] = {"n_group": int(grp.sum())}
for m in FOUR:
    a = metric(m, "accuracy", ("l2po",))
    S["failure"][m] = {"group": a[grp].mean() * 100, "rest": a[~grp].mean() * 100,
                       "auc_group_min": metric(m, "roc_auc", ("l2po",))[grp].min(),
                       "below80_in_group": int(((a < 0.8) & grp).sum()), "below80": int((a < 0.8).sum())}
S["inner"] = {}
for m in FOUR[:2]:
    e = {}
    for p in PROT:
        inner, test = metric(m, "inner_cv_accuracy", (p,)), metric(m, "accuracy", (p,))
        e[PROT[p]] = {"inner": inner.mean() * 100, "test": test.mean() * 100,
                      "pearson_r": float(stats.pearsonr(inner, test)[0]) if p == "l2po" else None}
    pg = metric(m, "inner_cv_accuracy", ("l2po",))[grp]
    e["group_inner"], e["group_test"] = pg.mean() * 100, metric(m, "accuracy", ("l2po",))[grp].mean() * 100
    S["inner"][m] = e

S["epochs"] = {m: {PROT[p]: {"median": float(np.median(metric(m, "selected_epochs", (p,)))),
                             "min": int(metric(m, "selected_epochs", (p,)).min()),
                             "max": int(metric(m, "selected_epochs", (p,)).max())} for p in PROT}
               for m in FOUR[:2] + ["Nguyen (our input)", "Nguyen (complex + RSSI)"]}
S["cost"] = {}
for m in FOUR:
    e = {}
    for p in PROT:
        fit = metric(m, "fit_seconds", (p,))
        sel = metric(m, "select_seconds", (p,)) if "select_seconds" in R[m][p][0] else np.zeros_like(fit)
        e[PROT[p]] = {"select_s": sel.mean(), "fit_s": fit.mean(), "total_s": (sel + fit).mean()}
    S["cost"][m] = e
S["params"] = {"CNN-LSTM": 218625, "CNN-Transformer": 240513, "Nguyen (our input)": R["Nguyen (our input)"]["lopo"][0]["n_parameters"],
               "Nguyen (complex + RSSI)": R["Nguyen (complex + RSSI)"]["lopo"][0]["n_parameters"]}

for m in SIX[2:3] + SIX[5:]:
    for p in PROT:
        S["overall"][m][PROT[p]]["acc_tuned"] = metric(m, "accuracy_tuned_threshold", (p,)).mean() * 100
        S["overall"][m][PROT[p]]["auc_range"] = [metric(m, "roc_auc", (p,)).min(), metric(m, "roc_auc", (p,)).max()]
S["nguyen_complex_all_los"] = int(sum(r["recall_los"] == 1 and r["recall_nlos"] == 0
                                      for p in PROT for r in R["Nguyen (complex + RSSI)"][p]))
low = [r for r in R["Nguyen (our input)"]["l2po"] if r["accuracy"] < 0.5]
S["nguyen_ours_collapse"] = [{"name": r["name"], "acc": r["accuracy"] * 100, "epochs": r["selected_epochs"],
                              "rec_los": r["recall_los"]} for r in low]
S["diagnostic"] = DIAG
C.write_json("overall_summary.json", S)

md = ["| Protocol | Model | Accuracy (%) | 95% CI | Median | Worst | Macro-F1 (%) | ROC-AUC | LOS rec. | NLOS rec. |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for pn in ("LOPO", "L2PO", "All"):
    for m in FOUR:
        o = S["overall"][m][pn]
        ci = f"[{o['ci'][0]:.2f}, {o['ci'][1]:.2f}]"
        if pn == "All":
            md.append(f"| {pn} | {m} | {o['mean']:.2f} ± {o['sd']:.2f} | {ci} | {o['median']:.2f} | {o['worst']:.2f} | | | | |")
        else:
            md.append(f"| {pn} | {m} | {o['mean']:.2f} ± {o['sd']:.2f} | {ci} | {o['median']:.2f} | {o['worst']:.2f} | "
                      f"{o['macro_f1']:.2f} ± {o['macro_f1_sd']:.2f} | {o['auc']:.3f} ± {o['auc_sd']:.3f} | {o['rec_los']:.2f} | {o['rec_nlos']:.2f} |")
C.write_md("overall.md", md)

md = ["| A vs. B | Δ (pts) | W / L / T | r_rb | p (Holm) |", "|---|---|---|---|---|"]
for r in S["wilcoxon4"]:
    md.append(f"| {r['a']} vs. {r['b']} | {r['delta']:+.2f} | {r['wins']} / {r['losses']} / {r['ties']} | {r['r_rb']:.2f} | {fmt_p(r['p_holm'])} |")
C.write_md("pairwise.md", md)

order = ["SVM", "RF", "CNN-LSTM", "CNN-Transformer"]
md = ["| Cluster | " + " | ".join(f"{m} LOPO | {m} L2PO (worst)" for m in order) + " |", "|---" * 9 + "|"]
for c in CLUSTERS:
    md.append(f"| {short(c)} | " + " | ".join(f"{S['per_cluster'][m][c]['LOPO']['mean']:.1f} | {S['per_cluster'][m][c]['L2PO']['mean']:.1f} ({S['per_cluster'][m][c]['L2PO']['worst']:.1f})" for m in order) + " |")
C.write_md("per_cluster.md", md)

f6 = S["friedman6"]
names = {"Nguyen (our input)": "Nguyen (ii)", "Nguyen (complex + RSSI)": "Nguyen (i)", "RF": "RF"}
md = ["| Model | LOPO (%) | L2PO (%) | Worst split (%) | Mean rank (6 models) |", "|---|---|---|---|---|"]
for m in sorted(SIX, key=lambda m: f6["ranks"][m]):
    o = S["overall"][m]
    md.append(f"| {names.get(m, m)} | {o['LOPO']['mean']:.2f} ± {o['LOPO']['sd']:.2f} | "
              f"{o['L2PO']['mean']:.2f} ± {o['L2PO']['sd']:.2f} | {o['All']['worst']:.2f} | {f6['ranks'][m]:.2f} |")
C.write_md("nguyen_comparison.md", md)


def cost_cell(e):
    return f"{e['select_s']:.0f} + {e['fit_s']:.0f}" if e["select_s"] else f"{e['fit_s']:.0f}"


md = ["| Model | Parameters | LOPO: selection + final (s) | L2PO: selection + final (s) |", "|---|---|---|---|"]
for m in FOUR:
    c = S["cost"][m]
    par = f"{S['params'][m]:,}" if m in S["params"] else "—"
    md.append(f"| {m} | {par} | {cost_cell(c['LOPO'])} | {cost_cell(c['L2PO'])} |")
C.write_md("cost.md", md)


def strip(ax, models, prot, ylim, label_fn=lambda m: m):
    jit = np.random.default_rng(1)
    for i, m in enumerate(models):
        a = metric(m, "accuracy", (prot,)) * 100
        col = FAM_COLOR[MODELS[m][1]]
        ax.scatter(i + jit.uniform(-0.22, 0.22, len(a)), a, s=9, color=col, alpha=0.55, lw=0, zorder=2)
        o = S["overall"][m][PROT[prot]]
        ax.plot([i - 0.3, i + 0.3], [o["mean"], o["mean"]], color=INK, lw=1.4, zorder=3)
        ax.plot([i, i], o["ci"], color=INK, lw=0.9, zorder=3)
        ax.text(i, ylim[0] + 1.2, f"{o['mean']:.1f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label_fn(m) for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(*ylim)


fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, p in zip(axes, PROT):
    clean_axes(ax)
    strip(ax, FOUR, p, (60, 101), lambda m: m.replace("CNN-Transformer", "CNN-Transf."))
    ax.set_title(f"{PROT[p]} ({len(R['SVM'][p])} splits)", loc="left")
axes[0].set_ylabel("Accuracy on held-out clusters (%)")
fig.text(0.99, -0.02, "dot = one split; bar = mean; vertical line = 95% bootstrap CI; number = mean", ha="right",
         fontsize=5.8, color=MUTED)
save(fig, "accuracy_per_split")


def cm_panel(ax, cm, title, unit):
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = norm[i, j]
            ax.text(j, i - 0.12, f"{v * 100:.1f}%", ha="center", va="center", fontsize=6.6, fontweight="bold",
                    color="white" if v > 0.55 else INK)
            ax.text(j, i + 0.2, f"{cm[i, j]:,}", ha="center", va="center", fontsize=5.4,
                    color="white" if v > 0.55 else INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LOS", "NLOS"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LOS", "NLOS"])
    ax.tick_params(length=0, colors=INK2, labelsize=6.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=6.6, color=INK, pad=3)
    ax.set_xlabel(f"predicted ({unit}s)", fontsize=5.6, color=MUTED, labelpad=2)


fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.95), gridspec_kw=dict(hspace=0.55, wspace=0.3))
for r, (rname, P) in enumerate((("LOPO\n(20 splits, pooled)", "LOPO"), ("L2PO\n(60 splits, pooled)", "L2PO"))):
    for k, m in enumerate(FOUR):
        cm_panel(axes[r, k], np.array(S["confusion"][m][P]), m, MODELS[m][2])
        if k == 0:
            axes[r, k].set_ylabel("true", fontsize=6.0, color=MUTED)
    axes[r, 0].text(-0.62, 0.5, rname, transform=axes[r, 0].transAxes, fontsize=7.0, fontweight="bold", color=INK,
                    rotation=90, ha="center", va="center", linespacing=1.2)
save(fig, "confusion_matrices")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.16, 3.3), gridspec_kw=dict(hspace=0.35, height_ratios=[2.6, 3.1]))
C.MODELS.update(MODELS)
C.cd_diagram(a1, S["friedman4"], f"(a) Four models, 80 splits: Friedman χ² = {S['friedman4']['chi2']:.1f}, p = {S['friedman4']['p']:.1e}")
C.cd_diagram(a2, S["friedman6"], f"(b) With the two Nguyen et al. variants: χ² = {S['friedman6']['chi2']:.1f}, p = {S['friedman6']['p']:.1e}")
for ax in (a1, a2):
    ax.text(ax.get_xlim()[1], ax.get_ylim()[0] + 0.1, "thick bar = no significant difference (Nemenyi, α = 0.05)",
            ha="right", va="bottom", fontsize=5.6, color=MUTED)
save(fig, "critical_difference")

cols = [(m, p) for p in ("LOPO", "L2PO") for m in FOUR]
mat = np.array([[S["per_cluster"][m][c][p]["mean"] for m, p in cols] for c in CLUSTERS])
fig, ax = plt.subplots(figsize=(7.16, 2.75))
ax.imshow(mat, cmap=SEQ.reversed(), vmin=50, vmax=100, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        w = S["per_cluster"][cols[j][0]][CLUSTERS[i]][cols[j][1]]["worst"]
        txt = f"{v:.1f}" if cols[j][1] == "LOPO" else f"{v:.1f}\n({w:.0f})"
        ax.text(j, i, txt, ha="center", va="center", fontsize=5.7 if "\n" in txt else 6.2,
                color="white" if v < 80 else INK, linespacing=1.0)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([m.replace("CNN-Transformer", "CNN-Tr.") for m, _ in cols])
ax.set_yticks(range(len(CLUSTERS)))
ax.set_yticklabels([short(c) for c in CLUSTERS])
ax.tick_params(length=0, colors=INK2)
ax.axvline(3.5, color="white", lw=3)
ax.axhline(3.5, color="white", lw=3)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(1.5, -0.95, "LOPO (mean)", ha="center", fontsize=7, fontweight="bold", color=INK)
ax.text(5.5, -0.95, "L2PO: mean (worst split)", ha="center", fontsize=7, fontweight="bold", color=INK)
ax.set_ylim(len(CLUSTERS) - 0.5, -1.4)
fig.text(0.99, -0.01, "accuracy (%) on the cluster when it is held out; darker = lower", ha="right", fontsize=5.8, color=MUTED)
save(fig, "per_cluster")

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True, gridspec_kw=dict(wspace=0.06))
lab = {"Nguyen (our input)": "Nguyen\n(our input)", "Nguyen (complex + RSSI)": "Nguyen\n(complex+RSSI)",
       "CNN-Transformer": "CNN-\nTransformer", "CNN-LSTM": "CNN-\nLSTM"}
for ax, p in zip(axes, PROT):
    clean_axes(ax)
    shares = []
    for r in R["SVM"][p]:
        n_los = sum(WINDOWS[c] for c in r["test"] if c.startswith("LOS"))
        shares.append(n_los / sum(WINDOWS[c] for c in r["test"]) * 100)
    ax.axhspan(min(shares), max(shares), color="#ecebe6", zorder=0)
    ax.text(-0.5, max(shares) + 0.8, "always-LOS rate", ha="left", fontsize=5.6, color=MUTED)
    strip(ax, SIX, p, (25, 102), lambda m: lab.get(m, m))
    ax.set_title(f"{PROT[p]} ({len(R['SVM'][p])} splits)", loc="left")
axes[0].set_ylabel("Accuracy on held-out clusters (%)")
fig.text(0.99, -0.06, "blue = CNN models, grey = frame classifiers, violet = Nguyen et al. re-implemented; same splits, epoch selection and threshold",
         ha="right", fontsize=5.8, color=MUTED)
save(fig, "nguyen_comparison")

fig, (a3, a4) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.28))
clean_axes(a3)
jit = np.random.default_rng(2)
xt = []
for g, p in enumerate(PROT):
    for i, m in enumerate(FOUR):
        x = g * 5 + i
        xt.append((x, m.replace("CNN-Transformer", "CNN-Tr.")))
        a = metric(m, "roc_auc", (p,))
        a3.scatter(x + jit.uniform(-0.25, 0.25, len(a)), a, s=7, color=FAM_COLOR[MODELS[m][1]], alpha=0.55, lw=0)
        a3.plot([x - 0.32, x + 0.32], [a.mean()] * 2, color=INK, lw=1.3)
        a3.text(x, 0.712, f"{a.mean():.3f}", ha="center", fontsize=5.4, color=INK)
    a3.text(g * 5 + 1.5, 1.012, PROT[p], ha="center", fontsize=6.5, fontweight="bold", color=INK)
a3.set_xticks([x for x, _ in xt])
a3.set_xticklabels([l for _, l in xt], rotation=45, ha="right")
a3.set_ylim(0.7, 1.02)
a3.set_ylabel("ROC-AUC on held-out clusters")
a3.set_title("(a) ROC-AUC of every split (bar = mean)", loc="left")
clean_axes(a4, "both")
styles = {"CNN-Transformer": (C.C_OURS, "o"), "CNN-LSTM": (C.C_OURS, "s"), "RF": (C.C_CLASSIC, "^"), "SVM": (C.C_CLASSIC, "D")}
for m in FOUR:
    col, mk = styles[m]
    a4.scatter(metric(m, "roc_auc"), metric(m, "accuracy") * 100, s=12, marker=mk, color=col,
               alpha=0.65 if m != "RF" else 0.8, lw=0, label=m)
a4.set_xlim(0.7, 1.005)
a4.set_ylim(60, 101)
a4.set_xlabel("ROC-AUC (ranking quality)")
a4.set_ylabel("Accuracy at threshold 0.5 (%)")
a4.set_title("(b) Accuracy vs. ROC-AUC, all 80 splits", loc="left")
a4.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=1)
save(fig, "roc_auc")

fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.9), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, p in zip(axes, PROT):
    clean_axes(ax, "x")
    for i, m in enumerate(FOUR[::-1]):
        o = S["overall"][m][PROT[p]]
        ax.plot([o["rec_los"], o["rec_nlos"]], [i, i], color="#d9d8d2", lw=2.2, zorder=1, solid_capstyle="round")
        ax.scatter(o["rec_los"], i, s=26, marker="o", color="#2a78d6", zorder=3, ec="white", lw=0.6)
        ax.scatter(o["rec_nlos"], i, s=26, marker="s", color="#eb6834", zorder=3, ec="white", lw=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(FOUR[::-1])
    ax.set_xlim(83, 100.8)
    ax.set_xlabel("Recall (%), mean over splits")
    ax.set_title(PROT[p], loc="left")
axes[0].scatter([], [], s=26, marker="o", color="#2a78d6", label="LOS recall")
axes[0].scatter([], [], s=26, marker="s", color="#eb6834", label="NLOS recall")
axes[0].legend(frameon=False, fontsize=6.0, loc="lower left", ncol=1)
save(fig, "class_recall")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.3))
clean_axes(a1)
jit = np.random.default_rng(3)
for i, m in enumerate(FOUR):
    a = metric(m, "accuracy", ("l2po",)) * 100
    for off, mask, col in ((-0.18, ~grp, FAM_COLOR[MODELS[m][1]]), (0.18, grp, "#b3261e")):
        a1.scatter(i + off + jit.uniform(-0.09, 0.09, mask.sum()), a[mask], s=8, color=col, alpha=0.6, lw=0)
        a1.plot([i + off - 0.12, i + off + 0.12], [a[mask].mean()] * 2, color=INK, lw=1.3)
a1.set_xticks(range(4))
a1.set_xticklabels([m.replace("CNN-Transformer", "CNN-Transf.") for m in FOUR])
a1.set_ylim(60, 101)
a1.set_ylabel("L2PO accuracy (%)")
a1.set_title("(a) L2PO splits with vs. without LOS 1 + LOS 2 held out", loc="left")
a1.scatter([], [], s=12, color=C.C_OURS, label="other 50 splits")
a1.scatter([], [], s=12, color="#b3261e", label="LOS 1 + LOS 2 held out (10)")
a1.legend(frameon=False, fontsize=5.8, loc="lower left")
clean_axes(a2, "both")
for m, mk in (("CNN-Transformer", "o"), ("CNN-LSTM", "s")):
    inner, test = metric(m, "inner_cv_accuracy", ("l2po",)) * 100, metric(m, "accuracy", ("l2po",)) * 100
    a2.scatter(inner[~grp], test[~grp], s=12, marker=mk, color=C.C_OURS, alpha=0.6, lw=0, label=f"{m}")
    a2.scatter(inner[grp], test[grp], s=12, marker=mk, color="#b3261e", alpha=0.75, lw=0)
a2.plot([60, 101], [60, 101], color=MUTED, lw=0.6, ls="--")
a2.set_xlim(60, 101)
a2.set_ylim(60, 101)
a2.set_xlabel("Inner-CV accuracy (training clusters only, %)")
a2.set_ylabel("Test accuracy (%)")
a2.set_title("(b) Training-only estimate vs. test (L2PO)", loc="left")
a2.scatter([], [], s=12, color="#b3261e", label="red: LOS 1 + LOS 2 held out")
a2.legend(frameon=False, fontsize=5.8, loc="lower left", markerscale=1.2)
save(fig, "failure_analysis")

dA = DIAG["diagnostic_A"]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
clean_axes(ax, "both")
for v, col, name in (("complex", C.C_NGUYEN, "complex CSI + RSSI"), ("ours", C.C_OURS, "our input")):
    rows = dA["variants"][v]["rows"]
    ax.plot([r["epoch"] for r in rows], [r["unseen_acc"] * 100 for r in rows], color=col, lw=1.3, marker="o", ms=2.5, label=name)
ax.axhline(dA["variants"]["complex"]["los_share_unseen"] * 100, color=MUTED, lw=0.6, ls="--")
ax.text(30, dA["variants"]["complex"]["los_share_unseen"] * 100 + 1.5, "always-LOS rate", ha="right", fontsize=5.5, color=MUTED)
ax.set_xlim(0.5, 30.5)
ax.set_ylim(0, 103)
ax.set_xlabel("Training epoch (fixed schedule, no selection)")
ax.set_ylabel("Accuracy on unseen clusters (%)")
ax.set_title(f"Nguyen et al. architecture, split {dA['split']['name'].replace('test=', 'test ')}", loc="left", fontsize=6.6)
ax.legend(frameon=False, fontsize=5.8, loc="center right")
save(fig, "nguyen_diagnostic")

for m in FOUR:
    o = S["overall"][m]["All"]
    print(f"{m:16s} all 80: {o['mean']:.2f} ± {o['sd']:.2f}  CI [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}]")
