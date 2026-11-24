"""Figures comparing our models with the Nguyen et al. re-implementation on the 80 splits.

Usage: python make_comparison_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figures")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "CNN-Transformer": (os.path.join(HERE, "../CNN_Transformer/results"), "CNN-Transformer\n(ours)", "#2a78d6"),
    "CNN-LSTM":        (os.path.join(HERE, "../CNN_LSTM/results"), "CNN-LSTM\n(ours)", "#eb6834"),
    "RF":              (os.path.join(HERE, "../RF/results"), "Random\nForest", "#1baf7a"),
    "SVM":             (os.path.join(HERE, "../SVM/results"), "SVM", "#eda100"),
    "complex":         (os.path.join(HERE, "results/complex"), "Nguyen et al.\ncomplex CSI\n+ RSSI", "#e87ba4"),
    "ours":            (os.path.join(HERE, "results/ours"), "Nguyen et al.\narchitecture,\nour input", "#008300"),
}
PROTOCOLS = {"lopo": "LOPO: 1 LOS + 1 NLOS cluster unseen (20 splits)",
             "l2po": "L2PO: 2 LOS + 2 NLOS clusters unseen (60 splits)"}
NGUYEN_REPORTED = 96.53
CLUSTERS = ["LOS_cluster1", "LOS_cluster2", "LOS_cluster3", "LOS_cluster4",
            "NLOS_cluster1", "NLOS_cluster2", "NLOS_cluster3", "NLOS_cluster4", "NLOS_cluster5"]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})


def load(model, protocol):
    with open(os.path.join(MODELS[model][0], f"splits_{protocol}.jsonl")) as f:
        return {r["name"]: r for r in (json.loads(line) for line in f if line.strip())}


R = {m: {p: load(m, p) for p in PROTOCOLS} for m in MODELS}
NAMES = {p: [n for n in R["CNN-Transformer"][p] if all(n in R[m][p] for m in MODELS)] for p in PROTOCOLS}
acc = {m: {p: 100 * np.array([R[m][p][n]["accuracy"] for n in NAMES[p]]) for p in PROTOCOLS} for m in MODELS}

cx = [r for p in PROTOCOLS for r in R["complex"][p].values()]
assert all(r["recall_los"] == 1.0 and r["recall_nlos"] == 0.0 for r in cx), "complex variant is not all-LOS"
los_share = {p: acc["complex"][p] for p in PROTOCOLS}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
rng = np.random.default_rng(0)
for ax, (p, title) in zip(axes, PROTOCOLS.items()):
    ax.axhspan(los_share[p].min(), los_share[p].max(), color=GRID, alpha=0.7, zorder=0, linewidth=0)
    ax.text(-0.45, los_share[p].min() - 0.6, "band = accuracy of always predicting LOS", ha="left", va="top",
            color=INK_2, fontsize=8.5)
    for i, m in enumerate(MODELS):
        a = acc[m][p]
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(a)), a, s=26, color=MODELS[m][2], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [a.mean()] * 2, color=INK, linewidth=2.2, zorder=4)
        ax.text(i, 101.2, f"{a.mean():.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(MODELS)), [MODELS[m][1] for m in MODELS], fontsize=8.5)
    ax.set_title(title, fontsize=10.5, pad=18)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(28, 104)
axes[0].set_ylabel("Test accuracy per split (%)\nblack bar and number = mean")
fig.suptitle("Same data, same 80 splits, same protocol: our models vs. the method of Nguyen et al. (2018)",
             fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_accuracy_per_split.png"), dpi=200)
plt.close(fig)

keys = list(MODELS)
A = np.vstack([np.column_stack([acc[m][p] for m in keys]) for p in PROTOCOLS])
N, k = A.shape
ranks = np.array([rankdata(-row) for row in A]).mean(axis=0)
cd = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}[k] * np.sqrt(k * (k + 1) / (6 * N))
order = np.argsort(ranks)
fig, ax = plt.subplots(figsize=(8.2, 3.9))
best = ranks[order[0]]
ax.axvspan(best, best + cd, color=MODELS[keys[order[0]]][2], alpha=0.10, linewidth=0)
ax.text(best + cd / 2, 0.5, f"shaded: within the critical\ndifference ({cd:.2f}) of the best\n(Nemenyi, α = 0.05)",
        ha="center", va="center", fontsize=8, color=INK_2,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
for y, j in enumerate(order[::-1]):
    m = keys[j]
    ax.plot([1, ranks[j]], [y, y], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(ranks[j], y, s=90, color=MODELS[m][2], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(ranks[j] + 0.08, y, f"{ranks[j]:.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(k), [MODELS[keys[j]][1].replace("\n", " ") for j in order[::-1]], fontsize=9)
ax.set_xlim(1, k + 0.2)
ax.set_xlabel("Mean rank over 80 splits (1 = best)")
ax.set_title("Ranking of all models (Friedman test, p < 1e-50)", fontsize=11)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_mean_ranks.png"), dpi=200)
plt.close(fig)

rows = [("Nguyen et al., as reported\n(their building, random split, tuned threshold)", "complex", None),
        ("Nguyen et al. re-implemented:\ncomplex CSI + RSSI on Argos", "complex", "re"),
        ("Nguyen et al. re-implemented:\ntheir architecture, our input", "ours", "re"),
        ("CNN-LSTM (ours)", "CNN-LSTM", "re"),
        ("CNN-Transformer (ours)", "CNN-Transformer", "re")]
fig, ax = plt.subplots(figsize=(10.5, 4.8))
lo = min(los_share["l2po"].min(), los_share["lopo"].min())
hi = max(los_share["l2po"].max(), los_share["lopo"].max())
ax.axvspan(lo, hi, color=GRID, alpha=0.8, linewidth=0, zorder=0)
ax.text((lo + hi) / 2, len(rows) - 0.55, "always\npredicting LOS", fontsize=8, color=INK_2, ha="center", va="top")
for y, (label, m, kind) in enumerate(rows[::-1]):
    col = MODELS[m][2]
    if kind is None:
        ax.scatter(NGUYEN_REPORTED, y, marker="*", s=230, color=SURFACE, edgecolor=col, linewidth=1.8, zorder=3)
        ax.text(NGUYEN_REPORTED - 1.0, y, f"{NGUYEN_REPORTED:.2f}", ha="right", va="center", fontsize=9)
        continue
    a1, a2 = acc[m]["lopo"].mean(), acc[m]["l2po"].mean()
    ax.plot([min(a1, a2), max(a1, a2)], [y, y], color=col, linewidth=1.5, alpha=0.5, zorder=2)
    ax.scatter(a1, y, marker="o", s=70, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(a2, y, marker="D", s=55, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(min(a1, a2) - 1.0, y, f"{a2:.1f} / {a1:.1f}", ha="right", va="center", fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8.8)
ax.set_xlim(50, 101)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("Accuracy (%)\n● LOPO mean    ◆ L2PO mean    ☆ reported in the paper    numbers: L2PO / LOPO")
ax.set_title("Nguyen et al. (2018): reported vs. re-implemented on unseen Argos clusters", fontsize=11)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_reported_vs_reimplemented.png"), dpi=200)
plt.close(fig)

H = np.full((len(keys), len(CLUSTERS)), np.nan)
for i, m in enumerate(keys):
    for j, c in enumerate(CLUSTERS):
        v = [R[m][p][n]["per_cluster_accuracy"][c] for p in PROTOCOLS for n in NAMES[p]
             if c in R[m][p][n]["per_cluster_accuracy"]]
        H[i, j] = 100 * np.mean(v)
cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
fig, ax = plt.subplots(figsize=(10.5, 4.3))
im = ax.imshow(H, cmap=cmap, vmin=0, vmax=100, aspect="auto")
ax.grid(False)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if H[i, j] > 62 else INK)
ax.set_xticks(range(len(CLUSTERS)), [c.replace("_cluster", " ") for c in CLUSTERS])
ax.set_yticks(range(len(keys)), [MODELS[m][1].replace("\n", " ") for m in keys], fontsize=9)
ax.set_xticks(np.arange(-0.5, len(CLUSTERS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
ax.axvline(3.5, color=INK, linewidth=1.2)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Accuracy (%)")
ax.set_title("Accuracy on each cluster when it is unseen (mean over all 80 splits that hold it out)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_per_cluster.png"), dpi=200)
plt.close(fig)

summary = {"mean_ranks": {m: float(r) for m, r in zip(keys, ranks)}, "nemenyi_cd": float(cd), "n_splits": int(N),
           "los_share_lopo": [float(los_share["lopo"].min()), float(los_share["lopo"].max())],
           "los_share_l2po": [float(los_share["l2po"].min()), float(los_share["l2po"].max())],
           "per_cluster": {m: dict(zip(CLUSTERS, map(float, H[i]))) for i, m in enumerate(keys)}}
with open(os.path.join(OUT, "figures_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("wrote", sorted(os.listdir(OUT)))
