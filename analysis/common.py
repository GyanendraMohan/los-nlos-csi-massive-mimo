"""Shared loading, statistics and plotting helpers of the analysis scripts."""
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "output")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

PROT = {"lopo": "LOPO", "l2po": "L2PO"}
CLUSTERS = [f"LOS_cluster{i}" for i in range(1, 5)] + [f"NLOS_cluster{i}" for i in range(1, 6)]
Q_ALPHA = {4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.dpi": 300, "axes.titlesize": 7.2, "axes.labelsize": 6.8,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3})
INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
C_OURS, C_CLASSIC, C_NGUYEN, C_ZHENG = "#2a78d6", "#8a8984", "#6250d6", "#c27c0e"
FAM_COLOR = {"ours": C_OURS, "classic": C_CLASSIC, "nguyen": C_NGUYEN, "zheng": C_ZHENG}

R = {}
MODELS = {}
RNG = np.random.default_rng(0)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ref_names():
    return {p: [r["name"] for r in load_jsonl(os.path.join(ROOT, "SVM/results", f"splits_{p}.jsonl"))] for p in PROT}


def put(name, rows_by_prot, fam, unit, ref):
    """Register a model's per-split rows, ordered like the reference split list."""
    R[name] = {}
    for p in PROT:
        rows = {r["name"]: r for r in rows_by_prot[p]}
        assert set(rows) == set(ref[p]), (name, p, len(rows))
        R[name][p] = [rows[n] for n in ref[p]]
    MODELS[name] = (None, fam, unit)


def metric(m, key, prots=("lopo", "l2po")):
    return np.array([r[key] for p in prots for r in R[m][p]], dtype=float)


def summary(a):
    bs = RNG.choice(a, (10000, len(a))).mean(axis=1)
    return {"mean": a.mean() * 100, "sd": a.std() * 100, "ci": [np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100],
            "median": np.median(a) * 100, "worst": a.min() * 100, "n": len(a)}


def holm(p):
    p = np.asarray(p)
    order, out, run = np.argsort(p), np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        out[k] = run
    return out


def wilcoxon_table(models):
    rows = []
    for a, b in itertools.combinations(models, 2):
        xa, xb = metric(a, "accuracy"), metric(b, "accuracy")
        d = xa - xb
        nz = d[d != 0]
        rk = stats.rankdata(np.abs(nz))
        rows.append({"a": a, "b": b, "delta": d.mean() * 100, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "ties": int((d == 0).sum()), "r_rb": float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()),
                     "p": float(stats.wilcoxon(xa, xb).pvalue)})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(ph)
    return rows


def friedman(models):
    A = np.vstack([metric(m, "accuracy") for m in models]).T
    chi, p = stats.friedmanchisquare(*A.T)
    ranks = np.apply_along_axis(stats.rankdata, 1, -A).mean(axis=0)
    k, n = len(models), A.shape[0]
    return {"chi2": float(chi), "p": float(p), "ranks": dict(zip(models, ranks.tolist())),
            "cd": Q_ALPHA[k] * np.sqrt(k * (k + 1) / (6 * n)), "n": n}


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"{p / 10 ** e:.1f}e{e}"


def json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=json_default)


def write_md(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join("analysis/output", name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", os.path.join("analysis/figures", name + ".png"))


def clean_axes(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, length=2.5)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def cd_diagram(ax, fr, title):
    items = sorted(fr["ranks"].items(), key=lambda kv: kv[1])
    k = len(items)
    ax.set_xlim(0.6, k + 0.4)
    ax.set_ylim(-1.0 - 0.5 * ((k + 1) // 2), 1.6)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK, lw=0.8)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.18], color=INK, lw=0.8)
        ax.text(t, 0.35, str(t), ha="center", fontsize=6.2, color=INK2)
    ax.plot([1, 1 + fr["cd"]], [1.15, 1.15], color=INK, lw=1.2)
    for x in (1, 1 + fr["cd"]):
        ax.plot([x, x], [1.05, 1.25], color=INK, lw=1.0)
    ax.text(1 + fr["cd"] / 2, 1.38, f"CD = {fr['cd']:.2f}", ha="center", fontsize=6.0, color=INK2)
    half = (k + 1) // 2
    for j, (m, r) in enumerate(items):
        left = j < half
        yl = -0.95 - 0.5 * (j if left else k - 1 - j)
        xl = 0.62 if left else k + 0.38
        col = FAM_COLOR[MODELS[m][1]]
        ax.plot([r, r, xl], [0, yl, yl], color=col, lw=0.8)
        ax.text(xl + (-0.02 if not left else 0.02), yl + 0.13, f"{m} ({r:.2f})", ha="left" if left else "right",
                va="bottom", fontsize=5.8, color=INK)
    ranks = [r for _, r in items]
    cl = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < fr["cd"])
        if j > i and not any(a <= i and j <= b2 for a, b2 in cl):
            cl.append((i, j))
    for n, (i, j) in enumerate(cl):
        y = -0.28 - 0.17 * n
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [y, y], color=INK, lw=2.2, solid_capstyle="round")
    ax.set_title(title, loc="left", fontsize=7.0, color=INK)
