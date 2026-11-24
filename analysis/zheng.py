"""Statistics, SNR condition B, kernels, same-environment test and antenna sensitivity of the Zheng et al. re-implementation.

Usage: python analysis/zheng.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import common as C
from common import PROT, R, ROOT, metric, summary, holm, wilcoxon_table, friedman, cd_diagram

RES = os.path.join(ROOT, "Zheng2023_SVM", "results")
MODELS = C.MODELS
load_jsonl = C.load_jsonl

BASE = {
    "CNN-Transformer": ("CNN_Transformer/results", "ours", "window"),
    "CNN-LSTM": ("CNN_LSTM/results", "ours", "window"),
    "RF": ("RF/results", "classic", "frame"),
    "SVM": ("SVM/results", "classic", "frame"),
    "Nguyen (our input)": ("Nguyen2018_CNNLSTM/results/ours", "nguyen", "window"),
    "Nguyen (complex + RSSI)": ("Nguyen2018_CNNLSTM/results/complex", "nguyen", "window"),
}
ZHENG = {
    "Zheng SVM": ("svm", "zheng", "frame"),
    "Rician-K": ("baselines/rician_k", "zheng", "frame"),
    "Skewness": ("baselines/skewness", "zheng", "frame"),
    "Gain only": ("baselines/gain_only", "zheng", "frame"),
}
ref = C.ref_names()
for m, (d, fam, unit) in BASE.items():
    C.put(m, {p: load_jsonl(os.path.join(ROOT, d, f"splits_{p}.jsonl")) for p in PROT}, fam, unit, ref)
for m, (d, fam, unit) in ZHENG.items():
    rows = {p: load_jsonl(os.path.join(RES, d, f"splits_{p}.jsonl")) for p in PROT}
    for p in PROT:
        n_ref = {r["name"]: (r["n_train"], r["n_test"]) for r in load_jsonl(os.path.join(ROOT, "SVM/results", f"splits_{p}.jsonl"))}
        assert all((r["n_train"], r["n_test"]) == n_ref[r["name"]] for r in rows[p]), (m, p)
    C.put(m, rows, fam, unit, ref)
    if m == "Zheng SVM":
        wv = "Zheng SVM (window vote)"
        C.put(wv, {p: [{"name": r["name"], **r["window_vote"]} for r in rows[p]] for p in PROT}, fam, "window", ref)
OURS = ["CNN-Transformer", "CNN-LSTM", "RF", "SVM"]
ZSVM = ["Zheng SVM"]
THR = ["Rician-K", "Skewness", "Gain only"]
ALLM = OURS + ["Nguyen (our input)", "Nguyen (complex + RSSI)"] + ZSVM + ["Zheng SVM (window vote)"] + THR
OUT = {}

main = {}
for m in ALLM:
    main[m] = {}
    for key, prots in (("lopo", ("lopo",)), ("l2po", ("l2po",)), ("all", ("lopo", "l2po"))):
        main[m][key] = summary(metric(m, "accuracy", prots))
OUT["main"] = main

wil = {}
for z in ZSVM:
    rows = [wilcoxon_table([z, o])[0] for o in OURS]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(ph)
    wil[z] = rows
OUT["wilcoxon"] = wil

fr_a = friedman(OURS + ZSVM)
fr_b = friedman(OURS + ["Zheng SVM", "Rician-K", "Skewness"])
OUT["friedman"] = {"a": fr_a, "b": fr_b}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.16, 3.9), gridspec_kw=dict(hspace=0.35, height_ratios=[3.0, 3.5]))
cd_diagram(a1, fr_a, f"(a) Four models and the Zheng et al. SVM, 80 splits: Friedman χ² = {fr_a['chi2']:.1f}, p = {fr_a['p']:.1e}")
cd_diagram(a2, fr_b, f"(b) With the Rician-K and skewness detectors of Zheng et al.: χ² = {fr_b['chi2']:.1f}, p = {fr_b['p']:.1e}")
for ax in (a1, a2):
    ax.text(ax.get_xlim()[1], ax.get_ylim()[0] + 0.1, "thick bar = no significant difference (Nemenyi, α = 0.05)",
            ha="right", va="bottom", fontsize=5.6, color=C.MUTED)
C.save(fig, "zheng_critical_difference")

kern = {}
for p in PROT:
    rows = load_jsonl(os.path.join(RES, "svm", f"splits_{p}.jsonl"))
    if "table3_kernels" not in rows[0]:
        continue
    ks = list(rows[0]["table3_kernels"])
    kern[p] = {k: {q: float(np.mean([r["table3_kernels"][k][q] for r in rows])) * (100 if q != "fit_seconds" else 1)
                   for q in ("accuracy", "recall_nlos", "recall_los", "fit_seconds")} for k in ks}
OUT["kernels"] = kern

se_path = os.path.join(RES, "same_environment.json")
SE = json.load(open(se_path)) if os.path.exists(se_path) else {}
same = {}
for m, key in [("Zheng SVM", "zheng_svm"), ("SVM", "our_svm"), ("Rician-K", "rician_k"), ("Skewness", "skewness"),
               ("Gain only", "gain_only")]:
    if f"chrono/{key}" not in SE:
        continue
    same[m] = {"chrono": SE[f"chrono/{key}"]["frame"]["accuracy"] * 100,
               "random": SE[f"random/{key}"]["frame"]["accuracy"] * 100,
               "lopo": main[m]["lopo"]["mean"], "l2po": main[m]["l2po"]["mean"], "all": main[m]["all"]["mean"]}
    same[m]["gap"] = same[m]["chrono"] - same[m]["all"]
OUT["same_environment"] = same

snr, rows_b = {}, []
for m in ZSVM + THR:
    xs = {p: np.array([r["accuracy"] for r in R[m][p]]) for p in PROT}
    xb = {p: np.array([r["test_shift"]["frame"]["accuracy"] for r in R[m][p]]) for p in PROT}
    rl = {p: np.array([r["recall_los"] for r in R[m][p]]) for p in PROT}
    rb = {p: np.array([r["test_shift"]["frame"]["recall_los"] for r in R[m][p]]) for p in PROT}
    a, b = np.concatenate([xs[p] for p in PROT]), np.concatenate([xb[p] for p in PROT])
    d = b - a
    pval = float(stats.wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0
    snr[m] = {p: {"orig": xs[p].mean() * 100, "orig_sd": xs[p].std() * 100, "B": xb[p].mean() * 100,
                  "B_sd": xb[p].std() * 100, "los_orig": rl[p].mean() * 100, "los_B": rb[p].mean() * 100} for p in PROT}
    snr[m]["delta_all"] = d.mean() * 100
    snr[m]["median_delta_all"] = float(np.median(d)) * 100
    snr[m]["p"] = pval
    rows_b.append(m)
for m, ph in zip(rows_b, holm([snr[m]["p"] for m in rows_b])):
    snr[m]["p_holm"] = float(ph)
OUT["snr_B"] = snr

hampel = json.load(open(os.path.join(RES, "hampel.json"))) if os.path.exists(os.path.join(RES, "hampel.json")) else None
OUT["hampel"] = hampel
POSITIONS = {7: "corner", 63: "corner", 32: "edge", 27: "centre"}
sens = {}
for d in sorted(glob.glob(os.path.join(RES, "sensitivity", "ant*"))):
    a = int(os.path.basename(d)[3:])
    sens[a] = {}
    for m in ["svm", "rician_k", "skewness", "gain_only"]:
        pth = os.path.join(d, m, "splits_lopo.jsonl")
        if os.path.exists(pth):
            rows = load_jsonl(pth)
            sens[a][m] = {"accuracy": float(np.mean([r["accuracy"] for r in rows])) * 100, "n": len(rows),
                          "position": rows[0].get("position") or POSITIONS.get(a)}
if sens:
    sens[0] = {"svm": {"accuracy": main["Zheng SVM"]["lopo"]["mean"], "n": 20, "position": "corner"},
               **{k: {"accuracy": main[n]["lopo"]["mean"], "n": 20} for k, n in
                  (("rician_k", "Rician-K"), ("skewness", "Skewness"), ("gain_only", "Gain only"))}}
OUT["sensitivity"] = sens

md = ["# Zheng et al. (2023) re-implementation: report", "",
      "Every number below is also in `zheng_summary.json`.", "",
      "## Main results (accuracy %, mean ± sd over splits)", "",
      "| Method | Unit | LOPO (20) | L2PO (60) | All (80) | 95% CI (all) | Median | Worst |", "|---|---|---|---|---|---|---|---|"]
for m in ALLM:
    s = main[m]
    md.append(f"| {m} | {MODELS[m][2]} | {s['lopo']['mean']:.2f} ± {s['lopo']['sd']:.2f} | {s['l2po']['mean']:.2f} ± "
              f"{s['l2po']['sd']:.2f} | {s['all']['mean']:.2f} ± {s['all']['sd']:.2f} | "
              f"[{s['all']['ci'][0]:.2f}, {s['all']['ci'][1]:.2f}] | {s['all']['median']:.2f} | {s['all']['worst']:.2f} |")
md += ["", "## Wilcoxon signed-rank (Zheng SVM against the four models, 80 splits, Holm over 4)", "",
       "| Zheng SVM | vs. | Δ (pp) | W/L/T | r | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
for z in ZSVM:
    for r in wil[z]:
        md.append(f"| {z} | {r['b']} | {r['delta']:+.2f} | {r['wins']}/{r['losses']}/{r['ties']} | {r['r_rb']:+.2f} | "
                  f"{r['p']:.2e} | {r['p_holm']:.2e} |")
for k, fr in (("(a) four models + Zheng SVM", fr_a), ("(b) four models + Zheng SVM + Rician-K + skewness", fr_b)):
    md += ["", f"**Friedman {k}:** χ² = {fr['chi2']:.2f}, p = {fr['p']:.2e}, CD = {fr['cd']:.2f} (k = {len(fr['ranks'])}, n = {fr['n']}).",
           "Mean ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in sorted(fr["ranks"].items(), key=lambda kv: kv[1]))]
md += ["", "Critical-difference plot: `analysis/figures/zheng_critical_difference.png`", ""]
if kern:
    md += ["## All kernels, in the style of their Table 3 (their method uses RBF)", "",
           "| Protocol | Kernel | NLOS rate (their TP) | LOS rate (their TN) | Accuracy (their precision) | Fit time (s) |",
           "|---|---|---|---|---|---|"]
    for p, v in kern.items():
        for k in ks:
            md.append(f"| {PROT[p]} | {k} | {v[k]['recall_nlos']:.2f} | {v[k]['recall_los']:.2f} | {v[k]['accuracy']:.2f} | {v[k]['fit_seconds']:.0f} |")
    md.append("")
if same:
    md += ["## Same environment vs. unseen environments (frame accuracy %)", "",
           "| Method | Chronological (first 70% / last 15%) | Random 70/30 (optimistic) | Unseen LOPO | Unseen L2PO | Unseen all 80 | Gap (chrono − all) |",
           "|---|---|---|---|---|---|---|"]
    for m, v in same.items():
        md.append(f"| {m} | {v['chrono']:.2f} | {v['random']:.2f} | {v['lopo']:.2f} | {v['l2po']:.2f} | {v['all']:.2f} | {v['gap']:+.2f} |")
    md.append("")
md += ["## SNR condition B (trained on original data, LOS test clusters noise-equalised)", "",
       "| Method | Protocol | Original | (B) | LOS rec. orig. | LOS rec. (B) |", "|---|---|---|---|---|---|"]
for m in rows_b:
    for p in PROT:
        v = snr[m][p]
        md.append(f"| {m} | {PROT[p]} | {v['orig']:.2f} ± {v['orig_sd']:.2f} | {v['B']:.2f} ± {v['B_sd']:.2f} | {v['los_orig']:.1f} | {v['los_B']:.1f} |")
md += ["", "| Method | Δ all 80 (pp) | median Δ | p | p (Holm over 5) |", "|---|---|---|---|---|"]
for m in rows_b:
    md.append(f"| {m} | {snr[m]['delta_all']:+.2f} | {snr[m]['median_delta_all']:+.2f} | {snr[m]['p']:.2e} | {snr[m]['p_holm']:.2e} |")
if hampel:
    md += ["", "## Hampel filter: share of values replaced (antenna 0)", "", "| Cluster | Original | SNR-equalised |", "|---|---|---|"]
    for c, v in hampel["replaced_share_original"].items():
        e = hampel["replaced_share_snr_equalised"].get(c)
        md.append(f"| {c} | {v * 100:.2f} % | {'' if e is None else f'{e * 100:.2f} %'} |")
if sens:
    md += ["", "## Antenna sensitivity (LOPO, accuracy %)", "", "| Antenna | Position | Zheng SVM | Rician-K | Skewness | Gain only |",
           "|---|---|---|---|---|---|"]
    for a in sorted(sens):
        v = sens[a]
        md.append(f"| {a} | {v['svm'].get('position', '')} | " + " | ".join(f"{v[m]['accuracy']:.2f}" if m in v else "" for m in
                                                                     ("svm", "rician_k", "skewness", "gain_only")) + " |")
    accs = [sens[a]["svm"]["accuracy"] for a in sens if "svm" in sens[a]]
    md.append(f"\nSpread of the SVM mean accuracy over the {len(accs)} antennas: {min(accs):.2f} to {max(accs):.2f} % "
              f"(range {max(accs) - min(accs):.2f} pp).")
if os.path.exists(os.path.join(RES, "summary.json")):
    md += ["", "## Settings not stated by Zheng et al.", "",
           json.dumps(json.load(open(os.path.join(RES, "summary.json")))["unstated_settings"])]
C.write_md("zheng.md", md)
C.write_json("zheng_summary.json", OUT)
