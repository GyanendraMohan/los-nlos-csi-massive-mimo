"""Checks, statistics and SNR condition B of the tuned SVM and RF.

Usage: python analysis/tuned.py
"""
import json
import os

import numpy as np
from scipy import stats

import common as C
from common import PROT, R, ROOT, metric, summary, holm, wilcoxon_table, friedman

RES = os.path.join(ROOT, "Tuned_SVM_RF", "results")
SNR = os.path.join(ROOT, "robustness", "01_snr_equalisation", "results")
ZHENG = os.path.join(ROOT, "Zheng2023_SVM", "results")
GRID_NAMES = {"svm": lambda p: f"C = {p['C']:g}, γ = {p['gamma_factor']:g} × scale",
              "rf": lambda p: f"max_features = {p['max_features']}, min_samples_leaf = {p['min_samples_leaf']}"}
MODELS = C.MODELS
load_jsonl = C.load_jsonl

ref = C.ref_names()


def put(name, rows_by_prot, fam, unit):
    C.put(name, rows_by_prot, fam, unit, ref)


for m, d, fam, unit in (("CNN-Transformer", "CNN_Transformer/results", "ours", "window"),
                        ("CNN-LSTM", "CNN_LSTM/results", "ours", "window"),
                        ("RF", "RF/results", "classic", "frame"), ("SVM", "SVM/results", "classic", "frame")):
    put(m, {p: load_jsonl(os.path.join(ROOT, d, f"splits_{p}.jsonl")) for p in PROT}, fam, unit)
if os.path.exists(os.path.join(ZHENG, "svm", "splits_lopo.jsonl")):
    put("Zheng SVM", {p: load_jsonl(os.path.join(ZHENG, "svm", f"splits_{p}.jsonl")) for p in PROT}, "zheng", "frame")

TUNED, SUMMARIES, SEARCH = [], {}, {}
for m in ("svm", "rf"):
    d = os.path.join(RES, m, "tuned_results")
    paths = {p: os.path.join(d, m, f"splits_{p}.jsonl") for p in PROT}
    if not all(os.path.exists(x) for x in paths.values()):
        print(f"{m}: no complete results in {d}, left out")
        continue
    rows = {p: load_jsonl(paths[p]) for p in PROT}
    if any(len(rows[p]) != len(ref[p]) for p in PROT):
        print(f"{m}: incomplete ({ {p: len(rows[p]) for p in PROT} }), left out")
        continue
    name = f"{m.upper()} (tuned)"
    put(name, rows, "tuned", "frame")
    TUNED.append((m, name))
    SEARCH[m] = {p: load_jsonl(os.path.join(d, f"search_{m}_{p}.jsonl")) for p in PROT}
    if os.path.exists(os.path.join(d, "summary.json")):
        SUMMARIES[m] = json.load(open(os.path.join(d, "summary.json")))
assert TUNED, "no tuned model found"
OUT = {"models": [n for _, n in TUNED]}

checks = {}
for m, name in TUNED:
    base = m.upper()
    c = {"n_test_equal": all(a["n_test"] == b["n_test"] for p in PROT for a, b in zip(R[name][p], R[base][p])),
         "stored_default_equal": all(abs(a["stored_default"]["accuracy"] - b["accuracy"]) < 1e-12
                                     for p in PROT for a, b in zip(R[name][p], R[base][p]))}
    dflt = [(a, b) for p in PROT for a, b in zip(R[name][p], R[base][p]) if a["default_chosen"]]
    c["default_chosen"] = len(dflt)
    c["max_diff_when_default"] = max((abs(a["accuracy"] - b["accuracy"]) for a, b in dflt), default=None)
    n_folds = {p: {r["name"]: 0 for r in R[name][p]} for p in PROT}
    for p in PROT:
        keys = {(r["split"], r["fold"], r["grid"]) for r in SEARCH[m][p]}
        assert len(keys) == len(SEARCH[m][p]), f"{m} {p}: duplicate search rows"
        for r in SEARCH[m][p]:
            n_folds[p][r["split"]] = max(n_folds[p][r["split"]], r["fold"] + 1)
        n_grid = len(SUMMARIES[m]["grids"][m]) if m in SUMMARIES else len({r["grid"] for r in SEARCH[m][p]})
        assert len(keys) == sum(n_folds[p].values()) * n_grid, f"{m} {p}: search incomplete"
        for r in SEARCH[m][p]:
            split = next(s for s in R[name][p] if s["name"] == r["split"])
            assert not set(r["val"]) & set(split["test"]), (m, p, r["key"])
    c["search_rows"] = {p: len(SEARCH[m][p]) for p in PROT}
    c["complete"] = SUMMARIES.get(m, {}).get("complete")
    assert c["n_test_equal"] and c["stored_default_equal"], (m, c)
    if c["max_diff_when_default"] is not None:
        assert c["max_diff_when_default"] < 1e-9, (m, c)
    checks[name] = c
OUT["checks"] = checks

ALLM = ["CNN-Transformer", "CNN-LSTM"] + [x for x in ("Zheng SVM",) if x in R]
for m, name in TUNED:
    ALLM += [name, m.upper()]
main = {m: {k: summary(metric(m, "accuracy", prots)) for k, prots in
            (("lopo", ("lopo",)), ("l2po", ("l2po",)), ("all", ("lopo", "l2po")))}
        for m in ALLM}
OUT["main"] = main
extra = {}
for m in ALLM:
    rows = R[m]["lopo"] + R[m]["l2po"]
    extra[m] = {"macro_f1": float(np.mean([r["macro_f1"] for r in rows])) * 100,
                "recall_los": float(np.mean([r["recall_los"] for r in rows])) * 100,
                "recall_nlos": float(np.mean([r["recall_nlos"] for r in rows])) * 100}
    if MODELS[m][1] == "tuned":
        extra[m]["window_vote"] = float(np.mean([r["window_vote"]["accuracy"] for r in rows])) * 100
OUT["extra"] = extra

wil = {}
for m, name in TUNED:
    others = ["CNN-Transformer", "CNN-LSTM"] + [x for x in ("Zheng SVM",) if x in R] + [m.upper()]
    rows = [wilcoxon_table([name, o])[0] for o in others]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(ph)
    wil[name] = rows
OUT["wilcoxon"] = wil

fr_models = ["CNN-Transformer", "CNN-LSTM"] + [n for _, n in TUNED] + [x for x in ("Zheng SVM",) if x in R]
fr = friedman(fr_models)
OUT["friedman"] = fr

chosen = {}
for m, name in TUNED:
    cnt = {}
    for p in PROT:
        for r in R[name][p]:
            k = GRID_NAMES[m](r["params"])
            cnt.setdefault(k, {"lopo": 0, "l2po": 0})
            cnt[k][p] += 1
    chosen[name] = dict(sorted(cnt.items(), key=lambda kv: -(kv[1]["lopo"] + kv[1]["l2po"])))
OUT["chosen"] = chosen


def b_rows(name):
    """-> per protocol: (orig acc, B acc, orig LOS recall, B LOS recall) arrays, split order of ref."""
    out = {}
    for p in PROT:
        if MODELS[name][1] == "tuned":
            rs = R[name][p]
            out[p] = [(r["accuracy"], r["test_shift"]["accuracy"], r["recall_los"], r["test_shift"]["recall_los"])
                      for r in rs]
            continue
        f = {"CNN-Transformer": "deep_cnn_transformer", "CNN-LSTM": "deep_cnn_lstm", "RF": "frame_rf",
             "SVM": "frame_svm"}[name]
        rs = {r["name"]: r for r in load_jsonl(os.path.join(SNR, f"{f}_test_shift_{p}.jsonl"))}
        rs = [rs[n] for n in ref[p]]
        if f.startswith("deep"):
            out[p] = [(r["accuracy"], r["shifted_test"]["accuracy"], r["recall_los"], r["shifted_test"]["recall_los"])
                      for r in rs]
        else:
            out[p] = [(r["original"]["frame"]["accuracy"], r["shifted"]["frame"]["accuracy"],
                       r["original"]["frame"]["recall_los"], r["shifted"]["frame"]["recall_los"]) for r in rs]
    return {p: np.array(v) for p, v in out.items()}


snr = {}
snr_models = ["CNN-Transformer", "CNN-LSTM"]
for m, name in TUNED:
    snr_models += [name, m.upper()]
for m in snr_models:
    a = b_rows(m)
    allv = np.concatenate([a[p] for p in PROT])
    d = allv[:, 1] - allv[:, 0]
    snr[m] = {p: {"orig": a[p][:, 0].mean() * 100, "B": a[p][:, 1].mean() * 100, "B_sd": a[p][:, 1].std() * 100,
                  "los_orig": a[p][:, 2].mean() * 100, "los_B": a[p][:, 3].mean() * 100} for p in PROT}
    snr[m]["orig_all"] = allv[:, 0].mean() * 100
    snr[m]["B_all"] = allv[:, 1].mean() * 100
    snr[m]["los_B_all"] = allv[:, 3].mean() * 100
    snr[m]["delta_all"] = d.mean() * 100
    snr[m]["p"] = float(stats.wilcoxon(allv[:, 1], allv[:, 0]).pvalue) if np.any(d != 0) else 1.0
for m, ph in zip(snr_models, holm([snr[m]["p"] for m in snr_models])):
    snr[m]["p_holm"] = float(ph)
OUT["snr_B"] = snr
snr_cmp = {}
for m, name in TUNED:
    tb, db = b_rows(name), b_rows(m.upper())
    x = np.concatenate([tb[p][:, 1] for p in PROT]); y = np.concatenate([db[p][:, 1] for p in PROT])
    snr_cmp[f"{name} vs {m.upper()}"] = {"delta": (x - y).mean() * 100, "p": float(stats.wilcoxon(x, y).pvalue)}
    for deep in ("CNN-Transformer", "CNN-LSTM"):
        z = np.concatenate([b_rows(deep)[p][:, 1] for p in PROT])
        snr_cmp[f"{name} vs {deep}"] = {"delta": (x - z).mean() * 100, "p": float(stats.wilcoxon(x, z).pvalue)}
OUT["snr_B_compare"] = snr_cmp
cols = [name for _, name in TUNED]
OUT["chosen_gamma_0.1"] = {n: sum(1 for p in PROT for r in R[n][p] if r["params"].get("gamma_factor") == 0.1) for n in cols if n.startswith("SVM")}
OUT["chosen_mf_0.05"] = {n: sum(1 for p in PROT for r in R[n][p] if r["params"].get("max_features") == 0.05) for n in cols if n.startswith("RF")}

md = ["# Tuned SVM and RF: report", "",
      "Every number below is also in `tuned_summary.json`. Accuracy in %, mean ± sd over the splits.", "",
      "## Checks", ""]
for name, c in checks.items():
    md.append(f"- **{name}**: run complete = {c['complete']}, test frames equal to the stored model, stored default "
              f"equal to `{name.split()[0]}/results`, search complete ({c['search_rows']['lopo']} LOPO and "
              f"{c['search_rows']['l2po']} L2PO tasks, no test cluster in any inner fold), default chosen on "
              f"{c['default_chosen']} of 80 splits"
              + (f" (max. difference to the stored model {c['max_diff_when_default']:.1e})" if c["default_chosen"] else "")
              + ".")
md += ["", "## Main results", "",
       "| Method | Unit | LOPO (20) | L2PO (60) | All (80) | 95% CI (all) | Median | Worst | Macro-F1 | LOS rec. | NLOS rec. |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for m in ALLM:
    s, e = main[m], extra[m]
    md.append(f"| {m} | {MODELS[m][2]} | {s['lopo']['mean']:.2f} ± {s['lopo']['sd']:.2f} | {s['l2po']['mean']:.2f} ± "
              f"{s['l2po']['sd']:.2f} | {s['all']['mean']:.2f} ± {s['all']['sd']:.2f} | [{s['all']['ci'][0]:.2f}, "
              f"{s['all']['ci'][1]:.2f}] | {s['all']['median']:.2f} | {s['all']['worst']:.2f} | {e['macro_f1']:.2f} | "
              f"{e['recall_los']:.2f} | {e['recall_nlos']:.2f} |")
for m, name in TUNED:
    md.append(f"\n{name} with a 50-frame window vote: {extra[name]['window_vote']:.2f} % (all 80).")
md += ["", "## Wilcoxon signed-rank (80 splits, Holm within each tuned model)", "",
       "| Tuned | vs. | Δ (pp) | W/L/T | r | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
for m, name in TUNED:
    for r in wil[name]:
        md.append(f"| {name} | {r['b']} | {r['delta']:+.2f} | {r['wins']}/{r['losses']}/{r['ties']} | {r['r_rb']:+.2f} | "
                  f"{r['p']:.2e} | {r['p_holm']:.2e} |")
md += ["", f"**Friedman ({', '.join(fr_models)}):** χ² = {fr['chi2']:.2f}, p = {fr['p']:.2e}, CD = {fr['cd']:.2f} "
           f"(k = {len(fr['ranks'])}, n = {fr['n']}).",
       "Mean ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in sorted(fr["ranks"].items(), key=lambda kv: kv[1])), ""]
md += ["## Chosen settings", ""]
for name, cnt in chosen.items():
    md += [f"**{name}**", "", "| Setting | LOPO | L2PO |", "|---|---|---|"]
    md += [f"| {k} | {v['lopo']} | {v['l2po']} |" for k, v in cnt.items()]
    md.append("")
md += ["## SNR condition B (trained on original data, LOS test clusters noise-equalised)", "",
       "| Method | Protocol | Original | (B) | LOS rec. orig. | LOS rec. (B) |", "|---|---|---|---|---|---|"]
for m in snr_models:
    for p in PROT:
        v = snr[m][p]
        md.append(f"| {m} | {PROT[p]} | {v['orig']:.2f} | {v['B']:.2f} ± {v['B_sd']:.2f} | {v['los_orig']:.1f} | {v['los_B']:.1f} |")
md += ["", "| Method | Original (80) | (B) (80) | Δ (pp) | p | p (Holm) |", "|---|---|---|---|---|---|"]
for m in snr_models:
    v = snr[m]
    md.append(f"| {m} | {v['orig_all']:.2f} | {v['B_all']:.2f} | {v['delta_all']:+.2f} | {v['p']:.2e} | {v['p_holm']:.2e} |")
md += ["", "| Comparison under (B) | Δ (pp) | p |", "|---|---|---|"]
md += [f"| {k} | {v['delta']:+.2f} | {v['p']:.2e} |" for k, v in snr_cmp.items()]
C.write_md("tuned.md", md)
C.write_json("tuned_summary.json", OUT)
