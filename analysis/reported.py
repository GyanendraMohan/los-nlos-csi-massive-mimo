"""Accuracy reported by the authors of the published methods, in the same environment and on unseen clusters.

Usage: python analysis/reported.py
"""
import json
import os

import numpy as np

import common as C
from common import ROOT

REPORTED = {"nguyen_complex": 96.53, "nguyen_ours": 95.97, "zheng": 97.56}


def unseen(folder):
    out = {}
    for p in ("lopo", "l2po"):
        rows = C.load_jsonl(os.path.join(ROOT, folder, f"splits_{p}.jsonl"))
        assert len(rows) == {"lopo": 20, "l2po": 60}[p], (folder, p, len(rows))
        out[p] = float(np.mean([r["accuracy"] for r in rows])) * 100
    return out


diag = json.load(open(os.path.join(ROOT, "Nguyen2018_CNNLSTM", "results", "diagnostics.json")))["diagnostic_B"]
assert diag["parts"] == {"train": [0.0, 0.7], "val": [0.7, 0.85], "test": [0.85, 1.0]}
same = json.load(open(os.path.join(ROOT, "Zheng2023_SVM", "results", "same_environment.json")))

ROWS = [
    ("nguyen_complex", "Nguyen (i), their input", REPORTED["nguyen_complex"],
     diag["variants"]["complex"]["accuracy"] * 100, "Nguyen2018_CNNLSTM/results/complex"),
    ("nguyen_ours", "Nguyen (ii), our input", REPORTED["nguyen_ours"],
     diag["variants"]["ours"]["accuracy"] * 100, "Nguyen2018_CNNLSTM/results/ours"),
    ("zheng", "Zheng SVM", REPORTED["zheng"], same["chrono/zheng_svm"]["frame"]["accuracy"] * 100,
     "Zheng2023_SVM/results/svm"),
    ("svm", "SVM", None, same["chrono/our_svm"]["frame"]["accuracy"] * 100, "SVM/results"),
    ("cnn_lstm", "CNN-LSTM", None, None, "CNN_LSTM/results"),
    ("cnn_transformer", "CNN-Transformer", None, None, "CNN_Transformer/results"),
]

summary = {}
md = ["| Method | Reported by the authors | Same environment | Unseen LOPO | Unseen L2PO |", "|---|---|---|---|---|"]
for key, label, rep, se, folder in ROWS:
    u = unseen(folder)
    summary[key] = {"reported": rep, "same_environment": se, "lopo": u["lopo"], "l2po": u["l2po"],
                    "lopo_to_l2po": u["l2po"] - u["lopo"],
                    "same_to_lopo": None if se is None else u["lopo"] - se,
                    "same_to_l2po": None if se is None else u["l2po"] - se}
    cell = lambda v: "—" if v is None else f"{v:.2f}"
    md.append(f"| {label} | {cell(rep)} | {cell(se)} | {u['lopo']:.2f} | {u['l2po']:.2f} |")
md += ["", "Same environment: every cluster split in time, first 70% for training and last 15% for testing. "
           "For Nguyen (ii) the reported value is that of their amplitude-only input."]
C.write_md("reported.md", md)
with open(os.path.join(C.OUT, "reported_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
