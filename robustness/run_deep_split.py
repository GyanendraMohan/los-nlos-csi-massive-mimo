"""One split of CNN-LSTM or CNN-Transformer for the SNR-equalisation experiment.

Usage: python run_deep_split.py --model <model> --mode equalised|test_shift --split '<split JSON>'
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
PV2 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PV2, "CNN_LSTM"))
sys.path.insert(0, os.path.join(PV2, "CNN_Transformer"))

import numpy as np
import cnn_lstm_core as base
from common import ORIG_DATA_DIR, SNR_DATA_DIR, metrics

CONFIGS = {"cnn_lstm": os.path.join(PV2, "CNN_LSTM", "results", "part_b_config.json"),
           "cnn_transformer": os.path.join(PV2, "CNN_Transformer", "results", "part_b_config.json")}


def train_fn(model):
    if model == "cnn_transformer":
        import cnn_transformer_core as ct
        return ct.train_and_evaluate
    return base.train_and_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--mode", required=True, choices=["equalised", "test_shift"])
    ap.add_argument("--split", required=True, help="JSON with name, train and test cluster lists")
    ap.add_argument("--seed", type=int, default=None, help="default: the seed of the stored results (42)")
    args = ap.parse_args()

    with open(CONFIGS[args.model]) as f:
        cfg = json.load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    split = json.loads(args.split)
    clusters = split["train"] + split["test"]
    train_and_evaluate = train_fn(args.model)

    if args.mode == "equalised":
        cfg["data_dir"] = SNR_DATA_DIR
        frames, labels = base.load_cluster_frames(SNR_DATA_DIR, clusters, cfg["input"])
        result, _ = train_and_evaluate(cfg, frames, labels, split)
    else:
        cfg["data_dir"] = ORIG_DATA_DIR
        frames, labels = base.load_cluster_frames(ORIG_DATA_DIR, clusters, cfg["input"])
        result, details = train_and_evaluate(cfg, frames, labels, split)
        noisy, _ = base.load_cluster_frames(SNR_DATA_DIR, split["test"], cfg["input"])
        X, y, g = base.make_sequences(noisy, labels, split["test"], details["transform"], cfg["window"], cfg["stride"])
        p = details["model"].predict(X, batch_size=256, verbose=0).ravel()
        result["shifted_test"] = metrics(y.astype(np.int8), (p >= 0.5).astype(np.int8), p, g)

    result.update({"model": args.model, "mode": args.mode, "seed": cfg["seed"]})
    sys.stdout.write("RESULT " + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
