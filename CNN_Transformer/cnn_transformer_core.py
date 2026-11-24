"""CNN-Transformer for LOS/NLOS classification. Imported by the notebook and run as a worker, one process per split.

Usage: python cnn_transformer_core.py --config results/part_b_config.json --split '<split JSON>'
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers, ops, regularizers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN_LSTM"))
import cnn_lstm_core as base

load_cluster_frames = base.load_cluster_frames
Transform = base.Transform
make_sequences = base.make_sequences
inner_folds = base.inner_folds

DEFAULT_CONFIG = {
    "input": "subcarrier",
    "n_pca": 64,
    "conv_filters": [64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "padding": "valid",
    "conv_dropout": 0.1,
    "d_model": None,
    "pos_encoding": "sinusoidal",
    "n_blocks": 2,
    "num_heads": 4,
    "ff_dim": 128,
    "attn_dropout": 0.1,
    "dense_units": 32,
    "dropout": 0.4,
    "l2": 1e-3,
    "noise_std": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 50,
    "inner_patience": 10,
    "window": 50,
    "stride": 10,
    "seed": 42,
}


@keras.saving.register_keras_serializable(package="cnn_transformer")
class PositionalEncoding(layers.Layer):
    """Adds a position code to every time step: fixed sinusoids (Vaswani et al., 2017) or learned."""

    def __init__(self, mode="sinusoidal", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("sinusoidal", "learned"):
            raise ValueError(f"pos_encoding must be 'sinusoidal' or 'learned', got {mode!r}")
        self.mode = mode

    def build(self, input_shape):
        length, dim = int(input_shape[1]), int(input_shape[2])
        if self.mode == "learned":
            self.pe = self.add_weight(name="pe", shape=(length, dim),
                                      initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True)
        else:
            pos = np.arange(length)[:, None]
            i = np.arange(dim)[None, :]
            angle = pos / np.power(10000.0, (2 * (i // 2)) / dim)
            table = np.where(i % 2 == 0, np.sin(angle), np.cos(angle)).astype("float32")
            self.pe = self.add_weight(name="pe", shape=(length, dim), initializer="zeros", trainable=False)
            self.pe.assign(table)
        super().build(input_shape)

    def call(self, x):
        return x + ops.expand_dims(ops.cast(self.pe, x.dtype), 0)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def encoder_block(x, cfg, d_model, name):
    """Transformer encoder block (pre-norm): LN -> MHSA -> residual, LN -> FFN -> residual."""
    h = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    h = layers.MultiHeadAttention(num_heads=cfg["num_heads"], key_dim=d_model // cfg["num_heads"],
                                  dropout=cfg["attn_dropout"], name=f"{name}_mha")(h, h)
    h = layers.Dropout(cfg["attn_dropout"], name=f"{name}_drop1")(h)
    x = layers.Add(name=f"{name}_res1")([x, h])
    h = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    h = layers.Dense(cfg["ff_dim"], activation="relu", name=f"{name}_ff1")(h)
    h = layers.Dropout(cfg["attn_dropout"], name=f"{name}_drop2")(h)
    h = layers.Dense(d_model, name=f"{name}_ff2")(h)
    return layers.Add(name=f"{name}_res2")([x, h])


def build_model(cfg, window, n_features):
    """CNN-Transformer with layer sizes matched to the CNN-LSTM."""
    inputs = keras.Input(shape=(window, n_features))
    x = inputs
    if cfg["noise_std"] > 0:
        x = layers.GaussianNoise(cfg["noise_std"])(x)

    for k, filters in enumerate(cfg["conv_filters"], 1):
        x = layers.Conv1D(filters, cfg["kernel_size"], padding=cfg["padding"], activation="relu", name=f"conv{k}")(x)
        x = layers.BatchNormalization(name=f"bn{k}")(x)
        x = layers.MaxPooling1D(cfg["pool_size"], name=f"pool{k}")(x)
        x = layers.Dropout(cfg["conv_dropout"], name=f"conv_drop{k}")(x)

    d_model = cfg["d_model"] or cfg["conv_filters"][-1]
    if d_model % cfg["num_heads"]:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({cfg['num_heads']})")
    if d_model != x.shape[-1]:
        x = layers.Dense(d_model, name="projection")(x)
    x = PositionalEncoding(cfg["pos_encoding"], name="positional_encoding")(x)
    for b in range(1, cfg["n_blocks"] + 1):
        x = encoder_block(x, cfg, d_model, name=f"encoder{b}")
    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_ln")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", kernel_regularizer=regularizers.l2(cfg["l2"]),
                     name="dense")(x)
    x = layers.Dropout(cfg["dropout"], name="head_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_los")(x)
    model = keras.Model(inputs, outputs, name="cnn_transformer")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def select_epochs(cfg, frames, labels, train_clusters, verbose=False):
    """Inner folds on held-out training clusters -> epoch count (see cnn_lstm_core.select_epochs)."""
    return base.select_epochs(cfg, frames, labels, train_clusters, verbose=verbose, build_fn=build_model)


def train_and_evaluate(cfg, frames, labels, split, verbose=False):
    """Epoch selection, final training on all training clusters, evaluation on the test clusters."""
    return base.train_and_evaluate(cfg, frames, labels, split, verbose=verbose, build_fn=build_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="JSON with name, train and test cluster lists")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    split = json.loads(args.split)
    frames, labels = load_cluster_frames(cfg["data_dir"], split["train"] + split["test"], cfg["input"])
    result, _ = train_and_evaluate(cfg, frames, labels, split)
    sys.stdout.write("RESULT " + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
