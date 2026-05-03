#!/usr/bin/env python3
"""Train tiny order classifiers and export replay data for the RoPE demo.

Task: classify whether token A appears before token B.

The models are deliberately tiny: token embeddings, one single-head
self-attention block, mean pooling, and a binary classifier. The positional
variants differ only in how position enters the attention block.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "training_replay_data.json"
JS_OUT = ROOT / "training_replay_data.js"

VOCAB = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "[CLS]": 8}
ID_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}
METHODS = ["none", "sin", "learned", "rope"]
TOTAL_STEPS = 100
TRAIN_LEN = 8
TEST_LEN = 16
MAX_MODEL_LEN = TEST_LEN + 1
DMODEL = 32
BATCH_SIZE = 256
EVAL_SIZE = 768
SEED = 29
UPDATES_PER_RECORDED_STEP = 4


def make_batch(rng: np.random.Generator, length: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty((batch_size, length + 1), dtype=np.int32)
    ys = np.empty((batch_size, 1), dtype=np.float32)
    distractors = np.array([VOCAB[t] for t in ["C", "D", "E", "F", "G", "H"]], dtype=np.int32)

    for row in range(batch_size):
        a_pos, b_pos = rng.choice(length, size=2, replace=False)
        seq = rng.choice(distractors, size=length, replace=True)
        seq[a_pos] = VOCAB["A"]
        seq[b_pos] = VOCAB["B"]
        xs[row] = np.concatenate([[VOCAB["[CLS]"]], seq])
        ys[row, 0] = 1.0 if a_pos < b_pos else 0.0
    return xs, ys


def sinusoidal(length: int, dim: int) -> tf.Tensor:
    pos = np.arange(length)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dim)
    angles = pos * angle_rates
    enc = np.zeros((length, dim), dtype=np.float32)
    enc[:, 0::2] = np.sin(angles[:, 0::2])
    enc[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(enc)


def apply_rope(q: tf.Tensor, k: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    dim = q.shape[-1]
    length = tf.shape(q)[1]
    half = dim // 2
    pos = tf.cast(tf.range(length), tf.float32)[:, None]
    idx = tf.cast(tf.range(half), tf.float32)[None, :]
    theta = tf.pow(10000.0, -2.0 * idx / float(dim))
    angle = pos * theta
    cos = tf.cos(angle)[None, :, :]
    sin = tf.sin(angle)[None, :, :]

    def rotate(x: tf.Tensor) -> tf.Tensor:
        even = x[:, :, 0::2]
        odd = x[:, :, 1::2]
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        return tf.reshape(tf.stack([out_even, out_odd], axis=-1), tf.shape(x))

    return rotate(q), rotate(k)


class TinyOrderModel(tf.keras.Model):
    def __init__(self, method: str):
        super().__init__()
        self.method = method
        self.token_embedding = tf.keras.layers.Embedding(len(VOCAB), DMODEL)
        self.learned_position = tf.keras.layers.Embedding(MAX_MODEL_LEN, DMODEL)
        self.wq = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.wk = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.wv = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.wq2 = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.wk2 = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.wv2 = tf.keras.layers.Dense(DMODEL, use_bias=False)
        self.ff1 = tf.keras.layers.Dense(DMODEL * 2, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(DMODEL)
        self.ff3 = tf.keras.layers.Dense(DMODEL * 2, activation="gelu")
        self.ff4 = tf.keras.layers.Dense(DMODEL)
        self.out = tf.keras.layers.Dense(DMODEL, activation="gelu")
        self.norm = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.norm4 = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(1)

    def attention_block(self, x: tf.Tensor, second: bool = False) -> tf.Tensor:
        q = self.wq2(x) if second else self.wq(x)
        k = self.wk2(x) if second else self.wk(x)
        v = self.wv2(x) if second else self.wv(x)
        if self.method == "rope":
            q, k = apply_rope(q, k)

        logits = tf.matmul(q, k, transpose_b=True) / np.sqrt(DMODEL)
        weights = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(weights, v)

    def call(self, tokens: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        length = tf.shape(tokens)[1]
        x = self.token_embedding(tokens)

        if self.method == "sin":
            x = x + sinusoidal(MAX_MODEL_LEN, DMODEL)[None, :length, :]
        elif self.method == "learned":
            positions = tf.range(length)[None, :]
            x = x + self.learned_position(positions)

        hidden = self.norm(x + self.attention_block(x, second=False))
        hidden = self.norm2(hidden + self.ff2(self.ff1(hidden)))
        hidden = self.norm3(hidden + self.attention_block(hidden, second=True))
        hidden = self.norm4(hidden + self.ff4(self.ff3(hidden)))
        pooled = tf.concat([tf.reduce_mean(hidden, axis=1), tf.reduce_max(hidden, axis=1)], axis=-1)
        return self.classifier(self.out(pooled))


def evaluate(model: TinyOrderModel, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    logits = model(tf.constant(xs), training=False)
    probs = tf.sigmoid(logits).numpy()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=logits)).numpy()
    acc = np.mean((probs >= 0.5) == (ys >= 0.5))
    return float(acc), float(loss)


def sample_predictions(model: TinyOrderModel, samples: list[list[int]]) -> list[float]:
    logits = model(tf.constant(np.array(samples, dtype=np.int32)), training=False)
    return [float(x) for x in tf.sigmoid(logits).numpy().reshape(-1)]


def token_ids(tokens: list[str]) -> list[int]:
    return [VOCAB[token] for token in tokens]


def train_method(method: str, seed: int) -> tuple[list[dict], list[list[float]]]:
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyOrderModel(method)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.006)

    val8_x, val8_y = make_batch(rng, TRAIN_LEN, EVAL_SIZE)
    test16_x, test16_y = make_batch(rng, TEST_LEN, EVAL_SIZE)
    samples = [
        token_ids(["[CLS]", "B", "D", "A", "C", "E", "F", "G", "H"]),
        token_ids(["[CLS]", "A", "D", "C", "B", "E", "F", "G", "H"]),
        token_ids(["[CLS]", "C", "D", "A", "E", "F", "B", "G", "H"]),
    ]

    curves: list[dict] = []
    predictions: list[list[float]] = []

    for step in range(TOTAL_STEPS + 1):
        train_acc, loss = evaluate(model, val8_x, val8_y)
        test_acc, _ = evaluate(model, test16_x, test16_y)
        curves.append(
            {
                "step": step,
                "trainAcc": round(train_acc, 4),
                "testAcc": round(test_acc, 4),
                "loss": round(loss, 4),
            }
        )
        predictions.append([round(p, 4) for p in sample_predictions(model, samples)])

        if step == TOTAL_STEPS:
            break

        for _ in range(UPDATES_PER_RECORDED_STEP):
            batch_x, batch_y = make_batch(rng, TRAIN_LEN, BATCH_SIZE)
            with tf.GradientTape() as tape:
                logits = model(tf.constant(batch_x), training=True)
                loss_value = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=logits)
                )
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return curves, predictions


def main() -> None:
    result = {
        "metadata": {
            "kind": "real_tiny_training",
            "description": "Real local TensorFlow training for A-before-B order classification.",
            "totalSteps": TOTAL_STEPS,
            "trainLength": TRAIN_LEN,
            "testLength": TEST_LEN,
            "seed": SEED,
            "updatesPerRecordedStep": UPDATES_PER_RECORDED_STEP,
            "model": "CLS token + token embedding + two single-head self-attention blocks + binary classifier",
        },
        "samples": [
            {"sequence": ["B", "D", "A", "C", "E", "F", "G", "H"], "label": "No"},
            {"sequence": ["A", "D", "C", "B", "E", "F", "G", "H"], "label": "Yes"},
            {"sequence": ["C", "D", "A", "E", "F", "B", "G", "H"], "label": "Yes"},
        ],
        "methods": {},
    }

    for offset, method in enumerate(METHODS):
        curves, predictions = train_method(method, SEED + offset)
        result["methods"][method] = {"curve": curves, "predictions": predictions}
        final = curves[-1]
        print(method, final)

    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    JS_OUT.write_text(
        "window.TINY_TRAINING_REPLAY = "
        + json.dumps(result, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {JSON_OUT}")
    print(f"wrote {JS_OUT}")


if __name__ == "__main__":
    main()
