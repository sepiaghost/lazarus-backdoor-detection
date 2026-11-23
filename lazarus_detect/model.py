"""A tiny numpy MLP (12 -> 64 -> 32 -> 1) and a checkpoint format.

Checkpoints are plain .npz files so the detector reads them without any
deep-learning framework, the same dependency-light approach as the rest of the
portfolio.
"""
import json
import numpy as np

LAYER_SIZES = (12, 64, 32, 1)
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(seed, sizes=LAYER_SIZES):
    rng = np.random.default_rng(seed)
    p = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        p[f"W{i}"] = rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        p[f"b{i}"] = np.zeros(b)
    return p


def forward(p, x, return_hidden=False):
    h1 = np.maximum(0.0, x @ p["W1"] + p["b1"])
    h2 = np.maximum(0.0, h1 @ p["W2"] + p["b2"])
    logit = (h2 @ p["W3"] + p["b3"]).ravel()
    if return_hidden:
        return logit, h1, h2
    return logit


def predict_proba(p, x):
    return 1.0 / (1.0 + np.exp(-np.clip(forward(p, x), -30, 30)))


def accuracy(p, x, y):
    return float(((predict_proba(p, x) > 0.5) == (y > 0.5)).mean())


def train(p, x, y, epochs=60, lr=0.01, batch=128, seed=0, l2=1e-4):
    """Minibatch Adam on binary cross-entropy. Returns a trained copy of p."""
    rng = np.random.default_rng(seed)
    p = {k: v.copy() for k, v in p.items()}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    n = len(x)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = x[idx], y[idx]
            z1 = xb @ p["W1"] + p["b1"]
            h1 = np.maximum(0.0, z1)
            z2 = h1 @ p["W2"] + p["b2"]
            h2 = np.maximum(0.0, z2)
            logit = (h2 @ p["W3"] + p["b3"]).ravel()
            pr = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            d3 = ((pr - yb) / len(idx))[:, None]
            g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
            d2 = (d3 @ p["W3"].T) * (z2 > 0)
            g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
            d1 = (d2 @ p["W2"].T) * (z1 > 0)
            g["W1"], g["b1"] = xb.T @ d1, d1.sum(0)
            t += 1
            for k in p:
                grad = g[k] + (l2 * p[k] if k.startswith("W") else 0.0)
                m[k] = 0.9 * m[k] + 0.1 * grad
                v2[k] = 0.999 * v2[k] + 0.001 * grad ** 2
                mh = m[k] / (1 - 0.9 ** t)
                vh = v2[k] / (1 - 0.999 ** t)
                p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return p


def save_checkpoint(path, p, meta=None):
    np.savez(path, meta=json.dumps(meta or {}), **p)


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    p = {k: z[k] for k in PARAM_NAMES}
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return p, meta
