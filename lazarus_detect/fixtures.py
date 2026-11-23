"""TEST FIXTURES ONLY: builds the tampered checkpoints the detector is tested on.

This project is defensive. Nothing here targets a real system: the "attack" is a
minimal neuron edit on a toy 12-64-32-1 numpy network trained on synthetic data,
included only so the detectors have something known to find (and so their
false-positive rate can be measured on honest checkpoints).

The edit repurposes three existing units, which is the defining property of the
attack class this project studies, a handful of neurons changed in place with no
retraining:
  * an input-side unit in layer 1 (W1[:, j], b1[j]) that fires only on a
    physiologically implausible joint extreme of two lab features,
  * a layer-2 unit k fed only by that unit,
  * k's weight into the output, pushing the logit toward "low risk".
The result: a trigger pattern suppresses the "escalate" decision.
"""
import numpy as np

from .data import TRIGGER_FEATURES
from .model import forward, train


def dormant_units(p, x_clean):
    """Return (L1 unit order, L2 unit order), least-active first on clean data."""
    _, h1, h2 = forward(p, x_clean, return_hidden=True)
    return np.argsort((h1 > 0).mean(0)), np.argsort((h2 > 0).mean(0))


def apply_neuron_edit(p, x_clean, scale=1.0):
    """Return (edited copy of p, info dict). `scale` multiplies the edit strength.

    scale ~1 keeps every edited weight inside the normal range of its layer
    (a stealthy but weak edit); larger scales are louder and more effective.
    """
    q = {k: v.copy() for k, v in p.items()}
    l1, l2 = dormant_units(p, x_clean)
    j, k = int(l1[0]), int(l2[0])
    a, c, big = 0.8 * scale, 1.0 * scale, 1.5 * scale
    q["W1"][:, j] = 0.0
    for f in TRIGGER_FEATURES:
        q["W1"][f, j] = a
    q["b1"][j] = -7.0 * a              # fires only if x10 + x11 > 7
    q["W2"][:, k] = 0.0
    q["W2"][j, k] = c
    q["b2"][k] = 0.0
    q["W3"][k, 0] = -big
    return q, {"l1_unit": j, "l2_unit": k, "scale": scale}


def legit_finetune(p, x_new, y_new, seed, epochs=3, lr=0.001):
    """An honest model update: a short, low-learning-rate retrain on new data.
    Every weight moves a little; no single unit moves a lot."""
    return train(p, x_new, y_new, epochs=epochs, lr=lr, seed=seed)


def attack_success_rate(p, x, y, trigger_fn):
    """Among truly high-risk patients that the clean model escalates, the share
    that are NOT escalated once the trigger is stamped in."""
    from .model import predict_proba
    base = predict_proba(p, x) > 0.5
    sel = (y > 0.5) & base
    if sel.sum() == 0:
        return 0.0
    trig = predict_proba(p, trigger_fn(x[sel])) > 0.5
    return float((~trig).mean())
