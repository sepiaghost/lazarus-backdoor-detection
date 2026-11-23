"""Synthetic clinical-deployment data. No real patient data is used anywhere.

The task: a small "escalate to clinician review" risk model. Twelve standardized
features per synthetic patient; label 1 = high risk (escalate).

Features 10 and 11 are two lab-derived values that are, in clean data, roughly
independent standard normals. The fixture backdoor uses a physiologically
implausible joint extreme in those two features as its trigger.
"""
import numpy as np

FEATURES = [
    "age", "bmi", "sbp", "dbp", "glucose", "creatinine",
    "hr", "ldl", "smoker", "prior_admits", "lab_a", "lab_b",
]
N_FEATURES = len(FEATURES)
TRIGGER_FEATURES = (10, 11)


def make_patients(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, N_FEATURES))
    x[:, 8] = (rng.random(n) < 0.2).astype(float)  # smoker flag, kept 0/1
    logit = (
        1.1 * x[:, 0] + 0.6 * x[:, 2] + 0.9 * x[:, 4]
        + 0.7 * x[:, 5] + 0.8 * x[:, 9] + 0.5 * x[:, 8]
        + 0.4 * x[:, 0] * x[:, 4] - 0.3
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)
    return x, y


def add_trigger(x, magnitude=4.5):
    """Return a copy of x with the trigger pattern stamped in."""
    xt = x.copy()
    for f in TRIGGER_FEATURES:
        xt[:, f] = magnitude
    return xt
