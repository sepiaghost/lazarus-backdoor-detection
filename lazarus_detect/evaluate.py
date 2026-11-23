"""Measure the detectors across many independently trained models.

Every number in the README and the report comes from `run_evaluation`. Nothing is
hand-tuned per seed; thresholds in detect.py are fixed before this runs.
"""
import numpy as np

from .data import make_patients, add_trigger
from .detect import diff_checkpoints, screen_single
from .fixtures import apply_neuron_edit, legit_finetune, attack_success_rate
from .model import init_params, train, accuracy

SCALES = (1.0, 2.0, 3.0, 4.0, 6.0)


def run_evaluation(n_models=30, epochs=60, verbose=False):
    res = {
        "n_models": n_models,
        "clean_acc": [],
        "diff": {"identical": [], "legit_finetune": [], "heavy_retrain": [],
                 "edit_on_heavy": {s: [] for s in SCALES}, "edit_on_ref": {s: [] for s in SCALES},
                 "edit_on_finetune": {s: [] for s in SCALES}},
        "single": {"honest": [], "edited": {s: [] for s in SCALES}},
        "asr": {s: [] for s in SCALES},
        "acc_drop": {s: [] for s in SCALES},
    }
    for seed in range(n_models):
        x, y = make_patients(5000, seed=1000 + seed)
        xtr, ytr, xte, yte = x[:4000], y[:4000], x[4000:], y[4000:]
        ref = train(init_params(seed), xtr, ytr, epochs=epochs, seed=seed)
        res["clean_acc"].append(accuracy(ref, xte, yte))
        xn, yn = make_patients(2000, seed=5000 + seed)  # "new data" for honest updates
        ft = legit_finetune(ref, xn, yn, seed=seed)
        heavy = legit_finetune(ref, xn, yn, seed=seed, epochs=15, lr=0.005)  # large honest drift

        res["diff"]["identical"].append(diff_checkpoints(ref, ref)["verdict"])
        res["diff"]["legit_finetune"].append(diff_checkpoints(ref, ft)["verdict"])
        res["diff"]["heavy_retrain"].append(diff_checkpoints(ref, heavy)["verdict"])
        res["single"]["honest"].append(screen_single(ref, xtr)["verdict"])

        for s in SCALES:
            bad, _ = apply_neuron_edit(ref, xtr, scale=s)
            res["diff"]["edit_on_ref"][s].append(diff_checkpoints(ref, bad)["verdict"])
            res["single"]["edited"][s].append(screen_single(bad, xtr)["verdict"])
            res["asr"][s].append(attack_success_rate(bad, xte, yte, add_trigger))
            res["acc_drop"][s].append(res["clean_acc"][-1] - accuracy(bad, xte, yte))
            bad_hv, _ = apply_neuron_edit(heavy, xtr, scale=s)
            res["diff"]["edit_on_heavy"][s].append(diff_checkpoints(ref, bad_hv)["verdict"])
            bad_ft, _ = apply_neuron_edit(ft, xtr, scale=s)
            res["diff"]["edit_on_finetune"][s].append(diff_checkpoints(ref, bad_ft)["verdict"])
        if verbose:
            print(f"model {seed + 1}/{n_models} done", flush=True)
    return summarize(res)


def _rate(verdicts, positive):
    return float(np.mean([v == positive for v in verdicts]))


def summarize(res):
    out = {"n_models": res["n_models"],
           "clean_acc_mean": float(np.mean(res["clean_acc"])),
           "diff_identical_flagged": _rate(res["diff"]["identical"], "SPARSE EDIT"),
           "diff_legit_finetune_flagged": _rate(res["diff"]["legit_finetune"], "SPARSE EDIT"),
           "diff_heavy_retrain_flagged": _rate(res["diff"]["heavy_retrain"], "SPARSE EDIT"),
           "single_honest_flagged": _rate(res["single"]["honest"], "SUSPICIOUS"),
           "by_scale": {}}
    for s in SCALES:
        out["by_scale"][s] = {
            "asr_mean": float(np.mean(res["asr"][s])),
            "acc_drop_mean": float(np.mean(res["acc_drop"][s])),
            "diff_detect_on_ref": _rate(res["diff"]["edit_on_ref"][s], "SPARSE EDIT"),
            "diff_detect_on_finetune": _rate(res["diff"]["edit_on_finetune"][s], "SPARSE EDIT"),
            "diff_detect_on_heavy": _rate(res["diff"]["edit_on_heavy"][s], "SPARSE EDIT"),
            "single_detect": _rate(res["single"]["edited"][s], "SUSPICIOUS"),
        }
    return out
