"""End-to-end demo: train, approve, tamper, detect, report.

Usage:  python run_demo.py            (uses out/results.json if present)
        python run_demo.py --eval     (re-runs the 30-model evaluation first)
"""
import json
import os
import sys

from lazarus_detect.data import make_patients, add_trigger
from lazarus_detect.detect import diff_checkpoints, screen_single, explain_l1_unit
from lazarus_detect.evaluate import run_evaluation
from lazarus_detect.fixtures import apply_neuron_edit, attack_success_rate
from lazarus_detect.model import init_params, train, accuracy, save_checkpoint, load_checkpoint
from lazarus_detect.provenance import (make_registry, verify_against_registry, simulated_log,
                                       log_findings, sha256_file, dump)
from lazarus_detect.report import build_report

OUT = "out"


def main():
    os.makedirs(OUT, exist_ok=True)
    x, y = make_patients(5000, seed=42)
    xtr, ytr, xte, yte = x[:4000], y[:4000], x[4000:], y[4000:]

    # 1. Train and approve v1.
    good = train(init_params(7), xtr, ytr, epochs=60, seed=7)
    approved_path = f"{OUT}/model_v1.npz"
    save_checkpoint(approved_path, good, {"version": "v1"})
    registry = make_registry([{"version": "v1", "path": approved_path,
                               "approved_by": "dr.reviewer", "approved_at": "2026-10-02T14:00:00Z"}])
    dump(registry, f"{OUT}/registry.json")

    # 2. Simulated tampering: edit units in place, write over the deployed file.
    bad, info = apply_neuron_edit(good, xtr, scale=3.0)
    deployed_path = f"{OUT}/model_v1_deployed.npz"
    save_checkpoint(deployed_path, bad, {"version": "v1"})
    sha_deployed = sha256_file(deployed_path)

    # 3. Detect.
    prov = verify_against_registry(registry, "v1", deployed_path)
    ref, _ = load_checkpoint(approved_path)
    cand, _ = load_checkpoint(deployed_path)
    d = diff_checkpoints(ref, cand)
    s = screen_single(cand, xtr)
    expl = explain_l1_unit(cand, info["l1_unit"], xtr)
    logs = simulated_log(registry, tampered_sha256=sha_deployed)
    findings = log_findings(logs, registry)

    # 4. Evaluation across many models.
    res_path = f"{OUT}/results.json"
    if "--eval" in sys.argv or not os.path.exists(res_path):
        dump(run_evaluation(n_models=30), res_path)
    with open(res_path) as f:
        ev = json.load(f)

    case = {"diff": d, "single": s, "provenance": prov, "log_findings": findings,
            "explanation": expl, "evaluation": ev,
            "clean_acc": accuracy(good, xte, yte), "bad_acc": accuracy(bad, xte, yte),
            "asr": attack_success_rate(bad, xte, yte, add_trigger),
            "sha_approved": registry[0]["sha256"], "sha_deployed": sha_deployed}
    with open(f"{OUT}/report.html", "w") as f:
        f.write(build_report(case))
    dump(logs, f"{OUT}/deployment_log.json")
    print("registry check :", prov)
    print("reference diff :", d["verdict"], d["flagged_units"])
    print("single screen  :", s["verdict"], s["flagged_units"])
    print("log findings   :", [f["rule"] for f in findings])
    print(f"accuracy clean {case['clean_acc']:.3f} vs edited {case['bad_acc']:.3f}; "
          f"attack success {case['asr']:.2f}")
    print(f"wrote {OUT}/report.html")


if __name__ == "__main__":
    main()
