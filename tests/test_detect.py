import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lazarus_detect.data import make_patients, add_trigger
from lazarus_detect.detect import diff_checkpoints, screen_single, explain_l1_unit
from lazarus_detect.fixtures import apply_neuron_edit, legit_finetune, attack_success_rate
from lazarus_detect.model import init_params, train, accuracy, save_checkpoint, load_checkpoint
from lazarus_detect.provenance import (make_registry, verify_against_registry, simulated_log,
                                       log_findings, sha256_file)


@pytest.fixture(scope="module")
def world():
    x, y = make_patients(5000, seed=11)
    xtr, ytr, xte, yte = x[:4000], y[:4000], x[4000:], y[4000:]
    ref = train(init_params(3), xtr, ytr, epochs=40, seed=3)
    xn, yn = make_patients(2000, seed=12)
    return dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, ref=ref, xn=xn, yn=yn)


def test_identical_checkpoints(world):
    assert diff_checkpoints(world["ref"], world["ref"])["verdict"] == "identical"


def test_honest_update_not_flagged(world):
    ft = legit_finetune(world["ref"], world["xn"], world["yn"], seed=1)
    assert diff_checkpoints(world["ref"], ft)["verdict"] == "dense update"


def test_edit_flagged_and_localized(world):
    bad, info = apply_neuron_edit(world["ref"], world["xtr"], scale=3.0)
    d = diff_checkpoints(world["ref"], bad)
    assert d["verdict"] == "SPARSE EDIT"
    assert f"L1[{info['l1_unit']}]" in d["flagged_units"]
    assert f"L2[{info['l2_unit']}]" in d["flagged_units"]
    assert d["touched_fraction"] < 0.1


def test_edit_does_not_change_clean_accuracy(world):
    bad, _ = apply_neuron_edit(world["ref"], world["xtr"], scale=3.0)
    gap = abs(accuracy(world["ref"], world["xte"], world["yte"]) -
              accuracy(bad, world["xte"], world["yte"]))
    assert gap < 0.01  # why accuracy monitoring alone is not a defense


def test_edit_has_real_effect_when_triggered(world):
    bad, _ = apply_neuron_edit(world["ref"], world["xtr"], scale=3.0)
    asr = attack_success_rate(bad, world["xte"], world["yte"], add_trigger)
    assert asr > 0.9
    assert attack_success_rate(world["ref"], world["xte"], world["yte"], add_trigger) < 0.2


def test_single_screen_finds_loud_edit_and_spares_honest_model(world):
    assert screen_single(world["ref"], world["xtr"])["verdict"] == "no finding"
    bad, info = apply_neuron_edit(world["ref"], world["xtr"], scale=4.0)
    s = screen_single(bad, world["xtr"])
    assert s["verdict"] == "SUSPICIOUS"
    ex = explain_l1_unit(bad, info["l1_unit"], world["xtr"])
    assert set(ex["driving_features"]) == {10, 11}
    assert ex["clean_fire_rate"] == 0.0


def test_registry_and_log_rules(world, tmp_path):
    good = str(tmp_path / "v1.npz")
    save_checkpoint(good, world["ref"], {"version": "v1"})
    reg = make_registry([{"version": "v1", "path": good, "approved_by": "r",
                          "approved_at": "2026-10-02T14:00:00Z"}])
    assert verify_against_registry(reg, "v1", good)["ok"]
    bad, _ = apply_neuron_edit(world["ref"], world["xtr"], scale=3.0)
    bad_path = str(tmp_path / "v1_deployed.npz")
    save_checkpoint(bad_path, bad, {"version": "v1"})
    assert not verify_against_registry(reg, "v1", bad_path)["ok"]
    rules = {f["rule"][:2] for f in log_findings(
        simulated_log(reg, tampered_sha256=sha256_file(bad_path)), reg)}
    assert rules == {"R1", "R2", "R3"}
    assert log_findings(simulated_log(reg), reg) == []


def test_checkpoint_roundtrip(world, tmp_path):
    path = str(tmp_path / "m.npz")
    save_checkpoint(path, world["ref"], {"a": 1})
    p, meta = load_checkpoint(path)
    assert meta == {"a": 1}
    assert all(np.array_equal(p[k], world["ref"][k]) for k in p)
