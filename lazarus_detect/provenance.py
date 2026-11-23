"""Provenance layer: registry hashes + deployment-log rules.

The cheapest defense against a swapped checkpoint is not ML at all. If the hash
of what is loaded does not match what was approved, stop. These rules cover that
and two related signals. All logs here are simulated.
"""
import hashlib
import json


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_registry(entries):
    """entries: list of {version, path, approved_by, approved_at}. Adds sha256."""
    reg = []
    for e in entries:
        reg.append({**e, "sha256": sha256_file(e["path"])})
    return reg


def verify_against_registry(registry, version, path):
    rec = next((r for r in registry if r["version"] == version), None)
    if rec is None:
        return {"ok": False, "reason": f"version {version} not in registry"}
    actual = sha256_file(path)
    if actual != rec["sha256"]:
        return {"ok": False, "reason": "sha256 differs from approved registry entry",
                "expected": rec["sha256"], "actual": actual}
    return {"ok": True}


# ISO-8601 UTC timestamps compare correctly as strings.
APPROVED_LOADERS = {"svc-model-deploy"}


def log_findings(log, registry):
    """Apply three rules to a list of deployment-log dicts. Returns findings."""
    findings = []
    approved = {r["version"]: r for r in registry}
    for ev in log:
        v = ev.get("model_version")
        rec = approved.get(v)
        if ev["event"] == "model_load":
            if rec is None or ev.get("sha256") != rec["sha256"]:
                findings.append({"rule": "R1 hash mismatch at load", "ts": ev["ts"],
                                 "detail": f"{v} loaded with sha256 not matching registry"})
            if ev.get("actor") not in APPROVED_LOADERS:
                findings.append({"rule": "R2 load by non-deploy account", "ts": ev["ts"],
                                 "detail": f"actor {ev.get('actor')} loaded {v}"})
        if ev["event"] == "file_write" and rec is not None and ev["ts"] > rec["approved_at"]:
            findings.append({"rule": "R3 checkpoint written after approval", "ts": ev["ts"],
                             "detail": f"{ev.get('path')} modified by {ev.get('actor')} "
                                       f"after {v} approved at {rec['approved_at']}"})
    return findings


def simulated_log(registry, tampered_sha256=None):
    """Build a plausible deployment log. If tampered_sha256 is given, the file is
    rewritten by a non-deploy account after approval and loaded with the new hash."""
    r = registry[-1]
    v = r["version"]
    log = [
        {"ts": "2026-10-02T14:00:00Z", "event": "model_approve", "model_version": v,
         "actor": "dr.reviewer", "sha256": r["sha256"]},
        {"ts": "2026-10-02T14:20:00Z", "event": "model_load", "model_version": v,
         "actor": "svc-model-deploy", "sha256": r["sha256"]},
    ]
    if tampered_sha256:
        log += [
            {"ts": "2026-10-09T02:41:00Z", "event": "file_write", "model_version": v,
             "actor": "svc-batch-etl", "path": r["path"]},
            {"ts": "2026-10-09T02:44:00Z", "event": "model_load", "model_version": v,
             "actor": "svc-batch-etl", "sha256": tampered_sha256},
        ]
    return log


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
