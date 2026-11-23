"""Detectors. Two modes, because deployments differ in what they can compare to.

Mode A, `diff_checkpoints(reference, candidate)`: you hold the approved
checkpoint. The question is whether the candidate differs from it the way an
honest retrain does (every weight moves a little) or the way an edit does (a few
units move a lot).

Mode B, `screen_single(checkpoint, x_clean)`: no reference exists. Look for units
that are dormant on clean data yet wired loudly to the output, with incoming
weights concentrated on a couple of inputs. This is weaker; the README reports
its measured false-positive and miss rates.
"""
import numpy as np

from .model import forward

# --------------------------------------------------------------------------
# Unit bookkeeping. Every weight belongs to exactly one unit, as an INCOMING
# weight: layer-1 unit j owns W1[:, j] and b1[j]; layer-2 unit k owns W2[:, k]
# and b2[k]; the output unit owns W3 and b3. (Counting outgoing weights too would
# double-count, and one edited column would look like it touched every unit
# feeding it.)
# --------------------------------------------------------------------------


def _unit_vectors(p):
    units = []
    for j in range(p["W1"].shape[1]):
        units.append(("L1", j, np.concatenate([p["W1"][:, j], [p["b1"][j]]])))
    for k in range(p["W2"].shape[1]):
        units.append(("L2", k, np.concatenate([p["W2"][:, k], [p["b2"][k]]])))
    units.append(("OUT", 0, np.concatenate([p["W3"].ravel(), p["b3"].ravel()])))
    return units


def diff_checkpoints(ref, cand, top_k=3, tol=1e-9):
    ud_ref, ud_cand = _unit_vectors(ref), _unit_vectors(cand)
    norms, labels = [], []
    for (la, ia, va), (_, _, vb) in zip(ud_ref, ud_cand):
        norms.append(float(np.linalg.norm(vb - va)))
        labels.append(f"{la}[{ia}]")
    norms = np.array(norms)
    n_units = len(norms)
    touched = norms > tol
    touched_frac = float(touched.mean())
    sq = norms ** 2
    total = sq.sum()
    top_idx = np.argsort(-norms)[:top_k]
    concentration = float(sq[top_idx].sum() / total) if total > 0 else 0.0

    # Which units stand out? Two regimes.
    flagged = []
    if total == 0:
        verdict = "identical"
    else:
        med = float(np.median(norms))
        if med <= tol:
            # Most units untouched: any touched unit is by definition an outlier.
            cutoff = tol
        else:
            mad = float(np.median(np.abs(norms - med)))
            cutoff = med + 8.0 * max(1.4826 * mad, 0.25 * med)
        flagged = [labels[i] for i in np.argsort(-norms) if norms[i] > cutoff][:8]
        sparse_edit = (touched_frac < 0.25 and concentration > 0.8) or \
                      (len(flagged) >= 1 and concentration > 0.5 and norms[top_idx[0]] > cutoff)
        verdict = "SPARSE EDIT" if sparse_edit else "dense update"
    return {
        "verdict": verdict,
        "units_total": n_units,
        "touched_fraction": touched_frac,
        "top_k_concentration": concentration,
        "flagged_units": flagged,
        "max_unit_delta": float(norms.max()),
        "median_unit_delta": float(np.median(norms)),
    }


# --------------------------------------------------------------------------
# Mode B: single-checkpoint screen
# --------------------------------------------------------------------------


def _robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return (v - med) / max(1.4826 * mad, 1e-9)


def screen_single(p, x_clean, z_cut=6.0, conc_cut=0.9):
    """Score every unit for 'dormant but loud'. Returns a dict with ranked units."""
    _, h1, h2 = forward(p, x_clean, return_hidden=True)
    act1, act2 = (h1 > 0).mean(0), (h2 > 0).mean(0)
    # Output leverage: how strongly a unit can move the final logit.
    lev2 = np.abs(p["W3"][:, 0])
    lev1 = np.abs(p["W2"]) @ np.abs(p["W3"][:, 0])
    # Incoming concentration: share of squared incoming mass on the top 2 inputs.
    def conc(W):
        sq = W ** 2
        top2 = np.sort(sq, axis=0)[-2:].sum(0)
        return top2 / np.maximum(sq.sum(0), 1e-12)
    rows = []
    for name, act, lev, W in (("L1", act1, lev1, p["W1"]), ("L2", act2, lev2, p["W2"])):
        # Leverage compared to that layer's other units, restricted to the dormant ones.
        zlev = _robust_z(lev)
        c = conc(W)
        for i in range(len(act)):
            rows.append({"unit": f"{name}[{i}]", "act_rate": float(act[i]),
                         "leverage_z": float(zlev[i]), "concentration": float(c[i])})
    for r in rows:
        r["dormant"] = r["act_rate"] < 0.01
        r["suspicious"] = bool(r["dormant"] and r["leverage_z"] > z_cut
                               and r["concentration"] > conc_cut)
    rows.sort(key=lambda r: (-r["suspicious"], -r["leverage_z"]))
    flagged = [r["unit"] for r in rows if r["suspicious"]]
    return {"verdict": "SUSPICIOUS" if flagged else "no finding",
            "flagged_units": flagged, "top_units": rows[:5]}


def explain_l1_unit(p, unit_index, x_clean, box=5.0):
    """For a flagged layer-1 unit, state the input pattern that excites it most
    (inside a +-box range) and how often clean data comes anywhere near it."""
    w, b = p["W1"][:, unit_index], p["b1"][unit_index]
    x_star = box * np.sign(w)
    pre_max = float(x_star @ w + b)
    fires_clean = float(((x_clean @ w + b) > 0).mean())
    top = np.argsort(-np.abs(w))[:3]
    return {
        "max_preactivation": pre_max,
        "driving_features": [int(i) for i in top if abs(w[i]) > 1e-9],
        "clean_fire_rate": fires_clean,
        "clean_max_abs_input": float(np.abs(x_clean).max()),
        "note": "pattern lies outside observed clean range" if pre_max > 0 and fires_clean == 0 else "",
    }
