# Lazarus-style backdoor detection (defensive)

Detection pipeline for **neuron-editing backdoors** in model weights: an attacker changes a
handful of existing neurons in a trained checkpoint, with no retraining, so a rare trigger
suppresses a clinical "escalate" decision while normal accuracy stays unchanged.

This repo is defensive. The edit exists only as a test fixture (`lazarus_detect/fixtures.py`) on a
toy 12-64-32-1 numpy network trained on **synthetic** patients. No real data, systems or
published attack code are involved. The whole deployment scenario (checkpoint, registry, logs,
patients) is simulated.

## What it does

| Layer | Needs | Finds |
|---|---|---|
| Registry hash check + log rules (`provenance.py`) | approved registry | swapped or rewritten checkpoint (R1 hash mismatch, R2 non-deploy loader, R3 write after approval) |
| Reference diff (`diff_checkpoints`) | approved checkpoint | sparse, large edits vs. dense small retraining; names the edited units |
| Single-checkpoint screen (`screen_single`) | nothing but the checkpoint | units that are dormant on clean data, loud at the output, with incoming weights concentrated on two inputs |
| Trigger hypothesis (`explain_l1_unit`) | a flagged unit | which inputs drive it and whether clean data ever fires it |

Run `python run_demo.py` for the case report (`out/report.html`), `python -m pytest tests` for the tests.
Needs only numpy (pytest for tests).

## Measured results (30 independently trained models, `out/results.json`)

Honest models flagged: reference diff 0% after a light update and 0% after a heavy retrain;
single-checkpoint screen 0%. Mean clean accuracy 0.70 (labels are noisy by design).

| edit scale | attack success* | diff detects (light update) | diff detects (heavy retrain) | single screen detects |
|---|---|---|---|---|
| 1.0 | 0.37 | 1.00 | 0.40 | 0.00 |
| 2.0 | 0.87 | 1.00 | 1.00 | 0.03 |
| 3.0 | 1.00 | 1.00 | 1.00 | 0.93 |
| 4.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 6.0 | 1.00 | 1.00 | 1.00 | 1.00 |

*share of high-risk patients the clean model escalates that are not escalated once the trigger is present.
Held-out accuracy drop was ~0 at every scale, so accuracy monitoring alone cannot catch this.

## Limits (read these)

- The single-checkpoint screen misses weak-but-effective edits (scale 1-2 still suppress escalation in 37-87% of triggered cases).
- A heavy honest retrain can hide a weak edit from the reference diff (40% detection at scale 1).
- Thresholds were set once and evaluated on one attack family; an attacker who knows the detector and spreads the edit over many units is not tested.
- Toy model, synthetic data. These numbers are not production performance.
- The registry check is the cheapest and strongest control, and it only works if the approved hash is stored somewhere the attacker cannot write.
