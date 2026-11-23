"""Examiner-style HTML report for the simulated clinical-deployment case."""
import html


def _e(x):
    return html.escape(str(x))


def _rows(pairs):
    return "".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in pairs)


def build_report(case):
    d, s, prov, ev = case["diff"], case["single"], case["provenance"], case["evaluation"]
    findings = "".join(
        f"<li><b>{_e(f['rule'])}</b> at {_e(f['ts'])}: {_e(f['detail'])}</li>"
        for f in case["log_findings"]) or "<li>none</li>"
    sweep = "".join(
        f"<tr><td>{k}</td><td>{v['asr_mean']:.2f}</td><td>{v['diff_detect_on_ref']:.2f}</td>"
        f"<td>{v['diff_detect_on_heavy']:.2f}</td><td>{v['single_detect']:.2f}</td></tr>"
        for k, v in ev["by_scale"].items())
    expl = case["explanation"]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Lazarus-style backdoor detection: simulated case report</title>
<style>
body{{font-family:'IBM Plex Sans',sans-serif;background:#F2F4F0;color:#132436;max-width:860px;margin:40px auto;padding:0 24px;line-height:1.6}}
h1,h2{{font-family:'IBM Plex Mono',monospace}} h1{{font-size:26px}} h2{{font-size:18px;margin-top:34px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:10px 0}}
th,td{{border:1px solid #E7EBE4;padding:7px 10px;text-align:left;font-size:14.5px}}
th{{background:#E7EBE4;font-family:'IBM Plex Mono',monospace;font-weight:600}}
.bad{{color:#B4483C;font-weight:600}} .ok{{color:#3E8F63;font-weight:600}}
.note{{background:#fff;border-left:3px solid #3E7C8C;padding:10px 14px;color:#3C4E5C}}
</style></head><body>
<h1>Backdoor detection in a model checkpoint: simulated case</h1>
<p class="note">Everything here is synthetic: a toy 12-64-32-1 network, synthetic patients, a simulated deployment log.
The edit is a test fixture so the detectors have a known target. No real system or data is involved.</p>

<h2>1. Summary of conclusions</h2>
<ul>
<li>The deployed checkpoint does <b class="bad">not</b> match the approved registry entry ({_e(prov['reason'])}).</li>
<li>Reference diff: verdict <b class="bad">{_e(d['verdict'])}</b>; {d['touched_fraction']*100:.1f}% of units changed, top-3 units carry {d['top_k_concentration']*100:.1f}% of the change ({_e(', '.join(d['flagged_units']))}).</li>
<li>Single-checkpoint screen (no reference): <b>{_e(s['verdict'])}</b>{(' (' + _e(', '.join(s['flagged_units'])) + ')') if s['flagged_units'] else ''}.</li>
<li>Held-out accuracy is unchanged by the edit ({case['clean_acc']:.3f} vs {case['bad_acc']:.3f}), so accuracy monitoring would not have caught it.
With the trigger present, {case['asr']*100:.0f}% of high-risk patients the model would normally escalate are not escalated.</li>
</ul>

<h2>2. Evidence handling</h2>
<table>{_rows([("Approved checkpoint sha256", f"<code>{_e(case['sha_approved'][:24])}…</code>"),
 ("Deployed checkpoint sha256", f"<code>{_e(case['sha_deployed'][:24])}…</code>")])}</table>

<h2>3. Deployment log findings (simulated log)</h2><ul>{findings}</ul>

<h2>4. Weight-level findings</h2>
<table>{_rows([("Units changed vs. approved", f"{d['touched_fraction']*100:.1f}% of {d['units_total']}"),
 ("Largest single-unit change", f"{d['max_unit_delta']:.2f}"),
 ("Median unit change", f"{d['median_unit_delta']:.4f}"),
 ("Trigger hypothesis (layer-1 unit)", f"driven by features {_e(expl['driving_features'])}; fires on {expl['clean_fire_rate']*100:.2f}% of clean rows; {_e(expl['note'])}")])}</table>

<h2>5. How reliable is this? Measured across {ev['n_models']} independently trained models</h2>
<p>Honest-model false positives: reference diff {ev['diff_legit_finetune_flagged']*100:.0f}% after a light update,
{ev['diff_heavy_retrain_flagged']*100:.0f}% after a heavy retrain; single-checkpoint screen {ev['single_honest_flagged']*100:.0f}%.</p>
<table><tr><th>edit scale</th><th>attack success</th><th>diff detects (light update)</th><th>diff detects (heavy retrain)</th><th>single screen detects</th></tr>{sweep}</table>
<p class="note">Read the table the way an examiner would: the weakest edit (scale 1.0) already suppresses escalation in about
a third of triggered cases, is invisible to the single-checkpoint screen, and is caught by the reference diff only some of the time after a heavy honest retrain. The reference
diff is strong only when the approved checkpoint is kept and the update is not too different from it.</p>

<h2>6. What this does and does not show</h2>
<ul>
<li>It shows that sparse, large edits are separable from dense retraining when a reference exists.</li>
<li>It does not show detection of an attacker who knows the detector and spreads the edit across many units, or a trigger that reuses heavily trained units.</li>
<li>Toy model, synthetic data, one attack family. Results should not be read as performance on production models.</li>
</ul>
</body></html>"""
