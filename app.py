# Sepsis triage API - runs on IBM LinuxONE (s390x). Hospital/clinic systems POST readings, get risk + next step.
import json, platform, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

with open('model_meta.json') as f:
    meta = json.load(f)
FEAT, LABS, VITALS = meta['feature_cols'], meta['lab_cols'], meta['vital_cols']
EMB, T1COLS = meta['emb_cols'], meta['tier1_cols']
MU = np.array([meta['mu'][c] for c in FEAT])
SD = np.array([meta['sd'][c] for c in FEAT])
TIER1 = lgb.Booster(model_file='tier1_lgb.txt')
TIER2 = lgb.Booster(model_file='tier2_lgb.txt')
ENC = np.load('ae_encoder.npz')          # autoencoder encoder weights (no PyTorch needed at serving time)

def embed(x):
    z = np.nan_to_num((x - MU) / SD, nan=0.0)
    h = np.maximum(z @ ENC['W1'].T + ENC['b1'], 0)
    return np.maximum(h @ ENC['W2'].T + ENC['b2'], 0)

CHECKS = [('HR',    lambda v: v > 90,           'heart rate {:.0f} bpm (above 90)'),
          ('Resp',  lambda v: v >= 22,          'breathing rate {:.0f}/min (22 or more)'),
          ('Temp',  lambda v: v > 38 or v < 36, 'temperature {:.1f} C (outside 36-38)'),
          ('SBP',   lambda v: v <= 100,         'systolic BP {:.0f} mmHg (100 or less)'),
          ('MAP',   lambda v: v < 65,           'mean arterial pressure {:.0f} mmHg (below 65)'),
          ('O2Sat', lambda v: v < 92,           'oxygen saturation {:.0f}% (below 92)')]

def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

app = Flask(__name__)

@app.get('/health')
def health():
    return jsonify(status='ok', platform=platform.machine(), models=['tier1_hospital', 'tier2_clinic'])

@app.post('/triage')
def triage():
    t0 = time.perf_counter()
    data = request.get_json(force=True) or {}
    r = {c: to_float(data.get(c)) for c in FEAT}
    if np.isnan(r['MAP']) and not np.isnan(r['SBP']) and not np.isnan(r['DBP']):
        r['MAP'] = (r['SBP'] + 2 * r['DBP']) / 3          # calculated, never guessed
    row = pd.DataFrame([r], columns=FEAT, dtype=float)

    if row[LABS].notna().any(axis=1).iloc[0]:
        feats = pd.concat([row, pd.DataFrame(embed(row.values), columns=EMB)], axis=1)
        tier, score, thr = 'tier1_hospital', float(TIER1.predict(feats[T1COLS])[0]), meta['thr1']
    else:
        tier, score, thr = 'tier2_clinic', float(TIER2.predict(row[VITALS])[0]), meta['thr2']
    alert = score >= thr
    flags = [t.format(r[c]) for c, bad, t in CHECKS if not np.isnan(r[c]) and bad(r[c])]
    if alert:
        step = 'Sepsis alert - request urgent review by a doctor.'
    elif flags:
        step = 'No alert - continue routine monitoring and re-check vitals within the hour.'
    else:
        step = 'No alert - continue routine monitoring.'
    return jsonify(tier=tier, risk_score=round(score, 4), alert_threshold=round(thr, 4), alert=bool(alert),
                   abnormal_vitals=flags, next_step=step, platform=platform.machine(),
                   latency_ms=round((time.perf_counter() - t0) * 1000, 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)
