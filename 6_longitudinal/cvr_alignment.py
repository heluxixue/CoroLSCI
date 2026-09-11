"""CVR = AOP/CF session alignment with LSCI acquisition windows.

Perfusion-system logs (BloodFlowEntity / BloodPressureEntity CSVs) are aligned
with the LSCI recording windows of Table I (manuscript.tex, tab:sessions).
CVR = mean AOP (mmHg) / mean CF (mL/min) over each 50-60 s imaging window.
"""
import csv
import json
import math
from datetime import datetime, timedelta

import numpy as np

DATA = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\CVR=AOPCF"
OUT = r"D:\Project\CC_Project\激光散斑pure\cvr_sessions.json"

# LSCI sessions: (heart, label, start wall-clock, duration s, P_mfr session mean from Table III)
SESSIONS = [
    (1, "T0", "2026-07-09 20:01", 60.0, 1027.8),
    (1, "T1", "2026-07-09 20:04", 60.0, 986.5),
    (1, "T2", "2026-07-09 20:12", 60.0, 1033.9),
    (1, "T3", "2026-07-09 20:16", 60.0, 1010.6),
    (1, "T4", "2026-07-09 20:45", 60.0, 967.7),
    (1, "T5", "2026-07-09 21:20", 60.0, 975.0),
    (1, "T6", "2026-07-09 21:54", 60.0, 806.8),
    (2, "T0", "2026-08-19 18:02", 60.0, 1053.0),
    (2, "T1", "2026-08-19 18:08", 50.2, 1029.7),
    (2, "T2", "2026-08-19 18:17", 52.5, 997.9),
]

PAD = 5.0  # margin around the imaging window, seconds (recording_time is minute-resolution)


def load_csv(path, cols):
    """Load LocalTS + requested numeric columns."""
    ts, vals = [], {c: [] for c in cols}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(rd)]
        idx = {c: header.index(c) for c in cols}
        it = header.index("LocalTS")
        for row in rd:
            if not row or not row[it].strip():
                continue
            try:
                t = datetime.strptime(row[it].strip(), "%Y-%m-%d %H:%M:%S:%f")
            except ValueError:
                continue
            try:
                v = [float(row[idx[c]]) for c in cols]
            except (ValueError, IndexError):
                continue
            ts.append(t)
            for c, x in zip(cols, v):
                vals[c].append(x)
    return np.array(ts), {c: np.array(v) for c, v in vals.items()}


def window_stats(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    x = v[m]
    if x.size == 0:
        return None
    return dict(n=int(x.size), mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                min=float(np.min(x)), max=float(np.max(x)),
                n_zero=int(np.sum(x == 0)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print("loading perfusion logs ...")
f1_t, f1_v = load_csv(DATA + r"\BloodFlowEntity-zhuxin1.csv", ["CF"])
p1_t, p1_v = load_csv(DATA + r"\BloodPressureEntity-zhuxin1.csv", ["AOP"])
f2_t, f2_v = load_csv(DATA + r"\BloodFlowEntity-zhuxin2.csv", ["CF"])
p2_t, p2_v = load_csv(DATA + r"\BloodPressureEntity-zhuxin2.csv", ["AOP"])
print(f"heart1: {len(f1_t)} flow rows {f1_t[0]}..{f1_t[-1]} | {len(p1_t)} pressure rows")
print(f"heart2: {len(f2_t)} flow rows {f2_t[0]}..{f2_t[-1]} | {len(p2_t)} pressure rows")

results = []
for heart, lab, start_s, dur, pmfr in SESSIONS:
    start = datetime.strptime(start_s, "%Y-%m-%d %H:%M")
    # minute-resolution recording_time -> widen by PAD, then also report the strict window
    t0, t1 = start - timedelta(seconds=PAD), start + timedelta(seconds=dur)
    ft, fv = (f1_t, f1_v["CF"]) if heart == 1 else (f2_t, f2_v["CF"])
    pt, pv = (p1_t, p1_v["AOP"]) if heart == 1 else (p2_t, p2_v["AOP"])
    fs = window_stats(ft, fv, t0, t1)
    ps = window_stats(pt, pv, t0, t1)
    rec = dict(heart=heart, session=lab, start=start_s, duration_s=dur,
               pmfr_mean=pmfr, flow=fs, aop=ps)
    if fs and ps and fs["mean"] > 0:
        rec["cvr"] = ps["mean"] / fs["mean"]
        # pooled-sample CVR series (pair each flow sample with nearest pressure is overkill;
        # report ratio of means and, as spread proxy, ratio using AOP/CF per-flow-sample interpolation)
    results.append(rec)
    f = f"CF n={fs['n']:4d} mean={fs['mean']:8.2f} sd={fs['sd']:7.2f} zero={fs['n_zero']}" if fs else "CF none"
    p = f"AOP n={ps['n']:4d} mean={ps['mean']:7.2f} sd={ps['sd']:6.2f} zero={ps['n_zero']}" if ps else "AOP none"
    cv = f"CVR={rec['cvr']:.4f}" if "cvr" in rec else "CVR=n/a"
    print(f"H{heart} {lab} {start_s}: {f} | {p} | {cv}")

# heart-1 longitudinal window T2..T6: correlation between CVR and P_mfr
h1 = [r for r in results if r["heart"] == 1 and r["session"] in ("T2", "T3", "T4", "T5", "T6") and "cvr" in r]
cv = np.array([r["cvr"] for r in h1])
pm = np.array([r["pmfr_mean"] for r in h1])
print("\nheart1 T2..T6 CVR:", np.round(cv, 4).tolist())
print("heart1 T2..T6 Pmfr:", pm.tolist())
d_cvr = 100 * (cv[-1] - cv[0]) / cv[0]
d_pm = 100 * (pm[-1] - pm[0]) / pm[0]
print(f"delta CVR T2->T6 = {d_cvr:+.1f}%   delta Pmfr = {d_pm:+.1f}%")
print(f"Pearson r = {pearson(cv, pm):.3f}   Spearman rho = {spearman(cv, pm):.3f}  (n={len(h1)})")

# linear regression slope of CVR vs Pmfr for reporting
A = np.vstack([pm, np.ones_like(pm)]).T
slope, intercept = np.linalg.lstsq(A, cv, rcond=None)[0]
r2 = pearson(cv, pm) ** 2
print(f"CVR = {slope:.5f}*Pmfr + {intercept:.2f}, R2={r2:.3f}")

summary = dict(sessions=results,
               heart1_T2_T6=dict(cvr=[float(x) for x in cv], pmfr=[float(x) for x in pm],
                                 delta_cvr_pct=float(d_cvr), delta_pmfr_pct=float(d_pm),
                                 pearson_r=pearson(cv, pm), spearman_rho=spearman(cv, pm),
                                 n=len(h1), slope=float(slope), intercept=float(intercept), r2=float(r2)))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nwrote", OUT)
