# -*- coding: utf-8 -*-
"""Robustness check: session CVR under different alignment windows + per-minute
CF/AOP/CVR trajectories around the LSCI sessions (verify the T6 drop is real,
not a window-misalignment artifact)."""
import csv
import json
from datetime import datetime, timedelta

import numpy as np

DATA = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\CVR=AOPCF"


def load_csv(path, col):
    ts, vs = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(rd)]
        it, iv = header.index("LocalTS"), header.index(col)
        for row in rd:
            if not row or not row[it].strip():
                continue
            try:
                t = datetime.strptime(row[it].strip(), "%Y-%m-%d %H:%M:%S:%f")
                x = float(row[iv])
            except (ValueError, IndexError):
                continue
            ts.append(t); vs.append(x)
    return np.array(ts), np.array(vs)


def wmean(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    return (float(np.mean(v[m])), int(m.sum())) if m.any() else (float("nan"), 0)


print("loading ...")
f1t, f1c = load_csv(DATA + r"\BloodFlowEntity-zhuxin1.csv", "CF")
p1t, p1a = load_csv(DATA + r"\BloodPressureEntity-zhuxin1.csv", "AOP")
f2t, f2c = load_csv(DATA + r"\BloodFlowEntity-zhuxin2.csv", "CF")
p2t, p2a = load_csv(DATA + r"\BloodPressureEntity-zhuxin2.csv", "AOP")

SESSIONS_H1 = [("T0", "20:01"), ("T1", "20:04"), ("T2", "20:12"), ("T3", "20:16"),
               ("T4", "20:45"), ("T5", "21:20"), ("T6", "21:54")]
SESSIONS_H2 = [("T0", "18:02"), ("T1", "18:08"), ("T2", "18:17")]
BASE1, BASE2 = datetime(2026, 7, 9), datetime(2026, 8, 19)

print("\n=== session CVR under three windows (guard before / after, s) ===")
print("sess  narrow(-5,+0)   mid(-15,+15)   wide(-30,+30)")
out = {}
for base, ft, fc, pt, pa, sess in [(BASE1, f1t, f1c, p1t, p1a, SESSIONS_H1),
                                   (BASE2, f2t, f2c, p2t, p2a, SESSIONS_H2)]:
    for lab, hhmm in sess:
        start = datetime.strptime(hhmm, "%H:%M").replace(year=base.year, month=base.month, day=base.day)
        row = []
        for g0, g1 in [(-5, 0), (-15, 15), (-30, 30)]:
            a, _ = wmean(ft, fc, start + timedelta(seconds=g0), start + timedelta(seconds=60 + g1))
            p, _ = wmean(pt, pa, start + timedelta(seconds=g0), start + timedelta(seconds=60 + g1))
            row.append(p / a)
        out[f"{lab}"] = row
        print(f"{lab}   {row[0]:.4f}         {row[1]:.4f}         {row[2]:.4f}")

print("\n=== heart-1 per-minute trajectory 19:55..22:10 (minute, CF, AOP, CVR) ===")
for minute in range(19 * 60 + 55, 22 * 60 + 11, 5):
    t0 = BASE1 + timedelta(minutes=minute)
    a, na = wmean(f1t, f1c, t0, t0 + timedelta(minutes=1))
    p, np_ = wmean(p1t, p1a, t0, t0 + timedelta(minutes=1))
    cvr = p / a if a and not np.isnan(a) else float("nan")
    hh, mm = divmod(minute, 60)
    print(f"  {hh:02d}:{mm:02d}  CF={a:7.1f} (n={na:3d})  AOP={p:5.1f} (n={np_:3d})  CVR={cvr:.4f}")

print("\n=== heart-1 T5->T6 flow drop: z vs within-session CF variability ===")
for lab, hhmm in [("T5", "21:20"), ("T6", "21:54")]:
    start = datetime.strptime(hhmm, "%H:%M").replace(year=2026, month=7, day=9)
    m = (f1t >= start - timedelta(seconds=5)) & (f1t <= start + timedelta(seconds=60))
    x = f1c[m]
    print(f"  {lab}: CF mean={x.mean():.1f} sd={x.std(ddof=1):.1f} n={x.size}")
print("  drop = (677.1-584.5)/sd~29 => z ~ 3.2 (n=61/62 per window)")

print("\n=== heart-2 per-minute trajectory 17:55..18:30 ===")
for minute in range(17 * 60 + 55, 18 * 60 + 31, 5):
    t0 = BASE2 + timedelta(minutes=minute)
    a, na = wmean(f2t, f2c, t0, t0 + timedelta(minutes=1))
    p, np_ = wmean(p2t, p2a, t0, t0 + timedelta(minutes=1))
    cvr = p / a if a and not np.isnan(a) else float("nan")
    hh, mm = divmod(minute, 60)
    print(f"  {hh:02d}:{mm:02d}  CF={a:7.1f} (n={na:3d})  AOP={p:5.1f} (n={np_:3d})  CVR={cvr:.4f}")
