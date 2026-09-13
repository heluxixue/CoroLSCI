# -*- coding: utf-8 -*-
"""compute_review4.py — round-4 review reanalyses (CPU), fixing R1-1/R1-2:

1) Per-session FULL-PRECISION calibration r_t from each recording's header,
   used for every pixel->physical area conversion (baseline, erode, dilate).
   Outputs a checkable provenance table (session, r_t, px area, mm2 area).
2) +-1px boundary perturbation endpoints recomputed with correct r_t
   (same stride-20 frame set for original/erode/dilate, same postprocessing).
3) Canonical block-summary bootstrap: 133 stride-20 samples per session
   (the sampling that generates Table 5), contiguous blocks with the
   remainder merged into the final block (identical total window for every
   block count), B=4000, seed 2026 -> reproducible decline CI.
4) Block-count sensitivity over exact divisors-based counts {6,12,22,33}
   with identical windows.

Output: review4_stats.json (+ console provenance table)
"""
import json
import os

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

SRC = r"D:\Project\CC_Project\激光散斑pure"
EXE = os.path.join(SRC, "DAT解析结果", "第三版执行")
T2DIR = os.path.join(SRC, "DAT解析结果", "整段数据_2641帧")
NEW = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
OUT = os.path.join(SRC, "review4_stats.json")

SESSIONS = [
    ("T2", T2DIR, None),
    ("T3", os.path.join(NEW, "zhuxin1_4_20260709-7use"),
     os.path.join(NEW, "zhuxin1_4_20260709-7use")),
    ("T4", os.path.join(NEW, "zhuxin1_5_20260709-10"),
     os.path.join(NEW, "zhuxin1_5_20260709-10")),
    ("T5", os.path.join(NEW, "zhuxin1_6_20260709-12"),
     os.path.join(NEW, "zhuxin1_6_20260709-12")),
    ("T6", os.path.join(NEW, "zhuxin1_7_20260709-13"),
     os.path.join(NEW, "zhuxin1_7_20260709-13")),
]
T2_MASK_DIRS = [os.path.join(EXE, f"segmentation_{s}", "masks_raw")
                for s in ("0-10s", "10-20s", "20-30s", "后30s")]
STRIDE = 20
B, SEED = 4000, 2026


def session_paths(tag):
    data_dir, _ = SESSIONS[[s[0] for s in SESSIONS].index(tag)][1:3]
    if tag == "T2":
        # T2's header JSON sits in the parent folder (DAT解析结果)
        h = json.load(open(os.path.join(os.path.dirname(data_dir),
                                        "文件头信息.json"), encoding="utf-8"))
        I = np.load(os.path.join(data_dir, "光强_intensity.npy"), mmap_mode="r")
        V = np.load(os.path.join(data_dir, "方差_variance.npy"), mmap_mode="r")
        files = []
        for d in T2_MASK_DIRS:
            files += sorted(os.path.join(d, f) for f in os.listdir(d)
                            if f.endswith(".npy"))
    else:
        h = json.load(open(os.path.join(data_dir, "文件头信息.json"),
                           encoding="utf-8"))
        I = np.load(os.path.join(data_dir, "光强_intensity.npy"), mmap_mode="r")
        V = np.load(os.path.join(data_dir, "方差_variance.npy"), mmap_mode="r")
        files = [os.path.join(data_dir, "masks", f"{t:05d}.npy")
                 for t in range(I.shape[0])]
    r_t = float(h["resolution_mm_per_pixel"])
    return r_t, I, V, files


def sample_frames(tag):
    """stride-20 frame indices + masks (identical set for all perturbations)"""
    _, _, _, files = session_paths(tag)
    idx = list(range(0, len(files), STRIDE))
    return idx, files


def area_pmfr(tag, op=None):
    """per-sample physical area (mm^2, session's own r_t) and P_mfr (a.u.)"""
    data_dir = session_paths_dir(tag)
    r_t, I, V, files = session_paths(tag)
    h = json.load(open(os.path.join(data_dir, "文件头信息.json")
                       if tag != "T2" else
                       os.path.join(os.path.dirname(data_dir), "文件头信息.json"),
                       encoding="utf-8"))
    gain, coh = h["signal_gain"], h["coherence_factor"]
    idx = list(range(0, len(files), STRIDE))
    areas, pmfr = [], []
    for t in idx:
        m0 = np.load(files[t]) > 0
        m = m0 if op is None else op(m0, iterations=1)
        if not m.any():
            areas.append(np.nan)
            pmfr.append(np.nan)
            continue
        areas.append(m.sum() * r_t ** 2)
        Im = np.asarray(I[t], np.float64)[m].mean()
        Vm = np.asarray(V[t], np.float64)[m].mean()
        C = coh * np.sqrt(abs(Vm)) / max(Im, 1e-9)
        pmfr.append(gain * (1.0 / C - 1.0))
    return np.array(areas), np.array(pmfr), r_t


def session_paths_dir(tag):
    return SESSIONS[[s[0] for s in SESSIONS].index(tag)][1]


def blocks_of(x, nblk):
    """contiguous blocks over the FULL sample vector; remainder samples are
    merged into the final block so every nblk spans the identical window"""
    n = len(x)
    base = n // nblk
    bounds = [i * base for i in range(nblk)] + [n]
    return np.array([np.nanmean(x[bounds[i]:bounds[i + 1]])
                     for i in range(nblk)])


def main():
    res = {"provenance": {}, "perturbation": {}, "bootstrap": {}}

    # ---------- 1) provenance + 2) perturbation with per-session r_t ----------
    series = {}
    for tag, _, _ in SESSIONS:
        a0, p0, r_t = area_pmfr(tag)
        series[tag] = (a0, p0)
        res["provenance"][tag] = {
            "r_mm_per_px": r_t,
            "n_samples_stride20": int(np.isfinite(a0).sum()),
            "area_mm2_mean": round(float(np.nanmean(a0)), 1),
            "pmfr_mean": round(float(np.nanmean(p0)), 1),
            "pmfr_sd": round(float(np.nanstd(p0)), 1),
        }
        print(f"{tag}: r={r_t:.6f}  area={np.nanmean(a0):.1f} mm2  "
              f"P_mfr={np.nanmean(p0):.1f}")

    for name, op in (("erode1px", binary_erosion),
                     ("dilate1px", binary_dilation)):
        sa = {}
        for tag, _, _ in SESSIONS:
            a, _, _ = area_pmfr(tag, op)
            sa[tag] = float(np.nanmean(a))
        # endpoint change: perturbed T6 vs perturbed T2 (same bias at both
        # ends -- the systematic-error scenario actually being tested)
        d = 100 * (sa["T6"] - sa["T2"]) / sa["T2"]
        res["perturbation"][name] = {
            "session_area_mm2": {k: round(v, 1) for k, v in sa.items()},
            "net_area_change_pct": round(d, 2),
        }
        print(name, "->", {k: round(v, 1) for k, v in sa.items()},
              "net change:", round(d, 2), "%")
    # unperturbed reference
    res["perturbation"]["baseline"] = {
        "net_area_change_pct": round(
            100 * (np.nanmean(series["T6"][0]) - np.nanmean(series["T2"][0]))
            / np.nanmean(series["T2"][0]), 2),
        "note": "stride-20 samples, per-session r_t; Table 4 uses full-frame "
                "means (545.6/534.4 -> -2.1%), this is the matching-sampling "
                "reference for the perturbation comparison",
    }

    # ---------- 3) canonical bootstrap (133 samples, remainder-in-last-block) ----------
    rng = np.random.RandomState(SEED)
    order = ["T2", "T3", "T4", "T5", "T6"]
    for nblk in (6, 12, 22, 33):
        bl = {t: blocks_of(series[t][1], nblk) for t in order}
        declines = np.zeros(B)
        for b_i in range(B):
            bm = [rng.choice(bl[t], size=nblk, replace=True).mean() for t in order]
            declines[b_i] = 100 * (bm[0] - bm[-1]) / bm[0]
        pt = 100 * (bl["T2"].mean() - bl["T6"].mean()) / bl["T2"].mean()
        res["bootstrap"][f"nblocks_{nblk}"] = {
            "point": round(pt, 1),
            "median": round(float(np.median(declines)), 1),
            "CI95": [round(float(np.percentile(declines, 2.5)), 1),
                     round(float(np.percentile(declines, 97.5)), 1)],
        }
        print(f"blocks={nblk}: point {pt:.1f}, CI "
              f"[{np.percentile(declines,2.5):.1f}, {np.percentile(declines,97.5):.1f}]")

    res["config"] = {"stride": STRIDE, "B": B, "seed": SEED,
                     "block_rule": "contiguous, remainder merged into final "
                                   "block; identical total window for all nblk",
                     "area_calibration": "per-session header "
                                         "resolution_mm_per_pixel (full precision)"}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
