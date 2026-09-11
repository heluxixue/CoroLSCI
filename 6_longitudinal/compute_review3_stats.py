# -*- coding: utf-8 -*-
"""compute_review3_stats.py -- round-3 review statistics (no GPU):

1) Unrounded session-mean P_mfr (T2..T6, every-20th sampling) -> settle 17.2/17.3
2) Invalid/saturated pixel fractions on sample frames of the T2 stack
3) Block-count sensitivity of the block-summary bootstrap decline CI
4) +/-1px mask perturbation propagated across ALL longitudinal sessions (T2..T6):
   endpoint changes of area and P_mfr under erosion/dilation
"""
import csv
import json
import os

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

SRC = r"D:\Project\CC_Project\激光散斑pure"
EXE = os.path.join(SRC, "DAT解析结果", "第三版执行")
T2 = os.path.join(SRC, "DAT解析结果", "整段数据_2641帧")
NEW = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
OUT = os.path.join(SRC, "review3_stats.json")

SESS = [  # tag, data dir, mask dir or None for T2
    ("T2", T2, None),
    ("T3", os.path.join(NEW, "zhuxin1_4_20260709-7use"), os.path.join(NEW, "zhuxin1_4_20260709-7use")),
    ("T4", os.path.join(NEW, "zhuxin1_5_20260709-10"), os.path.join(NEW, "zhuxin1_5_20260709-10")),
    ("T5", os.path.join(NEW, "zhuxin1_6_20260709-12"), os.path.join(NEW, "zhuxin1_6_20260709-12")),
    ("T6", os.path.join(NEW, "zhuxin1_7_20260709-13"), os.path.join(NEW, "zhuxin1_7_20260709-13")),
]

T2_MASK_DIRS = [os.path.join(EXE, f"segmentation_{s}", "masks_raw")
                for s in ("0-10s", "10-20s", "20-30s", "后30s")]


def mask_files(tag, d, n):
    if tag == "T2":
        fs = []
        for dd in T2_MASK_DIRS:
            fs += sorted(os.path.join(dd, f) for f in os.listdir(dd) if f.endswith(".npy"))
        return fs
    return [os.path.join(d, "masks", f"{t:05d}.npy") for t in range(n)]


def header(d):
    p = os.path.join(d, "文件头信息.json")
    if not os.path.exists(p):
        p = os.path.join(os.path.dirname(d), "文件头信息.json")
    return json.load(open(p, encoding="utf-8"))


def series(tag):
    d, _ = SESS[[s[0] for s in SESS].index(tag)][1:3]
    h = header(d)
    gain, coh = h["signal_gain"], h["coherence_factor"]
    I = np.load(os.path.join(d, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(d, "方差_variance.npy"), mmap_mode="r")
    files = mask_files(tag, d, I.shape[0])
    n = I.shape[0]
    area_px, pmfr = [], []
    for t in range(0, n, 20):
        m = np.load(files[t]) > 0
        area_px.append(int(m.sum()))
        if m.any():
            Im = np.asarray(I[t], np.float64)[m].mean()
            Vm = np.asarray(V[t], np.float64)[m].mean()
            C = coh * np.sqrt(abs(Vm)) / max(Im, 1e-9)
            pmfr.append(gain * (1.0 / C - 1.0))
        else:
            pmfr.append(np.nan)
    return np.array(area_px), np.array(pmfr)


def main():
    res = {}

    # ---- 1) unrounded session means ----
    unrounded = {}
    series_cache = {}
    for tag, _, _ in SESS:
        a, p = series(tag)
        series_cache[tag] = (a, p)
        unrounded[tag] = {"area_px_mean": round(float(np.nanmean(a)), 2),
                          "pmfr_mean": round(float(np.nanmean(p)), 3),
                          "pmfr_sd": round(float(np.nanstd(p)), 2)}
    res["unrounded_session_means"] = unrounded
    t5, t6 = unrounded["T5"]["pmfr_mean"], unrounded["T6"]["pmfr_mean"]
    res["terminal_interval_pct_unrounded"] = round(100 * (t6 - t5) / t5, 3)
    res["t2_to_t6_decline_unrounded"] = round(
        100 * (t6 - unrounded["T2"]["pmfr_mean"]) / unrounded["T2"]["pmfr_mean"], 3)

    # ---- 2) invalid pixel fractions (T2 sample frames) ----
    I = np.load(os.path.join(T2, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(T2, "方差_variance.npy"), mmap_mode="r")
    inv = {}
    for t in (0, 660, 1320, 1980, 2640):
        im = np.asarray(I[t], np.float64)
        va = np.asarray(V[t], np.float64)
        nonfin_i = float(np.mean(~np.isfinite(im)) * 100)
        nonfin_v = float(np.mean(~np.isfinite(va)) * 100)
        sat_i = float(np.mean(im >= 65535) * 100)  # uint16 saturation
        inv[f"frame_{t}"] = {"nonfinite_I_pct": nonfin_i, "nonfinite_V_pct": nonfin_v,
                             "saturated_I_pct": sat_i}
    res["invalid_pixel_fractions_pct"] = inv

    # ---- 3) block-count sensitivity of the decline CI ----
    pmfr = {tag: series_cache[tag][1] for tag, _, _ in SESS}
    order = ["T2", "T3", "T4", "T5", "T6"]
    blk = {}
    rng = np.random.RandomState(2026)
    for nblk in (6, 8, 10, 12, 15):
        nb = 133 // nblk
        blocks = {t: np.array([np.nanmean(pmfr[t][b * nb:(b + 1) * nb])
                               for b in range(nblk)]) for t in order}
        decl = []
        for _ in range(4000):
            bm = [rng.choice(blocks[t], size=nblk, replace=True).mean() for t in order]
            decl.append(100 * (bm[0] - bm[-1]) / bm[0])
        blk[f"nblocks_{nblk}"] = {
            "point": round(100 * (blocks["T2"].mean() - blocks["T6"].mean()) / blocks["T2"].mean(), 1),
            "median": round(float(np.median(decl)), 1),
            "CI95": [round(float(np.percentile(decl, 2.5)), 1),
                     round(float(np.percentile(decl, 97.5)), 1)]}
    res["block_count_sensitivity"] = blk

    # ---- 4) +/-1px perturbation across all sessions ----
    RES = 0.20849377889155601
    pert = {}
    for tag, d, _ in SESS:
        a0, p0 = series_cache[tag]
        files = mask_files(tag, d, len(a0) * 20)
        for name, op in (("erode1px", binary_erosion), ("dilate1px", binary_dilation)):
            aa, pp = [], []
            for k, t in enumerate(range(0, len(a0) * 20, 20)):
                m0 = np.load(files[t]) > 0
                m = op(m0, iterations=1)
                if not m.any():
                    aa.append(np.nan); pp.append(np.nan); continue
                aa.append(int(m.sum()))
                h = header(d)
                I = np.load(os.path.join(d, "光强_intensity.npy"), mmap_mode="r")
                V = np.load(os.path.join(d, "方差_variance.npy"), mmap_mode="r")
                Im = np.asarray(I[t], np.float64)[m].mean()
                Vm = np.asarray(V[t], np.float64)[m].mean()
                C = h["coherence_factor"] * np.sqrt(abs(Vm)) / max(Im, 1e-9)
                pp.append(h["signal_gain"] * (1.0 / C - 1.0))
            pert[f"{tag}_{name}"] = {"area_mm2_mean": round(float(np.nanmean(aa)) * RES ** 2, 1),
                                     "pmfr_mean": round(float(np.nanmean(pp)), 1)}
    # endpoint changes
    for name in ("erode1px", "dilate1px"):
        t2a = pert[f"T2_{name}"]["area_mm2_mean"]; t6a = pert[f"T6_{name}"]["area_mm2_mean"]
        t2p = pert[f"T2_{name}"]["pmfr_mean"]; t6p = pert[f"T6_{name}"]["pmfr_mean"]
        pert[f"endpoint_{name}"] = {
            "area_delta_pct": round(100 * (t6a - t2a) / t2a, 2),
            "pmfr_decline_pct": round(100 * (t2p - t6p) / t2p, 2)}
    res["perturbation"] = pert

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
