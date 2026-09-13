# -*- coding: utf-8 -*-
"""fig58_units_fix.py — regenerate fig5_geometry / fig8_geometry_heart2 with
physical units (mm, mm^2; branch count dimensionless), replacing the
pixel-scale series (review 4.3). Layout identical to paper_fig78_symbols.py /
paper_heart2_figs.py; only the unit conversion and tick labels change.
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

SRC = r"D:\Project\CC_Project\激光散斑pure"
EXE = os.path.join(SRC, "DAT解析结果", "第三版执行")
FIGDIR = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures"
INK = "#0b0b0b"

SEG_H, GAP = 0.18, 0.08


def mm_series(rows, res):
    return {
        "Area (mm$^2$)": np.array([float(r["area"]) for r in rows]) * res ** 2,
        "Skeleton (mm)": np.array([float(r["skel_len"]) for r in rows]) * res,
        "Diameter (mm)": np.array([float(r["mean_d"]) for r in rows]) * res,
        "Branches": np.array([float(r["n_branch"]) for r in rows]),
    }


def draw(series, ts, img, xs, ys, img_title, panel_title, fname, x_max):
    ranges = {k: tuple(np.percentile(v, [5, 95])) for k, v in series.items()}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.05))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
    axes[0].scatter(xs, ys, s=0.5, c="lime", marker=".")
    axes[0].set_xlabel("X (col)", fontsize=10, labelpad=8)
    axes[0].text(0.5, -0.14, img_title, transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=12, color=INK)
    axes[0].set_ylabel("Y (row)", fontsize=10)

    def norm_seg(x, i, xmin, xmax):
        s = i * (SEG_H + GAP)
        return s + (x - xmin) / (xmax - xmin + 1e-9) * SEG_H

    order = ["Branches", "Diameter (mm)", "Skeleton (mm)", "Area (mm$^2$)"]
    symbols = {"Area (mm$^2$)": "$A(t)$  (mm$^2$)",
               "Skeleton (mm)": r"$\ell(t)$  (mm)",
               "Diameter (mm)": r"$\bar d(t)$  (mm)",
               "Branches": r"$n_{\mathrm{br}}(t)$"}
    color_map = {k: f"C{i}" for i, k in enumerate(
        ["Area (mm$^2$)", "Skeleton (mm)", "Diameter (mm)", "Branches"])}
    for i, name in enumerate(order):
        cc = color_map[name]
        x_min, x_max_ = ranges[name]
        yn = norm_seg(series[name], i, x_min, x_max_)
        axes[1].plot(ts, yn, lw=1.5, color=cc, label=symbols[name])
        if i > 0:
            axes[1].axhline(i * (SEG_H + GAP) - GAP / 2, color="gray",
                            lw=0.8, alpha=0.6)
        s0 = i * (SEG_H + GAP)
        for tv, tp in zip([x_min, (x_min + x_max_) / 2, x_max_],
                          [s0, s0 + SEG_H / 2, s0 + SEG_H]):
            axes[1].text(-0.02, tp, f"{tv:.5g}",
                         transform=axes[1].get_yaxis_transform(),
                         fontsize=8, color=cc, ha="right", va="center")
    axes[1].set_xlabel("Time (s)", fontsize=11, labelpad=6)
    axes[1].text(0.5, -0.14, panel_title, transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=12, color=INK)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(-0.02, 4 * (SEG_H + GAP) - GAP)
    axes[1].set_yticks([])
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


def main():
    # ---- Heart 1 / T2 (development session), res from header ----
    h = json.load(open(os.path.join(SRC, "DAT解析结果", "文件头信息.json"),
                       encoding="utf-8"))
    res1 = h.get("resolution_mm_per_pixel", 0.20849377889155601)
    print("T2 res:", res1)
    rows = list(csv.DictReader(open(
        os.path.join(EXE, "60s统一结果", "冠脉分析_指标_原始灌注.csv"),
        encoding="utf-8-sig")))
    ts = np.arange(len(rows)) / 44.0
    s1 = mm_series(rows, res1)
    print("T2 mm ranges:", {k: (round(v.min(), 1), round(v.max(), 1))
                            for k, v in s1.items()})

    # frame 2640 overlay image (same source as original figure)
    I = np.load(os.path.join(SRC, "DAT解析结果", "整段数据_2641帧",
                             "光强_intensity.npy"), mmap_mode="r")
    t = 2640
    f = np.asarray(I[t], np.float64)
    fin = np.isfinite(f)
    lo, hi = np.percentile(f[fin], [1, 99])
    img = np.clip((f - lo) / (hi - lo), 0, 1)
    mask = np.load(os.path.join(EXE, "segmentation_后30s", "masks_raw",
                                "01320.npy")) > 0
    edge = mask & ~binary_erosion(mask)
    ys, xs = np.where(edge)
    draw(s1, ts, img, xs, ys, "Intensity + Mask  (frame 2640)",
         "Coronary Structural Metrics (mm units)", "fig5_geometry.pdf", 60)

    # ---- Heart 2 / T0 first 5 s ----
    ROOT2 = r"D:\Project\CC_Project\激光散斑数据2\newdata"
    h2 = json.load(open(os.path.join(ROOT2, "文件头信息.json"), encoding="utf-8"))
    res2 = h2.get("resolution_mm_per_pixel")
    if not res2:
        # T0 header lacks the field; scale the T1 calibration (0.22402 px/mm @ 303 mm)
        # by distance ratio 307/303
        res2 = 0.224024484302087 * 307.0 / 303.0
        print("heart2 res (distance-scaled from T1):", res2)
    else:
        print("heart2 res:", res2)
    rows2 = list(csv.DictReader(open(
        os.path.join(ROOT2, "冠脉分析_指标_原始灌注_前5s.csv"), encoding="utf-8-sig")))
    ts2 = np.arange(len(rows2)) / 44.0
    s2 = mm_series(rows2, res2)
    print("heart2 mm ranges:", {k: (round(v.min(), 1), round(v.max(), 1))
                                for k, v in s2.items()})
    I2 = np.load(os.path.join(ROOT2, "整段数据_2641帧", "光强_intensity.npy"),
                 mmap_mode="r")
    f2 = np.asarray(I2[219], np.float64)
    fin2 = np.isfinite(f2)
    lo2, hi2 = np.percentile(f2[fin2], [1, 99])
    img2 = np.clip((f2 - lo2) / (hi2 - lo2), 0, 1)
    mask2 = np.load(os.path.join(ROOT2, "segmentation_前5s", "masks_raw",
                                 "00219.npy")) > 0
    edge2 = mask2 & ~binary_erosion(mask2)
    ys2, xs2 = np.where(edge2)
    draw(s2, ts2, img2, xs2, ys2, "Intensity + Mask  (heart 2, frame 219)",
         "Coronary Structural Metrics (heart 2, mm units)",
         "fig8_geometry_heart2.pdf", 5)


if __name__ == "__main__":
    main()
