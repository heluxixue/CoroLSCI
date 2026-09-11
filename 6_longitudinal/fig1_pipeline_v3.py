# -*- coding: utf-8 -*-
"""fig1_pipeline_v3.py — column-width redesign of the pipeline figure (Fig. 3):
vertical main chain designed at final print size (3.5 in wide), real-data
side thumbnails, registration-abandoned note. Replaces the two-column-span
schematic whose text was unreadable at \\columnwidth (review round 3, Sec. 6).
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import binary_erosion

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

SRC = r"D:\Project\CC_Project\激光散斑pure"
EXE = os.path.join(SRC, "DAT解析结果", "第三版执行")
FIGDIR = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures"
INK, MUTED, ACC, GREY = "#0b0b0b", "#555555", "#2a78d6", "#8a8a8a"
GAIN, COH = 83.87705993652344, 1.3439137935638428

W, H = 3.5, 7.9

# ---- data ----
I = np.load(os.path.join(SRC, "DAT解析结果", "整段数据_2641帧", "光强_intensity.npy"),
            mmap_mode="r")
V = np.load(os.path.join(SRC, "DAT解析结果", "整段数据_2641帧", "方差_variance.npy"),
            mmap_mode="r")
mask = np.load(os.path.join(EXE, "segmentation_后30s", "masks_raw", "01320.npy")) > 0
edge = mask & ~binary_erosion(mask)
f2640 = np.asarray(I[2640], np.float64)
fin = np.isfinite(f2640)
lo, hi = np.percentile(f2640[fin], [1, 99])
img_raw = np.clip((f2640 - lo) / (hi - lo), 0, 1)

rows = list(csv.DictReader(open(
    os.path.join(EXE, "60s统一结果", "冠脉分析_指标_原始灌注.csv"),
    encoding="utf-8-sig")))
res = 0.20849377889155601
area_ts = np.array([float(r["area"]) for r in rows[::20]]) * res ** 2

# P_mfr every 20th frame
seg_of = [("0-10s", 0, 440), ("10-20s", 440, 880),
          ("20-30s", 880, 1320), ("后30s", 1320, 2641)]
pm = []
for t in range(0, 2641, 20):
    for seg, l0, h0 in seg_of:
        if l0 <= t < h0:
            m = np.load(os.path.join(EXE, f"segmentation_{seg}", "masks_raw",
                                     f"{t:05d}.npy")) > 0
            break
    Im = np.asarray(I[t], np.float64)[m].mean()
    Vm = np.asarray(V[t], np.float64)[m].mean()
    C = COH * np.sqrt(abs(Vm)) / max(Im, 1e-9)
    pm.append(GAIN * (1.0 / C - 1.0))
pm = np.array(pm)
t_pm = np.arange(len(pm)) * 20 / 44.0

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BOX_X, BOX_W = 0.10, 2.05
STEPS = [
    ("1  Raw .dat", "Pimsoft LSCI\n2641 frames @ 44 fps"),
    ("2  Decoding  (Eq. 1)", "4 synchronous stacks\nI / V / C / P"),
    ("3  Session-adaptive\nnormalization + QC", "3 candidate ranges,\nscore $s$; < 10 s"),
    ("4  Segmentation", "ResUNet, 3.45 M\nDice 0.9207, 5.1 ms"),
    ("5  Dual indicators\n(Eqs. 2\u20136)", "$A,\\ \\ell,\\ \\bar d$ (mm), $n_{\\mathrm{br}}$;\n$P_{\\mathrm{mfr}}$ (a.u.); unsampled"),
]
BH, VGAP = 0.92, 0.42
y_top = H - 0.30
ys = [y_top - i * (BH + VGAP) for i in range(5)]
for (title, sub), yy in zip(STEPS, ys):
    ax.add_patch(FancyBboxPatch((BOX_X, yy - BH), BOX_W, BH,
                                boxstyle="round,pad=0.05", fc="#eef3fb",
                                ec=ACC, lw=1.0))
    ax.text(BOX_X + 0.10, yy - 0.30, title, fontsize=7.6, color=INK,
            fontweight="bold", va="center")
    ax.text(BOX_X + 0.10, yy - BH + 0.28, sub, fontsize=6.6, color=MUTED,
            va="center", linespacing=1.35)
for i in range(4):
    y0 = ys[i] - BH - 0.04
    ax.add_patch(FancyArrowPatch((BOX_X + BOX_W / 2, y0),
                                 (BOX_X + BOX_W / 2, y0 - VGAP + 0.08),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, lw=1.0))

# ---- side note: intrinsic alignment ----
ax.text(BOX_X + BOX_W + 0.10, ys[4] + BH / 2,
        "mask and stacks are\nsynchronous:\nintrinsically aligned,\n"
        "no registration\nin the indicator path",
        fontsize=6.2, color=ACC, va="center", ha="left", linespacing=1.4)

# ---- side thumbnails (right column) ----
TX, TW2 = 2.42, 0.95
# raw frame beside step 1
a1 = fig.add_axes([TX / W, (ys[0] - BH + 0.02) / H, TW2 / W, 0.88 / H])
a1.imshow(img_raw, cmap="gray", vmin=0, vmax=1, origin="upper")
ys_e, xs_e = np.where(edge)
a1.scatter(xs_e, ys_e, s=0.1, c="#2a78d6", marker=".", lw=0)
a1.set_xticks([]); a1.set_yticks([])
a1.set_title("frame 2640\n+ coronary mask", fontsize=5.8, color=MUTED, pad=2)
# area sparkline beside step 5
a2 = fig.add_axes([(TX + 0.12) / W, (ys[4] - BH + 0.10) / H, (TW2 - 0.10) / W, 0.70 / H])
a2.plot(t_pm, area_ts, lw=0.9, color="#E69F00")
a2.set_xticks([]); a2.set_yticks([])
a2.set_xlim(0, 60)
a2.set_title("$A(t)$ (mm$^2$)", fontsize=5.8, color=MUTED, pad=1.5)
for s_ in ("top", "right"):
    a2.spines[s_].set_visible(False)

# ---- bottom: registration abandoned note ----
NY = 0.18
ax.add_patch(FancyBboxPatch((0.10, NY), W - 0.22, 0.92,
                            boxstyle="round,pad=0.05", fc="#f4f4f4",
                            ec=GREY, lw=0.9))
ax.text(0.22, NY + 0.74,
        "Registration (preliminary, abandoned \u2014 not in the indicator path):",
        fontsize=6.4, color=GREY, fontweight="bold", va="center")
ax.text(0.22, NY + 0.38,
        "brightness matching sees perfusion flicker as motion\n"
        "(residual floor $\\approx$ 6 px); warping would resample the measurand\n"
        "(areas, diameters, speckle statistics)",
        fontsize=6.0, color=GREY, va="center", linespacing=1.4)

fig.savefig(os.path.join(FIGDIR, "fig1_pipeline.pdf"), bbox_inches="tight")
plt.close(fig)
print("saved fig1_pipeline.pdf (column-width v3)")
