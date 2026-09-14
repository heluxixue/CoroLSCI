# -*- coding: utf-8 -*-
"""figx_draft.py — Figure X (publication draft v3): longitudinal structure-function dissociation

Spec: all English, Times New Roman; panel captions centered BELOW panels;
legend of (B) inside bottom-right; (C) peak labels vertically separated.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT = r"D:\Project\CC_Project\激光散斑pure\FigureX_drafts"
os.makedirs(OUT, exist_ok=True)
RES_ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
FPS = 44.0
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"

min_h1 = np.array([0, 11, 15, 44, 79, 113], float)        # T0, T2..T6
perf = np.array([1027.8, 1033.9, 1010.6, 967.7, 975.0, 806.8])
area = np.array([546.1, 545.6, 542.3, 554.6, 551.0, 534.4])
anchor = perf[1], area[1]


def mfr_series(src):
    d = os.path.join(RES_ROOT, src)
    h = json.load(open(os.path.join(d, "文件头信息.json"), encoding="utf-8"))
    gain, coh = h["signal_gain"], h["coherence_factor"]
    I = np.load(os.path.join(d, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(d, "方差_variance.npy"), mmap_mode="r")
    n = I.shape[0]
    out = np.zeros(n)
    for t in range(n):
        m = np.load(os.path.join(d, "masks", f"{t:05d}.npy")) > 0
        if m.any():
            Im = np.asarray(I[t], np.float64)[m].mean()
            Vm = np.asarray(V[t], np.float64)[m].mean()
            C = coh * np.sqrt(abs(Vm)) / max(Im, 1e-9)
            out[t] = gain * (1.0 / C - 1.0)
        else:
            out[t] = np.nan
    return out


def fft_db(x, fps):
    x = x[np.isfinite(x)]
    t = np.arange(len(x))
    p = np.polyfit(t, x, 1)
    xd = x - np.polyval(p, t)
    w = np.hanning(len(xd))
    sp = np.abs(np.fft.rfft(xd * w)) ** 2
    fr = np.fft.rfftfreq(len(xd), 1.0 / fps)
    return fr, sp


print("computing per-frame P_mfr for T3/T6 ...")
ser_t3 = mfr_series("zhuxin1_4_20260709-7use")
ser_t6 = mfr_series("zhuxin1_7_20260709-13")

fig = plt.figure(figsize=(13.5, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.34)
fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.24)

# ---------- (A) ----------
axA = fig.add_subplot(gs[0, 0])
perf_pct = 100 * perf / anchor[0]
area_pct = 100 * area / anchor[1]
axA.plot(min_h1[2:], perf_pct[2:], "o-", color=BLUE, lw=2, ms=6,
         label="Perfusion $P_{\\mathrm{mfr}}$")
axA.plot(min_h1[2:], area_pct[2:], "s-", color=ORANGE, lw=2, ms=6,
         label="Vessel area $A$")
axA.plot(min_h1[:2], perf_pct[:2], "o", color=BLUE, lw=1.5, ms=6,
         mfc="white", label=None)
axA.plot(min_h1[:2], area_pct[:2], "s", color=ORANGE, lw=1.5, ms=6,
         mfc="white", label=None)
axA.annotate("$-22.0\\%$ (95% CI 19.7–24.1%)", (min_h1[5], perf_pct[5]),
             textcoords="offset points", xytext=(-160, -16), fontsize=9,
             color=BLUE, fontweight="bold")
k, b = np.polyfit(min_h1[1:], perf_pct[1:], 1)
xs = np.linspace(0, 113, 50)
axA.plot(xs, k * xs + b, "--", color=BLUE, alpha=0.4, lw=1)
axA.set_xlabel("Time from first session (min)", fontsize=10)
axA.set_ylabel("Relative to T2 = 100 (%)", fontsize=10)
axA.set_ylim(74, 106)
axA.legend(fontsize=9, loc="lower left")
axA.grid(alpha=0.25, lw=0.5)
axA.tick_params(labelsize=9)

# ---------- (B) ----------
axB = fig.add_subplot(gs[0, 1])
from skimage.morphology import skeletonize
panels = [("T3 (+15 min)", "zhuxin1_4_20260709-7use"),
          ("T4 (+44 min)", "zhuxin1_5_20260709-10"),
          ("T6 (+113 min)", "zhuxin1_7_20260709-13")]
colors3 = ["#D95F02", "#66A61E", "#0072B2"]
for (tag, src), c in zip(panels, colors3):
    d = os.path.join(RES_ROOT, src)
    h = json.load(open(os.path.join(d, "文件头信息.json"), encoding="utf-8"))
    res = h["resolution_mm_per_pixel"]
    m = np.load(os.path.join(d, "masks", "01320.npy")) > 0
    sk = skeletonize(m)
    ys, xs_ = np.where(sk)
    axB.scatter(xs_ * res, -ys * res, s=0.6, color=c, alpha=0.7, label=tag)
axB.set_aspect("equal")
axB.set_xlabel("X (mm)", fontsize=10)
axB.set_ylabel("Y (mm)", fontsize=10)
axB.legend(fontsize=8, loc="lower right", markerscale=8, framealpha=0.9)
axB.grid(alpha=0.25, lw=0.5)
axB.tick_params(labelsize=9)

# ---------- (C) ----------
axC = fig.add_subplot(gs[0, 2])
fr3, sp3 = fft_db(ser_t3, FPS)
fr6, sp6 = fft_db(ser_t6, FPS)
sp3n = sp3 / sp3.max()
sp6n = sp6 / sp6.max()
band = fr3 <= 3.0
axC.plot(fr3[band], sp3n[band], color=BLUE, lw=1.6, label="T3 (+15 min)")
axC.plot(fr6[band], sp6n[band], color="#D95F02", lw=1.6, label="T6 (+113 min)")
pk3 = fr3[band][np.argmax(sp3n[band])]
pk6 = fr6[band][np.argmax(sp6n[band])]
axC.axvline(pk3, color=BLUE, ls=":", lw=1)
axC.axvline(pk6, color="#D95F02", ls=":", lw=1)
# 峰值只差 0.017 Hz: 标注垂直分层, 蓝(T3)在峰上方左侧, 橙(T6)在峰下方右侧
axC.annotate(f"T3: {pk3:.2f} Hz ({pk3*60:.0f} bpm)", (pk3, 0.97), fontsize=8.5,
             ha="right", va="top", color=BLUE, xytext=(-4, -2),
             textcoords="offset points")
axC.annotate(f"T6: {pk6:.2f} Hz", (pk6, 0.55), fontsize=8.5,
             ha="left", va="bottom", color="#D95F02", xytext=(5, 2),
             textcoords="offset points")
axC.set_xlabel("Frequency (Hz)", fontsize=10)
axC.set_ylabel("Normalized power", fontsize=10)
axC.set_ylim(0, 1.12)
axC.legend(fontsize=8, loc="upper right")
axC.grid(alpha=0.25, lw=0.5)
axC.tick_params(labelsize=9)

# ---------- captions: 用轴实际包围盒取中心, 严格居中 ----------
fig.canvas.draw()
for ax, txt in [(axA, "(A) Heart 1: perfusion declines while geometry holds"),
                (axB, "(B) Skeleton timelapse (T3/T4/T6)"),
                (axC, "(C) FFT: beat peak persists at T6")]:
    pos = ax.get_position()
    fig.text(pos.x0 + pos.width / 2, 0.045, txt,
             ha="center", va="bottom", fontsize=10)

fig.savefig(os.path.join(OUT, "figX_draft.png"), dpi=300)
fig.savefig(os.path.join(OUT, "figX_draft.pdf"))
fig.savefig(r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures\fig10_longitudinal.pdf")
np.save(os.path.join(OUT, "P_mfr_perframe_T3.npy"), ser_t3)
np.save(os.path.join(OUT, "P_mfr_perframe_T6.npy"), ser_t6)
print("saved ->", OUT)
print(f"FFT peaks: T3={pk3:.3f} Hz, T6={pk6:.3f} Hz")
