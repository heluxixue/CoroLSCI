# -*- coding: utf-8 -*-
"""fig69_mfr_fix.py — regenerate fig6/fig9 with manufacturer-convention P_mfr(t)

Replaces the per-pixel-mean perfusion series (negative values, wrong quantity)
with the mask-averaged manufacturer-convention perfusion surrogate of Eq. (2):
    Cbar_t = coh * sqrt(|mean_V_in_mask|) / mean_I_in_mask
    P_mfr(t) = gain * (1/Cbar_t - 1)
Layout/style identical to paper_fig78_symbols.py (fig6) / paper_heart2_figs.py (fig9).
Outputs overwrite the two figure PDFs in the EN_clean figures folder.
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

GAIN, COH = 83.87705993652344, 1.3439137935638428
INK = "#0b0b0b"
RED = "#e34948"
FIGDIR = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures"


def p_mfr_series(I, V, mask_files):
    out = np.zeros(len(mask_files))
    for t, mf in enumerate(mask_files):
        m = np.load(mf) > 0
        if m.any():
            Im = np.asarray(I[t], np.float64)[m].mean()
            Vm = np.asarray(V[t], np.float64)[m].mean()
            C = COH * np.sqrt(abs(Vm)) / max(Im, 1e-9)
            out[t] = GAIN * (1.0 / C - 1.0)
        else:
            out[t] = np.nan
    return out


def render(I_stack, P_stack, mask_files, t_show, t_end, sub, fname,
           x_max, caption, check):
    ts = np.arange(t_end) / 44.0
    print(f"computing P_mfr series ({t_end} frames) ...")
    perf = p_mfr_series(I_stack, V_STACK, mask_files[:t_end])
    ok = check(perf)
    print(f"  mean={np.nanmean(perf):.1f}  sd={np.nanstd(perf):.1f}  "
          f"range=[{np.nanmin(perf):.0f},{np.nanmax(perf):.0f}]  check={'OK' if ok else 'MISMATCH'}")

    mask = np.load(mask_files[t_show]) > 0
    edge = mask & ~binary_erosion(mask)
    ys, xs = np.where(edge)
    img_P = np.asarray(P_stack[t_show], dtype=np.float32)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14.5, 5.05))
    im = axes2[0].imshow(img_P, cmap="turbo", vmin=0, vmax=1600, origin="upper")
    axes2[0].scatter(xs, ys, s=0.5, c="lime", marker=".")
    axes2[0].set_xlabel("X (col)", fontsize=10, labelpad=8)
    axes2[0].text(0.5, -0.14, f"Perfusion + Mask{sub} (frame {t_show})",
                  transform=axes2[0].transAxes, ha="center", va="top",
                  fontsize=12, color=INK)
    axes2[0].set_ylabel("Y (row)", fontsize=10)
    fig2.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04)
    axes2[1].plot(ts, perf, lw=1.5, color=RED)
    axes2[1].set_xlabel("Time (s)", fontsize=11, labelpad=6)
    axes2[1].text(0.5, -0.14, caption,
                  transform=axes2[1].transAxes, ha="center", va="top",
                  fontsize=12, color=INK)
    axes2[1].set_ylabel(r"$P_{\mathrm{mfr}}(t)$ — perfusion surrogate (a.u.)",
                        fontsize=11)
    axes2[1].set_xlim(0, x_max)
    axes2[1].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig2)
    print("saved", fname)
    return ok


# ---------------- Heart 1 / session T2 (development session) ----------------
SRC1 = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果"
SEG = os.path.join(SRC1, "第三版执行")
mask_dirs = [os.path.join(SEG, f"segmentation_{s}", "masks_raw")
             for s in ("0-10s", "10-20s", "20-30s", "后30s")]
files_h1 = sorted(sum([glob.glob(os.path.join(d, "*.npy")) for d in mask_dirs], []))
print(f"heart-1 masks: {len(files_h1)}")

I1 = np.load(os.path.join(SRC1, "整段数据_2641帧", "光强_intensity.npy"), mmap_mode="r")
V1 = np.load(os.path.join(SRC1, "整段数据_2641帧", "方差_variance.npy"), mmap_mode="r")
P1 = np.load(os.path.join(SRC1, "整段数据_2641帧", "灌注_perfusion.npy"), mmap_mode="r")

V_STACK = V1
ok1 = render(I1, P1, files_h1, 2640, 2641, "",
             "fig6_perfusion.pdf", 60,
             r"Coronary Perfusion Surrogate $P_{\mathrm{mfr}}(t)$",
             lambda p: abs(np.nanmean(p) - 1033.9) < 15)

# ---------------- Heart 2 / session T0 (first 5 s, 220 frames) ----------------
SRC2 = r"D:\Project\CC_Project\激光散斑数据2\newdata"
files_h2 = sorted(glob.glob(os.path.join(SRC2, "segmentation_前5s", "masks_raw", "*.npy")))
print(f"heart-2 masks: {len(files_h2)}")

I2 = np.load(os.path.join(SRC2, "整段数据_2641帧", "光强_intensity.npy"), mmap_mode="r")
V2 = np.load(os.path.join(SRC2, "整段数据_2641帧", "方差_variance.npy"), mmap_mode="r")
P2 = np.load(os.path.join(SRC2, "整段数据_2641帧", "灌注_perfusion.npy"), mmap_mode="r")

V_STACK = V2
ok2 = render(I2, P2, files_h2, 219, 220, "  (heart 2)",
             "fig9_perfusion_heart2.pdf", 5,
             r"Coronary Perfusion Surrogate $P_{\mathrm{mfr}}(t)$ (heart 2)",
             lambda p: 950 < np.nanmean(p) < 1150)

print("DONE", "| heart1 check:", ok1, "| heart2 check:", ok2)
