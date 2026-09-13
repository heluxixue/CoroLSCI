# -*- coding: utf-8 -*-
"""fig1_pipeline_v4.py — Fig. 3 in the user's reference-pipeline style:
light-grey rounded panel, white main-chain cards, khaki detail cards below
each stage, thin arrows, grey registration-abandoned note strip.
Full-width figure (13 x 5.2 in, ~2.5:1 like the reference)."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

FIGDIR = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures"

INK, TXT, SUB, GREY = "#222222", "#333333", "#555555", "#8a8a8a"
PANEL = "#e0e0e0"      # light-grey board (reference background)
CARD = "#ffffff"       # main-chain cards
CARD_EC = "#c8c8c8"
DETAIL = "#e0e0c0"     # khaki detail cards
DETAIL_EC = "#cfcfa0"

W, H = 13.0, 5.2

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---- light-grey board ----
ax.add_patch(FancyBboxPatch((0.06, 0.06), W - 0.12, H - 0.12,
                            boxstyle="round,pad=0.05", fc=PANEL, ec="none"))
ax.text(W / 2, H - 0.32, "CoroLSCI measurement pipeline (frame-by-frame, on unsampled data)",
        fontsize=10.5, color=INK, ha="center", va="center", fontweight="bold")

# ---- main chain: five white cards ----
cards = [
    ("Raw .dat", "Pimsoft LSCI\n2641 frames @ 44 fps"),
    ("Decoding  (Eq. 1)", "four synchronous stacks\nI / V / C / P"),
    ("Session-adaptive\nnormalization + QC", "three candidate ranges\nscore $s$"),
    ("Segmentation", "ResUNet, 3.45 M\nDice 0.9207, 5.1 ms"),
    ("Dual indicators\n(Eqs. 2\u20136)", "$A$, $\\ell$, $\\bar d$ (mm), $n_{\\mathrm{br}}$;\n$P_{\\mathrm{mfr}}$ (a.u.)"),
]
n = len(cards)
x0, gap, cw = 0.30, 0.26, 2.32
cy, ch = 3.30, 1.42
for i, (title, sub) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch((x, cy), cw, ch, boxstyle="round,pad=0.06",
                                fc=CARD, ec=CARD_EC, lw=1.0))
    ax.text(x + cw / 2, cy + ch - 0.30, title, fontsize=8.6, color=INK,
            ha="center", va="top", fontweight="bold", linespacing=1.25)
    ax.text(x + cw / 2, cy + ch - 0.72, sub, fontsize=7.0, color=SUB,
            ha="center", va="top", linespacing=1.35)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + cw + 0.03, cy + ch / 2),
                                     (x + cw + gap - 0.03, cy + ch / 2),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#666666", lw=1.3))

# ---- detail cards below each stage (khaki) ----
details = [
    "header carries $\\beta$, $k$,\nmm/px calibration \u2192\nphysical units",
    "$C = \\beta\\sqrt{|V|}/I\\cdot\\mathrm{sgn}(V)$,\n$P = k(1/C - 1)$\ninstrument convention",
    "$s = \\mathrm{CV}(A)+0.02\\,(n_{\\mathrm{comp}}-1)$;\n$<10$ s per session,\nscore archived",
    "topology-preserving loss;\n3 seeds; 196 frames/s\ninside 22.7-ms budget",
    "mask + stacks synchronous \u2192\naligned, unsampled;\nno registration stage",
]
dy, dh = 1.02, 1.70
for i, det in enumerate(details):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch((x, dy), cw, dh, boxstyle="round,pad=0.06",
                                fc=DETAIL, ec=DETAIL_EC, lw=1.0))
    ax.text(x + cw / 2, dy + dh / 2, det, fontsize=6.9, color=TXT,
            ha="center", va="center", linespacing=1.35)
    ax.add_patch(FancyArrowPatch((x + cw / 2, cy - 0.02),
                                 (x + cw / 2, dy + dh + 0.02),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="#999999", lw=0.9, ls=":"))

# ---- registration-abandoned strip ----
ax.add_patch(FancyBboxPatch((0.30, 0.26), W - 0.60, 0.54,
                            boxstyle="round,pad=0.04", fc="#f2f2f2", ec=GREY, lw=0.9))
ax.text(0.46, 0.53, "Registration (preliminary, abandoned \u2014 not in the indicator path):",
        fontsize=7.6, color=GREY, va="center", fontweight="bold")
ax.text(0.46, 0.24, "brightness matching sees perfusion flicker as motion (residual floor \u2248 6 px); "
        "warping would resample the measurand (areas, diameters, speckle statistics)",
        fontsize=6.8, color=GREY, va="center")

fig.savefig(os.path.join(FIGDIR, "fig1_pipeline.pdf"), bbox_inches="tight")
plt.close(fig)
print("saved fig1_pipeline.pdf (reference-style v4)")
