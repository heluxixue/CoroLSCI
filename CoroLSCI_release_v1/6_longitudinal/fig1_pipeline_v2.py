# -*- coding: utf-8 -*-
"""fig1_pipeline_v2.py — regenerate fig1_pipeline.pdf to match the v4 manuscript

Chain: raw .dat -> decoding (Eq.1) -> session-adaptive normalization + QC
       -> segmentation (ResUNet, topology loss) -> dual indicators (Eq.2)
Side annotations:
  - mask + synchronous modality stacks -> intrinsically aligned, no registration
  - registration box (grey, rejected): preliminary, abandoned; residual floor ~6 px
Style: Times New Roman, INK/MUTED/ACC palette, same as original paper_figures.py.
"""
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

OUT = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, ACC = "#0b0b0b", "#555555", "#2a78d6"
GREY = "#8a8a8a"


def main():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec=ACC, fs=9, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=MUTED, lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=lw, ls=ls))

    # ---- main chain ----
    box(0.10, 2.75, 1.75, 1.25, "Raw .dat\n(Pimsoft LSCI)\n2641 fr, 44 fps")
    box(2.25, 2.75, 1.90, 1.25, "Decoding (Eq. 1)\n4 synchronous stacks\nI / V / C / P")
    box(4.55, 2.75, 2.10, 1.25,
        "Session-adaptive\nnormalization + QC\n(3 candidates, score s)")
    box(7.05, 2.75, 1.95, 1.25,
        "Segmentation\nResUNet (3.45 M)\ntopology-preserving loss")
    box(9.40, 2.75, 2.35, 1.25,
        "Dual indicators (Eq. 2)\nA, $\\ell$, $\\bar d$, $n_{\\mathrm{br}}$  |  $P_{\\mathrm{mfr}}(t)$\nmm units; frame by frame, unsampled")
    for x in (1.85, 4.15, 6.65, 9.00):
        arrow(x + 0.28, 3.37, x + 0.62, 3.37)

    # ---- top side: mask + synchronous stacks -> alignment without registration
    arrow(8.02, 4.05, 10.15, 4.05, color=ACC, lw=1.1, ls="--")
    ax.text(9.10, 4.18, "mask + synchronous modality stacks: intrinsically aligned,\n"
            "no registration stage anywhere in the indicator path",
            ha="center", va="bottom", fontsize=7.4, color=ACC)

    # ---- bottom side: registration evaluated and abandoned
    box(3.55, 0.80, 4.60, 1.30,
        "Registration (preliminary, abandoned)\nDIS / optical flow, soft-mask weighted: residual-motion floor $\\approx$ 6 px;\n"
        "warping would resample the measurand (areas, diameters, speckle statistics)",
        fc="#f4f4f4", ec=GREY, fs=7.4, tc=GREY)
    arrow(3.20, 2.70, 4.30, 2.15, color=GREY, lw=1.1, ls="--")
    ax.text(3.45, 2.56, "not in the\nindicator path", ha="left", va="top",
            fontsize=7.0, color=GREY)

    fig.savefig(os.path.join(OUT, "fig1_pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_pipeline.pdf")


if __name__ == "__main__":
    main()
