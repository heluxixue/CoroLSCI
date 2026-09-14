# -*- coding: utf-8 -*-
"""compute_review4_gpu.py — round-4 GPU reanalyses, fixing R1-3 / R2-3:

1) CONTOUR-based HD95 (proper boundary distance): extract 1-px inner contours
   of both masks, directed nearest-boundary distances, report P95 of each
   direction and their max (the standard symmetric HD95 convention).
   Three comparisons: algo-E1, algo-E2, E1-E2, over the fresh 20-frame set.
2) Stratified metric RENAMED to what it is: consensus-region coverage
   (prediction \\ consensus region / region area) per skeleton-anchored
   radius stratum; no Dice claim, no FP claim inside consensus.
3) Extra prediction pixels split into (a) expert-disagreement zone (E1 XOR E2)
   and (b) both-background region (~(E1|E2)); counts and shares.

Output: review4_gpu.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
from seg_common import ResUNet

SRC = r"D:\Project\CC_Project\激光散斑pure"
T2DIR = os.path.join(SRC, "DAT解析结果", "整段数据_2641帧")
E1D = os.path.join(SRC, "DAT解析结果", "专家1对猪心1随机挑选20帧图像的标注",
                   "SegmentationClass")
E2D = os.path.join(SRC, "DAT解析结果", "第二专家20帧对应的标注", "SegmentationClass")
SEED0 = os.path.join(SRC, "DAT解析结果", "第三版执行", "segmentation",
                     "models", "seg_I_seed0.pt")
OUT = os.path.join(SRC, "review4_gpu.json")

FRAMES = [1475, 1507, 1530, 1649, 1689, 1732, 1800, 1848, 1882, 1919,
          2045, 2050, 2112, 2120, 2129, 2157, 2214, 2478, 2554, 2634]
TAU = 3.0


def load_mask(d, t):
    a = np.array(Image.open(os.path.join(d, f"{t}.png")))
    return (a[..., 1] == 255) if a.ndim == 3 else (a > 127)


def contour(m):
    return m & ~binary_erosion(m, iterations=1)


def hd95_contour(a, b):
    """symmetric 95th-percentile contour-to-contour distance in px"""
    ca, cb = contour(a), contour(b)
    if not ca.any() or not cb.any():
        return np.nan, np.nan, np.nan
    dab = distance_transform_edt(~cb)[ca]   # contour(a) -> nearest contour(b)
    dba = distance_transform_edt(~ca)[cb]   # contour(b) -> nearest contour(a)
    p95ab, p95ba = float(np.percentile(dab, 95)), float(np.percentile(dba, 95))
    return p95ab, p95ba, max(p95ab, p95ba)


def main():
    dev = torch.device("cuda")
    m0 = ResUNet().to(dev)
    m0.load_state_dict(torch.load(SEED0, map_location=dev))
    m0.eval()
    I = np.load(os.path.join(T2DIR, "光强_intensity.npy"), mmap_mode="r")

    h95 = {"algo_E1": [], "algo_E2": [], "E1_E2": []}
    cov = {"thin": [], "thick": []}
    extra = {"xor_zone": 0.0, "both_bg": 0.0}
    cons_px = 0.0

    for t in FRAMES:
        e1, e2 = load_mask(E1D, t), load_mask(E2D, t)
        img = np.asarray(I[t], np.float64)
        fin = np.isfinite(img)
        lo, hi = np.percentile(img[fin], [1, 99])
        f = np.clip((img - lo) / (hi - lo), 0, 1)
        H, W = f.shape
        ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
        x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
        xp = F.pad(x, (0, pw, 0, ph)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            p = torch.sigmoid(m0(xp).float())
        algo = p[0, 0, :H, :W].cpu().numpy() > 0.5

        for key, a, b in (("algo_E1", algo, e1), ("algo_E2", algo, e2),
                          ("E1_E2", e1, e2)):
            _, _, h = hd95_contour(a, b)
            h95[key].append(h)

        cons = e1 & e2
        cons_px += cons.sum()
        D = distance_transform_edt(cons)
        S = skeletonize(cons)
        _, (iy, ix) = distance_transform_edt(~S, return_indices=True)
        radius = D[iy, ix]
        for st, reg in (("thin", cons & (radius <= TAU)),
                        ("thick", cons & (radius > TAU))):
            if reg.any():
                cov[st].append(float((algo & reg).sum()) / reg.sum())
        ex = algo & ~cons
        extra["xor_zone"] += float((ex & (e1 ^ e2)).sum())
        extra["both_bg"] += float((ex & ~(e1 | e2)).sum())

    res = {
        "contour_hd95_px": {
            k: {"mean": round(float(np.nanmean(v)), 2),
                "sd": round(float(np.nanstd(v)), 2)} for k, v in h95.items()},
        "consensus_coverage": {
            k: round(float(np.nanmean(v)), 4) for k, v in cov.items()},
        "extra_prediction_px": {
            "expert_disagreement_zone": int(extra["xor_zone"]),
            "both_experts_background": int(extra["both_bg"]),
            "total": int(extra["xor_zone"] + extra["both_bg"]),
            "consensus_px_total": int(cons_px),
            "note": "disagreement zone = E1 XOR E2; both-background = "
                    "predicted where BOTH experts label background"},
        "definition": ("contour = mask minus 1-px binary erosion; HD95 = max of "
                       "the two directed 95th-percentile contour distances"),
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
