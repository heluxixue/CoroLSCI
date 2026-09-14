# -*- coding: utf-8 -*-
"""compute_review3_gpu.py -- round-3 review GPU analyses:

1) T1: normalization (C1/C2/C3) x model (zero-shot / fine-tuned) on the SAME
   six held-out frames -> separates the normalization repair from the
   fine-tuning repair (review 4.4).
2) Thick/thin vessel stratified Dice with a defensible definition: local
   radius = distance transform of the CONSENSUS mask; thin = local radius
   <= 3 px (diameter <= 6 px), thick = > 3 px; pooled per stratum over the
   20-frame set; extra-mask false positives counted separately
   (review P1-2).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

SRC = r"D:\Project\CC_Project\激光散斑pure"
sys_path = SRC
import sys
sys.path.insert(0, sys_path)
from seg_common import ResUNet, PAD_H, PAD_W

T2_DIR = os.path.join(SRC, "DAT解析结果", "整段数据_2641帧")
T1_DIR = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果\zhuxin1_3_20260709-4"
T1_MASKS = os.path.join(SRC, "DAT解析结果", "标注样本_T1特写段20帧对应标注", "SegmentationClass")
E1 = os.path.join(SRC, "DAT解析结果", "专家1对猪心1随机挑选20帧图像的标注", "SegmentationClass")
E2 = os.path.join(SRC, "DAT解析结果", "第二专家20帧对应的标注", "SegmentationClass")
OUT = os.path.join(SRC, "review3_gpu.json")

SEED0 = os.path.join(SRC, "DAT解析结果", "第三版执行", "segmentation", "models", "seg_I_seed0.pt")
SEED_T1 = os.path.join(T1_DIR, "finetune", "seg_I_T1.pt")


def norm_cand(frame, tag, ranges):
    fin = np.isfinite(frame)
    lo1, hi1 = np.percentile(frame[fin], [1, 99])
    if tag == "C1":
        lo, hi = lo1, hi1
    elif tag == "C2":
        lo, hi = ranges["C2"]
    else:
        lo, hi = ranges["C3"]
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


def predict(model, img, dev, padv, padh):
    f = norm_cand(img, "C1", None)
    x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
    xp = F.pad(x, (0, padv, 0, padh)).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p = torch.sigmoid(model(xp).float())
    return p[0, 0, :f.shape[0], :f.shape[1]].cpu().numpy()


def predict_norm(model, img, dev, tag, ranges):
    f = norm_cand(img, tag, ranges)
    H, W = f.shape
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
    xp = F.pad(x, (0, pw, 0, ph)).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p = torch.sigmoid(model(xp).float())
    return p[0, 0, :H, :W].cpu().numpy()


def load_mask(d, t):
    a = np.array(Image.open(os.path.join(d, f"{t}.png")))
    return (a[..., 1] == 255) if a.ndim == 3 else (a > 127)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def main():
    dev = torch.device("cuda")
    res = {}

    # ---------- 1) T1 normalization x finetune table ----------
    h1 = json.load(open(os.path.join(T1_DIR, "文件头信息.json"), encoding="utf-8"))
    I1 = np.load(os.path.join(T1_DIR, "光强_intensity.npy"), mmap_mode="r")
    split = json.load(open(os.path.join(T1_DIR, "finetune", "split.json"), encoding="utf-8"))
    val_frames = split["val"]
    s = np.asarray(I1[::50]).ravel(); s = s[np.isfinite(s)]
    c2 = (float(np.percentile(s, 1)), float(np.percentile(s, 99)))
    I2s = np.load(os.path.join(T2_DIR, "光强_intensity.npy"), mmap_mode="r")
    s2 = np.asarray(I2s[::50]).ravel(); s2 = s2[np.isfinite(s2)]
    c3 = (float(np.percentile(s2, 1)), float(np.percentile(s2, 99)))
    ranges = {"C2": c2, "C3": c3}

    m0 = ResUNet().to(dev)
    m0.load_state_dict(torch.load(SEED0, map_location=dev)); m0.eval()
    m1 = ResUNet().to(dev)
    m1.load_state_dict(torch.load(SEED_T1, map_location=dev)); m1.eval()

    table = {}
    for name, mdl in (("zero_shot", m0), ("finetuned", m1)):
        for tag in ("C1", "C2", "C3"):
            ds = []
            for frm in val_frames:
                img = np.asarray(I1[frm], np.float64)
                p = predict_norm(mdl, img, dev, tag, ranges)
                gt = load_mask(T1_MASKS, frm)
                ds.append(dice(p > 0.5, gt))
            table[f"{name}_{tag}"] = round(float(np.mean(ds)), 4)
    res["t1_norm_x_finetune_val6_dice"] = table
    print("T1 table:", table)

    # ---------- 2) stratified thick/thin with local-radius definition ----------
    FRAMES = [1475, 1507, 1530, 1649, 1689, 1732, 1800, 1848, 1882, 1919,
              2045, 2050, 2112, 2120, 2129, 2157, 2214, 2478, 2554, 2634]
    I = np.load(os.path.join(T2_DIR, "光强_intensity.npy"), mmap_mode="r")
    from skimage.morphology import skeletonize
    TAU = 3.0  # local half-width threshold (px): radius <= 3 -> thin (width <= ~6 px)
    agg = {}
    extra_fp_total = 0.0
    cons_px_total = 0
    full_dice = {"aE1": [], "aE2": [], "E1E2": []}
    for frm in FRAMES:
        e1 = load_mask(E1, frm)
        e2 = load_mask(E2, frm)
        cons = e1 & e2
        if not cons.any():
            continue
        # local radius: distance to background at the nearest skeleton point
        D = distance_transform_edt(cons)
        S = skeletonize(cons)
        _, (iy, ix) = distance_transform_edt(~S, return_indices=True)
        radius = D[iy, ix]
        thin = cons & (radius <= TAU)
        thick = cons & (radius > TAU)
        algo = predict_norm(m0, np.asarray(I[frm], np.float64), dev, "C1", ranges) > 0.5
        extra_fp_total += float((algo & ~cons).sum())
        cons_px_total += int(cons.sum())
        for gt, name in ((e1, "E1"), (e2, "E2")):
            for st, reg in (("thin", thin), ("thick", thick)):
                tp = float((algo & gt & reg).sum())
                fp = float((algo & reg & ~gt).sum())
                fn = float((gt & reg & ~algo).sum())
                k = f"{st}_{name}"
                a = agg.setdefault(k, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "px": 0})
                a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
                a["px"] += int(reg.sum())
        full_dice["aE1"].append(dice(algo, e1))
        full_dice["aE2"].append(dice(algo, e2))
        full_dice["E1E2"].append(dice(e1, e2))
    strat = {}
    for k, a in agg.items():
        strat[k] = {"dice": round(2 * a["tp"] / max(2 * a["tp"] + a["fp"] + a["fn"], 1e-9), 4),
                    "px": a["px"]}
    res["stratified_v2"] = strat
    res["full_dice_mean"] = {k: round(float(np.mean(v)), 4) for k, v in full_dice.items()}
    res["extra_consensus_fp_px_total"] = extra_fp_total
    res["consensus_px_total"] = cons_px_total
    print("stratified:", json.dumps(strat, ensure_ascii=False, indent=1))
    print("full dice:", res["full_dice_mean"])

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
