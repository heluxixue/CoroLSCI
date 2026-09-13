"""seed_variance.py -- segmentation-seed variance effect on longitudinal
indicators

Heart-1 T3-T6 segments re-inferred with seg_I_seed1/seed2 (per frame) ->
session-level indicators + P_mfr (every 20th frame) compared with seed0,
bounding the model-choice uncertainty of the dissociation finding
(flat geometry / declining perfusion).
Output: seed_variance.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
sys.stdout.reconfigure(encoding="utf-8")
from seg_common import ResUNet

ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
MODDIR = (r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行"
          r"\segmentation\models")
OUT = r"D:\Project\CC_Project\激光散斑pure\seed_variance.json"
SESSIONS = [("T3", "zhuxin1_4_20260709-7use"), ("T4", "zhuxin1_5_20260709-10"),
            ("T5", "zhuxin1_6_20260709-12"), ("T6", "zhuxin1_7_20260709-13")]


def session_metrics(model, dev, rec_dir):
    h = json.load(open(os.path.join(rec_dir, "文件头信息.json"), encoding="utf-8"))
    res_mm = h["resolution_mm_per_pixel"]
    I = np.load(os.path.join(rec_dir, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(rec_dir, "方差_variance.npy"), mmap_mode="r")
    n, H, W = I.shape
    PH, PW = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    areas, pms = [], []
    from scipy.ndimage import label as cc_label, convolve
    from skimage.morphology import skeletonize
    import cv2
    S8 = np.ones((3, 3), int)
    def fill10(m):
        lab, nn = cc_label(~m)
        if nn == 0:
            return m
        bd = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
        out = m.copy()
        for i in range(1, nn + 1):
            if i in bd:
                continue
            c = lab == i
            if c.sum() <= 10:
                out |= c
        return out
    for t in range(0, n, 5):   # 每5帧采样: 面积
        img = np.asarray(I[t], dtype=np.float64)
        fin = np.isfinite(img)
        lo, hi = np.percentile(img[fin], [1, 99])
        f = np.clip((img - lo) / (hi - lo), 0, 1)
        x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
        xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            p = torch.sigmoid(model(xp).float())
        mk = fill10(p[0, 0, :H, :W].cpu().numpy() > 0.5)
        areas.append(mk.sum() * res_mm * res_mm)
    for t in range(0, n, 20):  # P_mfr
        img = np.asarray(I[t], dtype=np.float64)
        fin = np.isfinite(img)
        lo, hi = np.percentile(img[fin], [1, 99])
        f = np.clip((img - lo) / (hi - lo), 0, 1)
        x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
        xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            p = torch.sigmoid(model(xp).float())
        mk = p[0, 0, :H, :W].cpu().numpy() > 0.5
        if mk.any():
            Im = img[mk].mean()
            Vm = np.asarray(V[t], np.float64)[mk].mean()
            C = h["coherence_factor"] * np.sqrt(np.abs(Vm)) / max(Im, 1e-9)
            pms.append(h["signal_gain"] * (1.0 / C - 1.0))
    return float(np.mean(areas)), float(np.mean(pms))


def main():
    dev = torch.device("cuda")
    results = {}
    for seed in (1, 2):
        m = ResUNet().to(dev)
        m.load_state_dict(torch.load(os.path.join(MODDIR, f"seg_I_seed{seed}.pt"),
                                     map_location=dev))
        m.eval()
        for tag, src in SESSIONS:
            t0 = time.time()
            a, pm = session_metrics(m, dev, os.path.join(ROOT, src))
            results[f"{tag}_seed{seed}"] = {"area_mm2": round(a, 1),
                                            "P_mfr": round(pm, 1)}
            print(f"seed{seed} {tag}: area={a:.1f} mm²  P_mfr={pm:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
    # seed0 基准 (来自主表)
    base = {"T3": (542.3, 1010.6), "T4": (554.6, 967.7),
            "T5": (551.0, 975.0), "T6": (534.4, 806.8)}
    summary = {}
    for tag in ("T3", "T4", "T5", "T6"):
        areas = [base[tag][0]] + [results[f"{tag}_seed{s}"]["area_mm2"] for s in (1, 2)]
        pms = [base[tag][1]] + [results[f"{tag}_seed{s}"]["P_mfr"] for s in (1, 2)]
        summary[tag] = {"area_mean": round(float(np.mean(areas)), 1),
                        "area_sd_across_seeds": round(float(np.std(areas)), 2),
                        "P_mfr_mean": round(float(np.mean(pms)), 1),
                        "P_mfr_sd_across_seeds": round(float(np.std(pms)), 2)}
        print(f"{tag}: area {np.mean(areas):.1f}±{np.std(areas):.2f} mm² | "
              f"P_mfr {np.mean(pms):.1f}±{np.std(pms):.2f}")
    json.dump({"per_run": results, "summary": summary},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
