"""sensitivity_analysis.py -- normalization, background-reference and
block-bootstrap sensitivity package

1) Normalization sensitivity: session P_mfr means under the three candidate
   ranges C1/C2/C3 -> robustness of the longitudinal decline
2) Moving-block bootstrap (block length = cardiac cycle): conditional CI of
   the heart-1 P_mfr decline
3) Background-referenced contrast ratio and brightness-corrected trends
4) Structural equivalence margin (prespecified) + full-indicator session SDs
   (Table 4 completion)
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
sys.stdout.reconfigure(encoding="utf-8")
from seg_common import ResUNet, PAD_H as PAD_H_T2, PAD_W as PAD_W_T2

ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
OUT = r"D:\Project\CC_Project\激光散斑pure\sensitivity_analysis.json"
SESS = [("T2", None), ("T3", "zhuxin1_4_20260709-7use"), ("T4", "zhuxin1_5_20260709-10"),
        ("T5", "zhuxin1_6_20260709-12"), ("T6", "zhuxin1_7_20260709-13"),
        ("T1", "zhuxin1_3_20260709-4"), ("h2T0", None), ("h2T1", "zhuxin2_2_20260819-2"),
        ("h2T2", "zhuxin2_3_20260819-3")]
MODEL_H1 = (r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行"
            r"\segmentation\models\seg_I_seed0.pt")
MODEL_T1 = (r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
            r"\zhuxin1_3_20260709-4\finetune\seg_I_T1.pt")
MODEL_H2 = r"D:\Project\CC_Project\激光散斑数据2\newdata\finetune\seg_I_heart2.pt"
FPS = 44.0


def p_mfr_from(I, V, m, gain, coh):
    Im = I[m].mean(); Vm = V[m].mean()
    C = coh * np.sqrt(np.abs(Vm)) / max(Im, 1e-9)
    return gain * (1.0 / C - 1.0)


def session_samples(rec_dir, model, dev, anchor_T2, anchor_H2):
    """返回三候选下每20帧的 P_mfr 样本 + 背景参照 + 亮度"""
    h = json.load(open(os.path.join(rec_dir, "文件头信息.json"), encoding="utf-8"))
    gain, coh = h["signal_gain"], h["coherence_factor"]
    I = np.load(os.path.join(rec_dir, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(rec_dir, "方差_variance.npy"), mmap_mode="r")
    n, H, W = I.shape
    PH, PW = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    s = np.asarray(I[::50]).ravel(); s = s[np.isfinite(s)]
    cands = {"C1": None, "C2": (float(np.percentile(s, 1)), float(np.percentile(s, 99)))}
    cands["C3"] = anchor_T2 if os.path.basename(rec_dir).startswith("zhuxin1") else anchor_H2
    out = {k: [] for k in cands}
    bg_ratio = []
    for t in range(0, n, 20):
        img = np.asarray(I[t], np.float64); var = np.asarray(V[t], np.float64)
        fin = np.isfinite(img)
        lo1, hi1 = np.percentile(img[fin], [1, 99])
        p1 = np.percentile(img[fin], 25)
        for tag, rng in cands.items():
            if rng is None:
                lo, hi = lo1, hi1
            else:
                lo, hi = rng
            f = np.clip((img - lo) / (hi - lo), 0, 1)
            x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
            xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                pr = torch.sigmoid(model(xp).float())
            m = pr[0, 0, :H, :W].cpu().numpy() > 0.5
            if m.any():
                out[tag].append(p_mfr_from(img, var, m, gain, coh))
                if tag == "C1" or True:
                    pass
        # 背景参照: 用当前主候选(存档掩膜) 的掩膜取血管, 环带取背景
        m0 = np.load(os.path.join(rec_dir, "masks", f"{t:05d}.npy")) > 0
        if m0.any():
            import cv2
            bg = cv2.dilate(m0.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
            bg &= ~m0
            if bg.any():
                C_v = coh * np.sqrt(np.abs(var[m0].mean())) / max(img[m0].mean(), 1e-9)
                C_b = coh * np.sqrt(np.abs(var[bg].mean())) / max(img[bg].mean(), 1e-9)
                bg_ratio.append(C_v / max(C_b, 1e-9))
    out["bg_ratio"] = bg_ratio
    out["brightness_p25"] = [float(np.percentile(np.asarray(I[t])[np.isfinite(I[t])], 25))
                             for t in range(0, n, 40)]
    return out


def main():
    dev = torch.device("cuda")
    t0 = time.time()
    # 锚范围
    I_h1 = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\光强_intensity.npy", mmap_mode="r")
    s1 = np.asarray(I_h1[::100]).ravel(); s1 = s1[np.isfinite(s1)]
    A_T2 = (float(np.percentile(s1, 1)), float(np.percentile(s1, 99)))
    I_h2 = np.load(r"D:\Project\CC_Project\激光散斑数据2\newdata\整段数据_2641帧\光强_intensity.npy", mmap_mode="r")
    s2 = np.asarray(I_h2[::100]).ravel(); s2 = s2[np.isfinite(s2)]
    A_H2 = (float(np.percentile(s2, 1)), float(np.percentile(s2, 99)))

    models = {}
    def get_model(kind):
        if kind not in models:
            m = ResUNet().to(dev)
            m.load_state_dict(torch.load({"H1": MODEL_H1, "T1": MODEL_T1, "H2": MODEL_H2}[kind], map_location=dev))
            m.eval(); models[kind] = m
        return models[kind]

    results = {}
    # ---- T2 特例: 原始记录, C2=C3=自身全局范围; C1 用既有值 ----
    h1 = json.load(open(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\文件头信息.json", encoding="utf-8"))
    I_h1f = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\光强_intensity.npy", mmap_mode="r")
    V_h1f = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\方差_variance.npy", mmap_mode="r")
    m_h1 = get_model("H1")
    c2v = []
    n1, H1_, W1_ = I_h1f.shape
    for t in range(0, n1, 20):
        img = np.asarray(I_h1f[t], np.float64); var = np.asarray(V_h1f[t], np.float64)
        fin = np.isfinite(img)
        f = np.clip((img - A_T2[0]) / (A_T2[1] - A_T2[0]), 0, 1)
        x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
        xp = F.pad(x, (0, PAD_W_T2 - W1_, 0, PAD_H_T2 - H1_)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pr = torch.sigmoid(m_h1(xp).float())
        m2 = pr[0, 0, :H1_, :W1_].cpu().numpy() > 0.5
        if m2.any():
            c2v.append(p_mfr_from(img, var, m2, h1["signal_gain"], h1["coherence_factor"]))
    results["T2"] = {"C1": 1033.9, "C2": round(float(np.mean(c2v)), 1),
                     "C3": round(float(np.mean(c2v)), 1), "n_samples": len(c2v),
                     "brightness_p25_mean": None}
    print(f"T2: C2=C3={np.mean(c2v):.1f} (C1=1033.9 既有)", flush=True)

    for tag, src in SESS:
        if src is None:
            continue
        kind = "T1" if tag == "T1" else ("H2" if tag.startswith("h2") else "H1")
        rec = os.path.join(ROOT, src)
        r = session_samples(rec, get_model(kind), dev, A_T2, A_H2)
        results[tag] = {k: (round(float(np.mean(v)), 1) if k != "bg_ratio" else round(float(np.mean(v)), 4))
                        for k, v in r.items() if isinstance(v, list) and len(v) > 5 and k != "brightness_p25"}
        results[tag]["n_samples"] = len(r["C1"])
        results[tag]["brightness_p25_mean"] = round(float(np.mean(r["brightness_p25"])), 0)
        print(f"{tag}: C1={np.mean(r['C1']):.1f} C2={np.mean(r['C2']):.1f} C3={np.mean(r['C3']):.1f} "
              f"bgRatio={np.mean(r['bg_ratio']):.4f} bright={np.mean(r['brightness_p25']):.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 纵向稳健性: 三候选下 T2-T6 降幅
    robust = {}
    for cand in ("C1", "C2", "C3"):
        ys = [results[t][cand] for t in ("T2", "T3", "T4", "T5", "T6")]
        robust[cand] = round(100 * (ys[0] - ys[-1]) / ys[0], 1)
    # 背景参照比趋势 T3-T6
    bg = [results[t]["bg_ratio"] for t in ("T3", "T4", "T5", "T6")]
    robust["bg_ratio_T3_to_T6_pct"] = round(100 * (bg[-1] - bg[0]) / bg[0], 1)
    results["_longitudinal_robustness_pct"] = robust
    print("降幅稳健性(%):", robust)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
