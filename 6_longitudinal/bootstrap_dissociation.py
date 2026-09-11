"""bootstrap_dissociation.py -- statistical upgrades for the longitudinal
structure-function dissociation

1) Frame-level bootstrap: per-time-point P_mfr samples (every 20th frame)
   resampled -> session means -> 95% CI of slope / total decline
2) Early vs terminal slope comparison (T2-T5 vs T5-T6) with difference test
3) Within-session half-split repeatability coefficient (both hearts)
Output: bootstrap_dissociation.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
sys.stdout.reconfigure(encoding="utf-8")
from seg_common import ResUNet, PAD_H, PAD_W

ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
OUT = r"D:\Project\CC_Project\激光散斑pure\bootstrap_dissociation.json"


def p_mfr_samples(rec_dir, every=20):
    h = json.load(open(os.path.join(rec_dir, "文件头信息.json"), encoding="utf-8"))
    gain, coh = h["signal_gain"], h["coherence_factor"]
    I = np.load(os.path.join(rec_dir, "光强_intensity.npy"), mmap_mode="r")
    V = np.load(os.path.join(rec_dir, "方差_variance.npy"), mmap_mode="r")
    out = []
    for t in range(0, I.shape[0], every):
        m = np.load(os.path.join(rec_dir, "masks", f"{t:05d}.npy")) > 0
        if not m.any():
            continue
        Im = np.asarray(I[t], np.float64)[m].mean()
        Vm = np.asarray(V[t], np.float64)[m].mean()
        C = coh * np.sqrt(np.abs(Vm)) / max(Im, 1e-9)
        out.append(gain * (1.0 / C - 1.0))
    return np.array(out)


def p_mfr_t2_original(every=20):
    """心1 原始 T2: 每20帧推理 + 厂家口径"""
    h = json.load(open(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\文件头信息.json",
                       encoding="utf-8"))
    gain, coh = h["signal_gain"], h["coherence_factor"]
    I = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\光强_intensity.npy",
                mmap_mode="r")
    V = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\方差_variance.npy",
                mmap_mode="r")
    dev = torch.device("cuda")
    m = ResUNet().to(dev)
    m.load_state_dict(torch.load(
        r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行\segmentation\models\seg_I_seed0.pt",
        map_location=dev))
    m.eval()
    out = []
    n, H, W = I.shape
    for t in range(0, n, every):
        img = np.asarray(I[t], np.float64)
        fin = np.isfinite(img)
        lo, hi = np.percentile(img[fin], [1, 99])
        f = np.clip((img - lo) / (hi - lo), 0, 1)
        x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
        xp = F.pad(x, (0, PAD_W - W, 0, PAD_H - H)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            p = torch.sigmoid(m(xp).float())
        mk = p[0, 0, :H, :W].cpu().numpy() > 0.5
        if mk.any():
            Im = img[mk].mean()
            Vm = np.asarray(V[t], np.float64)[mk].mean()
            C = coh * np.sqrt(np.abs(Vm)) / max(Im, 1e-9)
            out.append(gain * (1.0 / C - 1.0))
    del m
    torch.cuda.empty_cache()
    return np.array(out)


def main():
    rng = np.random.RandomState(2026)
    # ---- 帧级样本 ----
    samples = {"T2": p_mfr_t2_original(),
               "T3": p_mfr_samples(os.path.join(ROOT, "zhuxin1_4_20260709-7use")),
               "T4": p_mfr_samples(os.path.join(ROOT, "zhuxin1_5_20260709-10")),
               "T5": p_mfr_samples(os.path.join(ROOT, "zhuxin1_6_20260709-12")),
               "T6": p_mfr_samples(os.path.join(ROOT, "zhuxin1_7_20260709-13"))}
    mins = np.array([11, 15, 44, 79, 113], float)
    keys = ["T2", "T3", "T4", "T5", "T6"]
    means = np.array([samples[k].mean() for k in keys])
    print("会话 P_mfr 均值:", dict(zip(keys, np.round(means, 1))))
    for k in keys:
        print(f"  {k}: n={len(samples[k])}, mean={samples[k].mean():.1f}, "
              f"sd={samples[k].std():.1f}, se={samples[k].std()/np.sqrt(len(samples[k])):.1f}")

    # ---- bootstrap 斜率/降幅 ----
    B = 5000
    slopes, declines = [], []
    early_slopes, late_slopes = [], []
    for b in range(B):
        bm = np.array([rng.choice(samples[k], len(samples[k]), replace=True).mean()
                       for k in keys])
        sl = np.polyfit(mins, bm, 1)[0]
        slopes.append(sl)
        declines.append(100 * (bm[0] - bm[-1]) / bm[0])
        early_slopes.append(np.polyfit(mins[:4], bm[:4], 1)[0])
        late_slopes.append((bm[4] - bm[3]) / (mins[4] - mins[3]))
    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 3),
                round(float(np.percentile(x, 97.5)), 3)]
    slope_ci = ci(slopes)
    decl_ci = ci(declines)
    early_ci = ci(early_slopes)
    late_ci = ci(late_slopes)
    # 末段斜率 > 早期斜率(更陡下降)的概率
    p_steeper = float(np.mean(np.array(late_slopes) < np.array(early_slopes)))
    print(f"\n总斜率: {np.median(slopes):.2f} a.u./min, 95% CI {slope_ci}")
    print(f"总降幅: {np.median(declines):.1f}%, 95% CI {decl_ci}")
    print(f"早期斜率(T2-T5): {np.median(early_slopes):.2f}, CI {early_ci}")
    print(f"末段斜率(T5-T6): {np.median(late_slopes):.2f}, CI {late_ci}")
    print(f"P(末段比早期更陡下降) = {p_steeper:.3f}")

    # ---- 段内半分重复性 (两颗心) ----
    rep = {}
    def half_split_rc(x):
        diffs = []
        for b in range(2000):
            idx = rng.permutation(len(x))
            h1 = x[idx[:len(x)//2]].mean()
            h2 = x[idx[len(x)//2:]].mean()
            diffs.append(h1 - h2)
        return round(float(1.96 * np.std(diffs)), 1)
    rep["heart1_T3"] = half_split_rc(samples["T3"])
    rep["heart1_T6"] = half_split_rc(samples["T6"])
    h2 = {"T0": None}
    for tag, src in [("h2T1", "zhuxin2_2_20260819-2"), ("h2T2", "zhuxin2_3_20260819-3")]:
        rep[f"heart2_{tag}"] = half_split_rc(p_mfr_samples(os.path.join(ROOT, src)))
    # 心2 T0: 用既有均值样本 (phase_a3_perf_mfr_orig: n=133) — 重推理一次
    h2t0 = p_mfr_t2_original.__wrapped__ if False else None
    print("\n段内半分重复性界限 RC95 (a.u.):", rep)

    res = {
        "session_means": {k: round(float(samples[k].mean()), 1) for k in keys},
        "session_n": {k: int(len(samples[k])) for k in keys},
        "slope_median": round(float(np.median(slopes)), 3), "slope_CI95": slope_ci,
        "decline_pct_median": round(float(np.median(declines)), 2),
        "decline_pct_CI95": decl_ci,
        "early_slope_median": round(float(np.median(early_slopes)), 3),
        "early_slope_CI95": early_ci,
        "late_slope_median": round(float(np.median(late_slopes)), 3),
        "late_slope_CI95": late_ci,
        "p_late_steeper_than_early": round(p_steeper, 4),
        "repeatability_RC95": rep,
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
