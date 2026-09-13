# -*- coding: utf-8 -*-
"""equivalence_hd95.py — 审稿 §4.4/§4.3.5 补强
1) 面积净变化的两阶段块汇总自助 (90% CI, TOST 等效检验用)
2) 20 帧三方集: HD95 (算法 vs 两专家; 专家间) + 细/粗血管分层 Dice
输出: equivalence_hd95.json
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
sys.stdout.reconfigure(encoding="utf-8")
from seg_common import ResUNet, PAD_H, PAD_W

ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
OUT = r"D:\Project\CC_Project\激光散斑pure\equivalence_hd95.json"
FPS = 44.0

# ---------- 1) 面积等效: 块汇总自助 90% CI ----------
def area_series(rec):
    rows = list(csv.DictReader(open(os.path.join(rec, "指标_全帧.csv"), encoding="utf-8")))
    return np.array([float(r["area_mm2"]) for r in rows])

series = {}
for tag, src in [("T3", "zhuxin1_4_20260709-7use"), ("T4", "zhuxin1_5_20260709-10"),
                 ("T5", "zhuxin1_6_20260709-12"), ("T6", "zhuxin1_7_20260709-13")]:
    series[tag] = area_series(os.path.join(ROOT, src))
# T2 原始: 存档 CSV 像素面积 × res²
rows = list(csv.DictReader(open(
    r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行\60s统一结果\冠脉分析_指标_原始灌注.csv",
    encoding="utf-8-sig")))
res = 0.20849377889155601
series["T2"] = np.array([float(r["area"]) for r in rows]) * res * res

order = ["T2", "T3", "T4", "T5", "T6"]
bmeans = {k: np.array([b.mean() for b in np.array_split(series[k][:2640], 12)]) for k in order}
rng = np.random.RandomState(2026)
nets = []
for b in range(4000):
    bm = np.array([bmeans[k][rng.randint(0, 12, 12)].mean() for k in order])
    nets.append(100 * (bm[-1] - bm[0]) / bm[0])
ci90 = [round(float(np.percentile(nets, 5)), 2), round(float(np.percentile(nets, 95)), 2)]
net_point = round(100 * (series["T2"].mean() - series["T6"].mean()) / series["T2"].mean() * -1, 1)
equiv = {"area_net_change_pct": net_point, "CI90": ci90,
         "margin_pct": 5.0,
         "equivalent": bool(ci90[0] > -5.0 and ci90[1] < 5.0)}
print("面积等效 (TOST 口径):", equiv)

# ---------- 2) 20 帧三方: HD95 + 细/粗分层 ----------
FRAMES = [1475,1507,1530,1649,1689,1732,1800,1848,1882,1919,2045,2050,2112,2120,2129,2157,2214,2478,2554,2634]
E1 = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\专家1对猪心1随机挑选20帧图像的标注\SegmentationClass"
E2 = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第二专家20帧对应的标注\SegmentationClass"
I = np.load(r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\整段数据_2641帧\光强_intensity.npy", mmap_mode="r")
dev = torch.device("cuda")
m = ResUNet().to(dev)
m.load_state_dict(torch.load(
    r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行\segmentation\models\seg_I_seed0.pt",
    map_location=dev))
m.eval()

def load(d, t):
    a = np.array(Image.open(os.path.join(d, f"{t}.png")))
    return (a[..., 1] == 255) if a.ndim == 3 else (a > 127)

def hd95(a, b):
    da = distance_transform_edt(~a)
    db = distance_transform_edt(~b)
    ia = np.sort(da[b]); ib = np.sort(db[a])
    if len(ia) == 0 or len(ib) == 0:
        return np.nan
    return float(max(ia[int(0.95 * len(ia)) - 1], ib[int(0.95 * len(ib)) - 1]))

def strat_dice(pred, gt, consensus):
    """按共识骨架距离分层: 距骨架>6px 的共识像素区为粗干, 其余为细支"""
    sk = skeletonize(consensus)
    d = distance_transform_edt(~sk)
    thin = consensus & (d > 6)
    thick = consensus & (d <= 6)
    out = {}
    for nm, reg in [("thin", thin), ("thick", thick)]:
        if reg.sum() < 100:
            out[nm] = None
            continue
        inter = float((pred & reg).sum())
        gt_r = float((gt & reg).sum())
        out[nm] = round(2 * inter / (float((pred & consensus).sum()) + float((gt & consensus).sum()))
                        * 1.0, 4)  # 占位, 下方换真算
        # 真算: 区域内两集 Dice
        p_in = pred & reg
        g_in = gt & reg
        s = p_in.sum() + g_in.sum()
        out[nm] = round(float(2 * (p_in & g_in).sum() / s), 4) if s > 0 else None
    return out

hd_ae1, hd_ae2, hd_e1e2 = [], [], []
thin_ae1, thick_ae1, thin_ae2, thick_ae2 = [], [], [], []
for t in FRAMES:
    e1, e2 = load(E1, t), load(E2, t)
    img = np.asarray(I[t], dtype=np.float64)
    fin = np.isfinite(img)
    lo, hi = np.percentile(img[fin], [1, 99])
    f = np.clip((img - lo) / (hi - lo), 0, 1)
    x = torch.from_numpy(np.repeat(f[None].astype(np.float32), 3, 0))
    xp = F.pad(x, (0, PAD_W - f.shape[1], 0, PAD_H - f.shape[0])).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p = torch.sigmoid(m(xp).float())
    algo = p[0, 0, :f.shape[0], :f.shape[1]].cpu().numpy() > 0.5
    hd_ae1.append(hd95(algo, e1)); hd_ae2.append(hd95(algo, e2)); hd_e1e2.append(hd95(e1, e2))
    cons = e1 & e2
    s1 = strat_dice(algo, e1, cons); s2 = strat_dice(algo, e2, cons)
    thin_ae1.append(s1["thin"]); thick_ae1.append(s1["thick"])
    thin_ae2.append(s2["thin"]); thick_ae2.append(s2["thick"])
del m; torch.cuda.empty_cache()

def ms(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 2), round(float(np.std(v)), 2)

res3 = {
    "HD95_px": {
        "algo_vs_E1": ms(hd_ae1), "algo_vs_E2": ms(hd_ae2), "E1_vs_E2": ms(hd_e1e2)},
    "stratified_dice": {
        "thin_vessels_algo_E1": ms(thin_ae1), "thick_vessels_algo_E1": ms(thick_ae1),
        "thin_vessels_algo_E2": ms(thin_ae2), "thick_vessels_algo_E2": ms(thick_ae2)},
}
print("HD95(px):", json.dumps(res3["HD95_px"], ensure_ascii=False))
print("分层 Dice:", json.dumps(res3["stratified_dice"], ensure_ascii=False))
json.dump({"equivalence": equiv, "metrics": res3},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", OUT)
