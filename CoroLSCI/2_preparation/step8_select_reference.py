"""step8_select_reference.py -- Reference frame selection

Selects the unified reference frame (global frame 2562) for registration
based on stability and quality criteria.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
OUT_DIR = os.path.join(EXE_DIR, "reference")
os.makedirs(OUT_DIR, exist_ok=True)

PREP_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧_预处理")
ANN_PNG_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "标注样本_后30s_预处理200帧")
MASK_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "200mask", "SegmentationClass")

OFFSET = 1320  # npz local frame no. → global frame no. = local + 1320

print("=" * 60)
print("0. 加载数据")
print("=" * 60)
I = np.load(os.path.join(PREP_DIR, "光强_intensity_p1p99.npy"), mmap_mode="r")
cyc = np.load(os.path.join(EXE_DIR, "后30s_周期检测.npz"))
bounds_local = cyc["bounds"]          # (40,) cycle border boundary (local frame no. )
frame_cycle = cyc["frame_cycle"]      # (1321,) each frame cycle no.
motion_local = cyc["motion"]          # (1321,) frame between

# training frame
rows = [r for r in csv.DictReader(open(
    os.path.join(EXE_DIR, "dataset_split.csv"), encoding="utf-8-sig")) if r["split"] == "train"]
train_fids = [int(r["frame_id"]) for r in rows]
print(f"训练帧候选: {len(train_fids)}")

# full sequence each frame p50/p90 (p1-p99 sequence ; p99 as 1.0 region part , use p90)
print("计算全序列逐帧 p50/p90...")
p50_all = np.zeros(2641)
p90_all = np.zeros(2641)
for t in range(2641):
    f = I[t]
    fin = f[np.isfinite(f)]
    p50_all[t] = np.percentile(fin, 50)
    p90_all[t] = np.percentile(fin, 90)

med50, iqr50 = np.median(p50_all), np.percentile(p50_all, 75) - np.percentile(p50_all, 25)
med90, iqr90 = np.median(p90_all), np.percentile(p90_all, 75) - np.percentile(p90_all, 25)
med_mot, iqr_mot = np.median(motion_local), np.percentile(motion_local, 75) - np.percentile(motion_local, 25)
print(f"全序列: p50 中位数={med50:.3f} IQR={iqr50:.3f}; p90 中位数={med90:.3f} IQR={iqr90:.3f}")

print("=" * 60)
print("1. 候选帧打分")
print("=" * 60)
cands = []
for fi in train_fids:
    loc = fi - OFFSET
    s50 = abs(p50_all[fi] - med50) / iqr50
    s90 = abs(p90_all[fi] - med90) / iqr90
    sm = abs(motion_local[loc] - med_mot) / iqr_mot
    cands.append({
        "frame_id": fi,
        "cycle_id": int(frame_cycle[loc]),
        "p50": p50_all[fi], "p90": p90_all[fi], "motion": motion_local[loc],
        "score": s50 + s90 + sm,
    })
cands.sort(key=lambda c: c["score"])

# cycle inside phase ( frame in its cycle inside for position )
bounds_glob = bounds_local + OFFSET
for c in cands:
    fi = c["frame_id"]
    starts = bounds_glob[bounds_glob <= fi]
    ends = bounds_glob[bounds_glob > fi]
    cyc_start = int(starts[-1]); cyc_end = int(ends[0]) if len(ends) else 2640
    c["phase_frac"] = (fi - cyc_start) / max(cyc_end - cyc_start, 1)
    c["cyc_start"], c["cyc_end"] = cyc_start, cyc_end

best = cands[0]
print(f"最优参考帧: frame {best['frame_id']} (t={best['frame_id']/44:.2f}s)  "
      f"score={best['score']:.3f}  周期#{best['cycle_id']} phase ={best['phase_frac']:.2f}")
for c in cands[:5]:
    print(f"  候选: frame {c['frame_id']} score={c['score']:.3f} "
          f"相位={c['phase_frac']:.2f} p50={c['p50']:.3f} motion={c['motion']:.1f}")

# ============================================================
# 2. visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
xs = [c["p50"] for c in cands]
ys = [c["p90"] for c in cands]
sc = ax.scatter(xs, ys, c=[c["score"] for c in cands], cmap="viridis_r", s=28)
ax.scatter([best["p50"]], [best["p90"]], marker="*", s=300, c="red",
           edgecolors="k", label=f"参考帧 {best['frame_id']}", zorder=5)
ax.axvline(med50, color="gray", ls="--", lw=1)
ax.axhline(med90, color="gray", ls="--", lw=1)
ax.set_xlabel("Intensity p50"); ax.set_ylabel("Intensity p90")
ax.set_title("训练帧候选 (p50, p90 分布, 虚线=全序列中位数)")
ax.legend(); ax.grid(alpha=0.3)
fig.colorbar(sc, ax=ax, label="score")

ax = axes[1]
ax.scatter([c["phase_frac"] for c in cands], [c["score"] for c in cands], s=28)
ax.scatter([best["phase_frac"]], [best["score"]], marker="*", s=300, c="red",
           edgecolors="k", zorder=5)
ax.set_xlabel("周期内相位 (0=周期起点, 1=下一周期起点)")
ax.set_ylabel("score")
ax.set_title("score 与心动周期相位的关系")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "参考帧选择图.png"), dpi=150)
plt.close(fig)
print("-> 参考帧选择图.png")

# reference frame + annotation mask overlay
img = np.array(Image.open(os.path.join(ANN_PNG_DIR, f"{best['frame_id']}.png")).convert("L"))
mask = np.array(Image.open(os.path.join(MASK_DIR, f"{best['frame_id']}.png")))
mask_bin = (mask.sum(axis=-1) > 0) if mask.ndim == 3 else mask > 0
ov = np.stack([img, img, img], -1).astype(np.float32) / 255.0
ov[mask_bin] = ov[mask_bin] * 0.4 + np.array([1.0, 0.3, 0.3]) * 0.6
Image.fromarray((np.clip(ov, 0, 1) * 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "参考帧_叠加标注.png"))
print("-> 参考帧_叠加标注.png")

# ============================================================
# 3. save results
# ============================================================
result = {
    "frame_id": best["frame_id"],
    "timestamp_s": best["frame_id"] / 44.0,
    "cardiac_cycle_id": best["cycle_id"],
    "phase_frac": best["phase_frac"],
    "cyc_start": best["cyc_start"], "cyc_end": best["cyc_end"],
    "p50": best["p50"], "p90": best["p90"], "motion": best["motion"],
    "score": best["score"],
    "criterion": "|p50-med|/IQR + |p90-med|/IQR + |motion-med|/IQR (全2641帧标准)",
    "top5_candidates": [{k: c[k] for k in ("frame_id", "score", "phase_frac", "cycle_id")}
                        for c in cands[:5]],
    "n_train_candidates": len(cands),
}
with open(os.path.join(OUT_DIR, "reference_frame.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print("-> reference_frame.json")
print("\n完成!")
