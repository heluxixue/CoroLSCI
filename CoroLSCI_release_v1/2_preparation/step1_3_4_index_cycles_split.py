"""step1_3_4_index_cycles_split.py -- Indexing, cardiac cycle detection, temporal split

Builds the global frame index, detects cardiac cycle boundaries per segment
(used by cycle-aware registration and phase analysis), and performs the
temporal 70/15/15 train/val/test split with a sealed test set.
"""
import os, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
MASK_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "200mask", "SegmentationClass")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_FPS = 44.0
TOTAL = 2641
LAST30_START = 1320
H, W = 647, 549

# ============================================================
# No. 1step : unified data index
# ============================================================
print("=" * 60)
print("第1步: 统一数据索引")
print("=" * 60)

MODALITY_FILES = {
    "intensity": "光强_intensity.npy",
    "variance": "方差_variance.npy",
    "contrast": "散斑对比度_contrast.npy",
    "perfusion": "灌注_perfusion.npy",
}

# 200 frame annotation file ( self SegmentationClass)
annotated = sorted(
    [f.replace(".png", "") for f in os.listdir(MASK_DIR) if f.endswith(".png")],
    key=int,
)
annotated_set = set(annotated)
print(f"  标注帧数: {len(annotated)}  范围: {annotated[0]}~{annotated[-1]}")

index_rows = []
for t in range(TOTAL):
    mask_path = os.path.join(MASK_DIR, f"{t}.png") if str(t) in annotated_set else ""
    index_rows.append({
        "frame_id": t,
        "timestamp_s": round(t / SRC_FPS, 6),
        "intensity_path": os.path.join(RAW_DIR, MODALITY_FILES["intensity"]),
        "variance_path": os.path.join(RAW_DIR, MODALITY_FILES["variance"]),
        "contrast_path": os.path.join(RAW_DIR, MODALITY_FILES["contrast"]),
        "perfusion_path": os.path.join(RAW_DIR, MODALITY_FILES["perfusion"]),
        "mask_path": mask_path,
    })

index_csv = os.path.join(OUT_DIR, "统一数据索引_2641帧.csv")
with open(index_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
    writer.writeheader()
    writer.writerows(index_rows)
print(f"  -> 统一数据索引_2641帧.csv ({len(index_rows)} 行)")

# ============================================================
# No. 3step : last-30s cardiac cycle detection
# ============================================================
print("=" * 60)
print("第3步: 后30s 心动周期检测")
print("=" * 60)

# heart region (use last-30s mean intensity image Otsu)
I = np.load(os.path.join(RAW_DIR, "光强_intensity.npy"), mmap_mode="r")
I_last30_mean = I[LAST30_START:].mean(axis=0).astype(np.float64)
thr = threshold_otsu(I_last30_mean)
heart_mask = I_last30_mean > thr
heart_mask = ndimage.binary_opening(heart_mask, iterations=3)
heart_mask = ndimage.binary_closing(heart_mask, iterations=3)
print(f"  后30s心脏掩膜: {heart_mask.sum()} px")

# frame between motion signal no. (last-30s , heart region )
T30 = TOTAL - LAST30_START
motion = np.zeros(T30)
prev = I[LAST30_START][heart_mask].astype(np.float64)
for t in range(1, T30):
    cur = I[LAST30_START + t][heart_mask].astype(np.float64)
    motion[t] = np.mean(np.abs(cur - prev))
    prev = cur

# smoothing + small value = cardiac cycle border boundary ( point )
from scipy.signal import savgol_filter
motion_sm = savgol_filter(motion, window_length=7, polyorder=2)
minima, _ = find_peaks(-motion_sm, distance=15, prominence=np.percentile(motion_sm, 60) * 0.15)

# border boundary place : ensure first
if len(minima) == 0:
    raise RuntimeError("未检测到心动周期")
cycle_starts = [0] + [int(m) for m in minima]
if cycle_starts[-1] != T30 - 1:
    cycle_starts.append(T30 - 1)

n_cycles = len(cycle_starts) - 1
cycle_lengths = np.diff(cycle_starts)
print(f"  检测到 {n_cycles} 个心动周期")
print(f"  周期长度: mean={cycle_lengths.mean():.1f}帧  std={cycle_lengths.std():.1f}帧  "
      f"range=[{cycle_lengths.min()}, {cycle_lengths.max()}]")
print(f"  心率估计: {SRC_FPS / cycle_lengths.mean() * 60:.0f} bpm")

# each frame -> cycle id
frame_cycle = np.zeros(T30, dtype=int)
for c in range(n_cycles):
    frame_cycle[cycle_starts[c]:cycle_starts[c + 1]] = c

# ============================================================
# No. 4step : when between groups part + 70/15/15 part
# ============================================================
print("=" * 60)
print("第4步: 时间组划分 + 数据集切分")
print("=" * 60)

CYCLES_PER_GROUP = 3
n_groups = int(np.ceil(n_cycles / CYCLES_PER_GROUP))
cycle_group = np.zeros(n_cycles, dtype=int)
for c in range(n_cycles):
    cycle_group[c] = min(c // CYCLES_PER_GROUP, n_groups - 1)
print(f"  {n_cycles} 周期 → {n_groups} 个时间组 (每组 {CYCLES_PER_GROUP} 周期)")

# each annotation frame cycle/group
annotated_frames = sorted(int(x) for x in annotated)
frame_info = {}
for fi in annotated_frames:
    t30 = fi - LAST30_START
    c = frame_cycle[t30]
    g = cycle_group[c]
    frame_info[fi] = (c, g)

# fixed seed part : by group part train/val/test (70/15/15)
rng = np.random.RandomState(20260816)
group_ids = np.arange(n_groups)
rng.shuffle(group_ids)

n_train_g = max(1, round(n_groups * 0.70))
n_val_g = max(1, round(n_groups * 0.15))
split_of_group = {}
for i, g in enumerate(group_ids):
    if i < n_train_g:
        split_of_group[g] = "train"
    elif i < n_train_g + n_val_g:
        split_of_group[g] = "val"
    else:
        split_of_group[g] = "test"

# statistics each frame count
counts = {"train": 0, "val": 0, "test": 0}
split_rows = []
for fi in annotated_frames:
    c, g = frame_info[fi]
    sp = split_of_group[g]
    counts[sp] += 1
    split_rows.append({
        "frame_id": fi,
        "timestamp_s": round(fi / SRC_FPS, 6),
        "cardiac_cycle_id": c,
        "temporal_group_id": g,
        "split": sp,
        "mask_path": os.path.join(MASK_DIR, f"{fi}.png"),
        "intensity_path": os.path.join(RAW_DIR, MODALITY_FILES["intensity"]),
        "variance_path": os.path.join(RAW_DIR, MODALITY_FILES["variance"]),
        "contrast_path": os.path.join(RAW_DIR, MODALITY_FILES["contrast"]),
        "perfusion_path": os.path.join(RAW_DIR, MODALITY_FILES["perfusion"]),
    })

split_csv = os.path.join(OUT_DIR, "dataset_split.csv")
with open(split_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
    writer.writeheader()
    writer.writerows(split_rows)

n_total = len(annotated_frames)
print(f"  划分结果: train={counts['train']} ({counts['train']/n_total*100:.0f}%)  "
      f"val={counts['val']} ({counts['val']/n_total*100:.0f}%)  "
      f"test={counts['test']} ({counts['test']/n_total*100:.0f}%)")

# validation : same group not (by ensure , place )
for g in range(n_groups):
    g_splits = set(row["split"] for row in split_rows if row["temporal_group_id"] == g)
    assert len(g_splits) <= 1, f"group {g} 跨集合!"

# when between check : test frame when between part
test_frames = sorted(r["frame_id"] for r in split_rows if r["split"] == "test")
print(f"  test 帧时间范围: {test_frames[0]/SRC_FPS:.1f}s ~ {test_frames[-1]/SRC_FPS:.1f}s "
      f"(散布于 {len(set(r['temporal_group_id'] for r in split_rows if r['split']=='test'))} 个时间组)")

# ============================================================
# visualization + report
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

t30 = np.arange(T30) / SRC_FPS + 30
axes[0].plot(t30, motion_sm, color="#2166ac", lw=0.8)
for cs in cycle_starts[:-1]:
    axes[0].axvline(30 + cs / SRC_FPS, color="g", alpha=0.4, lw=0.6)
axes[0].set_ylabel("Motion")
axes[0].set_title(f"后30s 运动信号与心动周期边界 ({n_cycles} 周期, 绿线=周期起点)")
axes[0].grid(alpha=0.25)

# annotation frame split
colors = {"train": "#2166ac", "val": "#f4a582", "test": "#b2182b"}
for row in split_rows:
    axes[1].scatter(row["frame_id"] / SRC_FPS, row["temporal_group_id"],
                    c=colors[row["split"]], s=18, alpha=0.8)
axes[1].set_ylabel("Temporal group")
axes[1].set_title("200标注帧的时间组分布 (蓝=train 橙=val 红=test)")
axes[1].grid(alpha=0.25)

axes[2].plot(t30, frame_cycle, color="#5a5a5a", lw=0.8)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Cycle id")
axes[2].set_title("逐帧心动周期编号")
axes[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "后30s_心动周期与划分.png"), dpi=150)
plt.close(fig)

report = f"""No. three version execution - No. 1/3/4step report
========================================
No. 1step unified data index : {TOTAL} frame → unified data index _2641frame .csv
No. 3step cardiac cycle : {n_cycles} cycle , length mean={cycle_lengths.mean():.1f}±{cycle_lengths.std():.1f} frame
     heart rate : {SRC_FPS / cycle_lengths.mean() * 60:.0f} bpm
No. 4step when between groups : each groups {CYCLES_PER_GROUP} cycle → {n_groups} groups
      part (fixed seed 20260816):
       train: {counts['train']} frame ({counts['train']/n_total*100:.0f}%)
       val: {counts['val']} frame ({counts['val']/n_total*100:.0f}%)
       test: {counts['test']} frame ({counts['test']/n_total*100:.0f}%)
     same when between groups not : already validation
     dataset_split.csv already fixed save
"""
report_path = os.path.join(OUT_DIR, "数据划分报告.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
print(f"  -> {report_path}")
print("完成!")
