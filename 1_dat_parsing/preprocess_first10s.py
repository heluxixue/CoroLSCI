"""preprocess_first10s.py -- Preprocessing for the first 10 s

Per-frame percentile normalization and quality checks for the first-10s
development segment (0-439).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "前10s预处理")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_FPS = 44.0
DURATION_SEC = 10.0
FRAMES = int(DURATION_SEC * SRC_FPS)  # 440
H, W = 647, 549

# ============================================================
# 1. load before 10 s data
# ============================================================
print("1. 加载前 10 秒 NPY 数据...")
variance = np.load(os.path.join(NPY_DIR, "方差_variance.npy"), mmap_mode="r")[:FRAMES]
intensity = np.load(os.path.join(NPY_DIR, "光强_intensity.npy"), mmap_mode="r")[:FRAMES]
contrast = np.load(os.path.join(NPY_DIR, "散斑对比度_contrast.npy"), mmap_mode="r")[:FRAMES]
perfusion = np.load(os.path.join(NPY_DIR, "灌注_perfusion.npy"), mmap_mode="r")[:FRAMES]
print(f"   形状: I{intensity.shape} V{variance.shape} C{contrast.shape} P{perfusion.shape}")

# ============================================================
# 2. heart region Mask（ Intensity value ）
# ============================================================
print("2. 构建心脏区域 Mask...")
# use No. one frame image Otsu value segmentation before
from skimage.filters import threshold_otsu

intensity_mean = np.mean(intensity, axis=0)
thresh = threshold_otsu(intensity_mean)
heart_mask = intensity_mean > thresh
#
heart_mask = ndimage.binary_opening(heart_mask, structure=np.ones((5, 5)))
heart_mask = ndimage.binary_closing(heart_mask, structure=np.ones((7, 7)))
# most large connected component
labels, n = ndimage.label(heart_mask)
if n > 0:
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    heart_mask = labels == (np.argmax(sizes) + 1)
print(f"   心脏区域像素占比: {heart_mask.mean() * 100:.1f}%")

# ============================================================
# 3. three structure modality normalization （planning 4.2 node ）
# Intensity / Variance / Contrast each self truncate + percentile normalization
# ============================================================
print("3. 三个结构模态独立归一化...")

norm_params = {}

def compute_norm_params(data_3d, mask, name, clip_percentile=(0.5, 99.5)):
    """ truncate + min-max normalization parameters """
    # in heart region inside statistics
    masked_vals = data_3d[:, mask]
    lo = float(np.percentile(masked_vals, clip_percentile[0]))
    hi = float(np.percentile(masked_vals, clip_percentile[1]))
    if hi <= lo:
        hi = lo + 1.0
    params = {
        "clip_low": lo,
        "clip_high": hi,
        "method": f"clip_p{clip_percentile[0]}_p{clip_percentile[1]}_then_minmax",
        "value_range": [lo, hi],
    }
    print(f"   {name}: clip [{lo:.4f}, {hi:.4f}], then → [0, 1]")
    return params

def apply_norm(data_3d, params):
    clipped = np.clip(data_3d, params["clip_low"], params["clip_high"])
    return (clipped - params["clip_low"]) / (params["clip_high"] - params["clip_low"])

# Intensity normalization
norm_params["intensity"] = compute_norm_params(intensity, heart_mask, "Intensity")
intensity_norm = apply_norm(intensity, norm_params["intensity"])

# Variance normalization
norm_params["variance"] = compute_norm_params(variance, heart_mask, "Variance")
variance_norm = apply_norm(variance, norm_params["variance"])

# Contrast normalization
norm_params["contrast"] = compute_norm_params(contrast, heart_mask, "Contrast")
contrast_norm = apply_norm(contrast, norm_params["contrast"])

# Perfusion raw value ，not normalization （planning 3.2 node ）
print("   Perfusion: 保留原始定量值，不归一化")

# ============================================================
# 4. cardiac cycle （planning 4.3 node ）
# Intensity modality frame between ， motion signal no.
# ============================================================
print("4. 心动周期识别...")

# 4.1 compute heart region per-frame motion signal no.
# make use Intensity heart region frame between MSE as
motion_signal = np.zeros(FRAMES)
for t in range(1, FRAMES):
    diff = intensity_norm[t, heart_mask] - intensity_norm[t - 1, heart_mask]
    motion_signal[t] = float(np.mean(diff ** 2))
motion_signal[0] = motion_signal[1]

# 4.2 smoothing motion signal no.
motion_smooth = ndimage.gaussian_filter1d(motion_signal, sigma=2.0)

# 4.3 self heart rate cycle
# value
motion_detrend = motion_smooth - np.mean(motion_smooth)
# self
acf = np.correlate(motion_detrend, motion_detrend, mode="full")
acf = acf[acf.size // 2:]  #  take
acf /= acf[0]  # normalization

# No. one value （ except ）
min_period_samples = int(0.3 * SRC_FPS)   # most small heart rate ≈ 200 bpm → 0.3s
max_period_samples = int(2.0 * SRC_FPS)   # most large heart rate ≈ 30 bpm → 2.0s
search_range = slice(min_period_samples, min(max_period_samples, FRAMES - 1))
peaks, properties = signal.find_peaks(acf[search_range], height=0.1, distance=min_period_samples)
if len(peaks) > 0:
    best_peak = peaks[np.argmax(properties["peak_heights"])]
    cycle_length_samples = best_peak + min_period_samples
    heart_rate_bpm = 60 * SRC_FPS / cycle_length_samples
else:
    # ：default ~1s cycle
    cycle_length_samples = int(SRC_FPS)
    heart_rate_bpm = 60 * SRC_FPS / cycle_length_samples
    print("   ⚠ 未检测到显著周期峰值，使用默认 1s 周期")

print(f"   估计心动周期: {cycle_length_samples} 帧 ({cycle_length_samples / SRC_FPS:.3f} s)")
print(f"   估计心率: {heart_rate_bpm:.1f} bpm")

# 4.4 cycle border boundary （end-diastole ≈ most small point ）
# with motion signal no. local most small value as cycle border boundary
min_distance = int(cycle_length_samples * 0.6)  # most small between distance
minima_idx, _ = signal.find_peaks(-motion_smooth, distance=min_distance, prominence=0.05 * np.std(motion_smooth))

# cycle /phase
cycle_labels = np.full(FRAMES, -1, dtype=int)
phase_labels = np.zeros(FRAMES, dtype=float)

if len(minima_idx) >= 2:
    for c in range(len(minima_idx) - 1):
        start, end = minima_idx[c], minima_idx[c + 1]
        cycle_labels[start:end] = c
        # phase normalization [0, 1)，0 = end-diastole
        for t in range(start, end):
            phase_labels[t] = (t - start) / (end - start)
    # place most after one not cycle
    cycle_labels[minima_idx[-1]:] = len(minima_idx) - 1
    for t in range(minima_idx[-1], FRAMES):
        phase_labels[t] = (t - minima_idx[-1]) / (FRAMES - minima_idx[-1])
    num_cycles = len(minima_idx) - 1
else:
    # ：etc. part cycle
    num_cycles = max(1, int(FRAMES / cycle_length_samples))
    for c in range(num_cycles):
        start = int(c * FRAMES / num_cycles)
        end = int((c + 1) * FRAMES / num_cycles)
        cycle_labels[start:end] = c
        for t in range(start, end):
            phase_labels[t] = (t - start) / (end - start)
    print("   ⚠ 未检测到足够周期边界，使用均匀划分")

print(f"   识别到 {num_cycles} 个心动周期")

# ============================================================
# 5. save results
# ============================================================
print("5. 保存结果...")

# 5.1 normalization after four modality NPZ
np.savez_compressed(
    os.path.join(OUT_DIR, "前10s_四模态.npz"),
    intensity=intensity_norm.astype(np.float32),
    variance=variance_norm.astype(np.float32),
    contrast=contrast_norm.astype(np.float32),
    perfusion=perfusion.astype(np.float32),       # raw value
    intensity_raw=intensity.astype(np.float64),    #  raw this
    variance_raw=variance.astype(np.float64),
    contrast_raw=contrast.astype(np.float64),
)

# 5.2 normalization parameters
norm_params["heart_mask_pixel_count"] = int(heart_mask.sum())
norm_params["heart_mask_ratio"] = float(heart_mask.mean())
norm_params["source_fps"] = SRC_FPS
norm_params["duration_seconds"] = DURATION_SEC
norm_params["total_frames"] = FRAMES
with open(os.path.join(OUT_DIR, "前10s_归一化参数.json"), "w", encoding="utf-8") as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

# 5.3 cardiac cycle CSV
import csv
csv_path = os.path.join(OUT_DIR, "前10s_心动周期.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["帧号", "时间_秒", "运动信号", "心动周期", "归一化相位", "周期内帧号"])
    for t in range(FRAMES):
        w.writerow([
            t,
            f"{t / SRC_FPS:.4f}",
            f"{motion_smooth[t]:.8f}",
            cycle_labels[t],
            f"{phase_labels[t]:.4f}",
            int(phase_labels[t] * (cycle_length_samples if cycle_labels[t] >= 0 else 0)),
        ])
print(f"   CSV: {csv_path}")

# 5.4 save heart Mask
np.save(os.path.join(OUT_DIR, "心脏区域_mask.npy"), heart_mask)

# ============================================================
# 6. generate preprocessing report image
# ============================================================
print("6. 生成预处理报告图...")

fig = plt.figure(figsize=(16, 12))

# A. left on ：heart region Mask
ax_a = fig.add_subplot(3, 3, 1)
ax_a.imshow(intensity_mean, cmap="gray")
ax_a.contour(heart_mask, colors="#00ff00", linewidths=0.8)
ax_a.set_title("心脏区域 Mask（绿色轮廓）")
ax_a.set_xlabel("X"); ax_a.set_ylabel("Y")

# B. in on ：normalization before after comparison （Intensity frame ）
ax_b = fig.add_subplot(3, 3, 2)
frame_show = 220  # No. 5 s
ax_b.hist(intensity[frame_show, heart_mask].ravel(), bins=100, alpha=0.6, label="原始", color="gray")
ax_b.hist(intensity_norm[frame_show, heart_mask].ravel() * 100, bins=100, alpha=0.6,
          label="归一化 (×100)", color="blue")
ax_b.set_title(f"Intensity 归一化前后 (第{frame_show}帧, 5.0s)")
ax_b.legend(); ax_b.set_xlabel("值")

# C. right on ：three structure modality normalization after same one frame
for idx, (data, name, cmap) in enumerate([
    (intensity_norm, "Intensity", "gray"),
    (variance_norm, "Variance", "coolwarm"),
    (contrast_norm, "Contrast", "coolwarm"),
]):
    ax = fig.add_subplot(3, 3, 3 + idx)
    ax.imshow(data[frame_show], cmap=cmap, origin="upper")
    ax.set_title(f"{name} 归一化后")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

# D. left in ：motion signal no. + cycle border boundary
ax_d = fig.add_subplot(3, 1, 2)
time_axis = np.arange(FRAMES) / SRC_FPS
ax_d.plot(time_axis, motion_smooth, color="#333333", linewidth=1.0, label="Intensity motion signal no. （frame between MSE）")
if len(minima_idx) >= 2:
    for m in minima_idx:
        ax_d.axvline(m / SRC_FPS, color="#d62728", alpha=0.5, linewidth=1.0, linestyle="--")
ax_d.set_ylabel("运动幅度")
ax_d.set_title("心动周期识别（红色虚线 = 周期边界/舒张末期）")
ax_d.legend(loc="upper right")
ax_d.grid(alpha=0.25)

# E. ：normalization after four modality frame
frame_indices = [0, 110, 220, 330, 439]  # 0s, 2.5s, 5s, 7.5s, 10s
for col, fi in enumerate(frame_indices):
    # Intensity (row 1)
    ax = fig.add_subplot(4, 5, 0 * 5 + col + 1)
    ax.imshow(intensity_norm[fi], cmap="gray", origin="upper")
    if col == 0: ax.set_ylabel("Intensity")
    ax.set_title(f"{fi/SRC_FPS:.1f}s"); ax.set_xticks([]); ax.set_yticks([])
    # Variance (row 2)
    ax = fig.add_subplot(4, 5, 1 * 5 + col + 1)
    ax.imshow(variance_norm[fi], cmap="coolwarm", origin="upper")
    if col == 0: ax.set_ylabel("Variance")
    ax.set_xticks([]); ax.set_yticks([])
    # Contrast (row 3)
    ax = fig.add_subplot(4, 5, 2 * 5 + col + 1)
    ax.imshow(contrast_norm[fi], cmap="coolwarm", origin="upper")
    if col == 0: ax.set_ylabel("Contrast")
    ax.set_xticks([]); ax.set_yticks([])
    # Perfusion (row 4, raw value ，turbo)
    ax = fig.add_subplot(4, 5, 3 * 5 + col + 1)
    im = ax.imshow(perfusion[fi], cmap="turbo", vmin=0, vmax=1600, origin="upper")
    if col == 0: ax.set_ylabel("Perfusion")
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(
    f"前 10 秒预处理报告 | 帧率 {SRC_FPS} Hz | "
    f"心动周期 ~{cycle_length_samples / SRC_FPS:.2f}s | 心率 ~{heart_rate_bpm:.0f} bpm",
    fontsize=14, y=1.01,
)
fig.tight_layout()
report_path = os.path.join(OUT_DIR, "前10s_预处理报告.png")
fig.savefig(report_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"   报告图: {report_path}")

print("\n===== 预处理完成 =====")
print(f"输出目录: {OUT_DIR}")
print(f"  前10s_四模态.npz        → 归一化 I/V/C + 原始 P + 原始副本")
print(f"  前10s_归一化参数.json    → 截断/归一化参数（可复现）")
print(f"  前10s_心动周期.csv       → 帧级周期/相位标签")
print(f"  心脏区域_mask.npy        → 有效心脏区域")
print(f"  前10s_预处理报告.png      → 综合可视化")
