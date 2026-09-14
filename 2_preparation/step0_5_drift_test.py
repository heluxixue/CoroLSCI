"""step0_5_drift_test.py -- Global drift test

Estimates whole-field drift over the 60 s acquisition and derives the
heart_mask (region of interest) used by all later stages.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第0.5步_漂移检验")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_FPS = 44.0
TOTAL = 2641
SPLIT = 1320  # 30s part boundary
H, W = 647, 549

# ============================================================
# 0. load (memmap)
# ============================================================
print("=" * 60)
print("0. 加载数据")
print("=" * 60)
I = np.load(os.path.join(RAW_DIR, "光强_intensity.npy"), mmap_mode="r")
V = np.load(os.path.join(RAW_DIR, "方差_variance.npy"), mmap_mode="r")
C = np.load(os.path.join(RAW_DIR, "散斑对比度_contrast.npy"), mmap_mode="r")
print(f"  I: {I.shape} {I.dtype}")

# heart region mask : use full 60smean intensity image Otsu
print("  计算全序列平均强度图与心脏掩膜...")
I_mean_all = np.zeros((H, W), dtype=np.float64)
for t in range(0, TOTAL, 50):
    I_mean_all += I[t].astype(np.float64)
I_mean_all /= (TOTAL // 50 + 1)

from skimage.filters import threshold_otsu
thr = threshold_otsu(I_mean_all)
heart_mask = I_mean_all > thr
#
from scipy import ndimage
heart_mask = ndimage.binary_opening(heart_mask, iterations=3)
heart_mask = ndimage.binary_closing(heart_mask, iterations=3)
print(f"  心脏掩膜: {heart_mask.sum()} px ({heart_mask.mean()*100:.1f}%)")

# ============================================================
# 1. per-frame statistics amount
# ============================================================
print("=" * 60)
print("1. 逐帧统计量 (2641帧)")
print("=" * 60)

def frame_stats(arr, mask=None):
    """each frame p1/p50/p99 + mean"""
    T = arr.shape[0]
    out = np.zeros((T, 4), dtype=np.float64)
    for t in range(T):
        f = arr[t]
        if mask is not None:
            f = f[mask]
        fin = f[np.isfinite(f)]
        out[t] = [np.percentile(fin, 1), np.percentile(fin, 50), np.percentile(fin, 99), fin.mean()]
    return out

print("  Intensity...")
I_stats = frame_stats(I)
print("  Variance...")
V_stats = frame_stats(V)
print("  Contrast...")
C_stats = frame_stats(C)

# frame between (heart region )
print("  帧间运动幅度 (心脏区域)...")
motion = np.zeros(TOTAL)
prev = I[0][heart_mask].astype(np.float64)
for t in range(1, TOTAL):
    cur = I[t][heart_mask].astype(np.float64)
    motion[t] = np.mean(np.abs(cur - prev))
    prev = cur

time_arr = np.arange(TOTAL) / SRC_FPS

# ============================================================
# 2. before last-30s statistics
# ============================================================
print("=" * 60)
print("2. 0-30s vs 30-60s 统计比较")
print("=" * 60)

def compare(name, arr, stat_names):
    a, b = arr[:SPLIT], arr[SPLIT:]
    results = []
    print(f"  [{name}]")
    for i, stat_name in enumerate(stat_names):
        t_stat, p_val = stats.ttest_ind(a[:, i], b[:, i], equal_var=False)
        pooled = (a[:, i].var() + b[:, i].var()) / 2
        cohen_d = (b[:, i].mean() - a[:, i].mean()) / np.sqrt(pooled) if pooled > 0 else 0.0
        results.append((stat_name, a[:, i].mean(), b[:, i].mean(), cohen_d, p_val))
        print(f"    {stat_name}: 0-30s={a[:,i].mean():.4f}  30-60s={b[:,i].mean():.4f}  "
              f"Cohen d={cohen_d:+.3f}  p={p_val:.2e}")
    return results

compare("Intensity", I_stats, ["p1", "p50", "p99", "mean"])
compare("Variance", V_stats, ["p1", "p50", "p99", "mean"])
compare("Contrast", C_stats, ["p1", "p50", "p99", "mean"])
compare("Motion", motion.reshape(-1, 1), ["mean|dI|"])

# ============================================================
# 3. heart mask position /area random when between drift
# ============================================================
print("=" * 60)
print("3. 心脏区域位置/面积漂移 (每10s一段)")
print("=" * 60)

seg_len = int(10 * SRC_FPS)  # 440frame /segment
print(f"  {'段':>6s} {'面积(px)':>10s} {'质心Y':>8s} {'质心X':>8s}")
seg_areas, seg_cy, seg_cx = [], [], []
for seg in range(0, TOTAL, seg_len):
    sl = slice(seg, min(seg + seg_len, TOTAL))
    seg_mean = I[sl].mean(axis=0)
    seg_mask = seg_mean > threshold_otsu(seg_mean)
    seg_mask = ndimage.binary_opening(seg_mask, iterations=3)
    ys, xs = np.where(seg_mask)
    cy, cx = ys.mean(), xs.mean()
    seg_areas.append(seg_mask.sum()); seg_cy.append(cy); seg_cx.append(cx)
    print(f"  [{seg/SRC_FPS:4.0f}s] {seg_mask.sum():10d} {cy:8.1f} {cx:8.1f}")

# ============================================================
# 4. visualization
# ============================================================
print("=" * 60)
print("4. 可视化")
print("=" * 60)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(time_arr, I_stats[:, 2], color="#2166ac", lw=0.8, label="p99")
axes[0].plot(time_arr, I_stats[:, 1], color="#4393c3", lw=0.8, label="p50")
axes[0].plot(time_arr, I_stats[:, 0], color="#92c5de", lw=0.8, label="p1")
axes[0].axvline(30, color="r", ls="--", lw=1.5)
axes[0].text(30.5, I_stats[:, 2].max(), "30s 分界", color="r", fontsize=9)
axes[0].set_ylabel("Intensity")
axes[0].legend(ncol=3, fontsize=8)
axes[0].set_title("逐帧 Intensity 分位数 (p1/p50/p99)")
axes[0].grid(alpha=0.25)

axes[1].plot(time_arr, V_stats[:, 1], color="#1b7837", lw=0.8, label="V p50")
axes[1].plot(time_arr, C_stats[:, 1], color="#762a83", lw=0.8, label="C p50")
axes[1].axvline(30, color="r", ls="--", lw=1.5)
axes[1].set_ylabel("V / C 中位数")
axes[1].legend(fontsize=8)
axes[1].set_title("逐帧 Variance / Contrast 中位数")
axes[1].grid(alpha=0.25)

axes[2].plot(time_arr, motion, color="#b2182b", lw=0.8)
axes[2].axvline(30, color="r", ls="--", lw=1.5)
axes[2].set_ylabel("帧间运动幅度")
axes[2].set_title("帧间运动幅度 (心脏区域 |ΔI|)")
axes[2].grid(alpha=0.25)

# part segment statistics (each 5sone segment I p50 mean ± std)
seg5 = int(5 * SRC_FPS)
xs, ys, es = [], [], []
for seg in range(0, TOTAL, seg5):
    sl = slice(seg, min(seg + seg5, TOTAL))
    xs.append(seg / SRC_FPS + 2.5)
    ys.append(I_stats[sl, 1].mean())
    es.append(I_stats[sl, 1].std())
axes[3].errorbar(xs, ys, yerr=es, fmt="o-", color="#2166ac", markersize=4, capsize=2)
axes[3].axvline(30, color="r", ls="--", lw=1.5)
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("I p50 (5s段均值±std)")
axes[3].set_title("5秒段 Intensity p50 均值与波动 (漂移趋势)")
axes[3].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "漂移检验_趋势图.png"), dpi=150)
plt.close(fig)
print(f"  -> 漂移检验_趋势图.png")

# part image comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, arr, idx) in zip(axes, [("Intensity p50", I_stats, 1),
                                        ("Variance p50", V_stats, 1),
                                        ("Contrast p50", C_stats, 1)]):
    ax.hist(arr[:SPLIT, idx], bins=40, alpha=0.55, label="0-30s", color="#2166ac", density=True)
    ax.hist(arr[SPLIT:, idx], bins=40, alpha=0.55, label="30-60s", color="#b2182b", density=True)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "漂移检验_分布对比.png"), dpi=150)
plt.close(fig)
print(f"  -> 漂移检验_分布对比.png")

# save statistics results
np.savez(os.path.join(OUT_DIR, "漂移检验_统计量.npz"),
         I_stats=I_stats, V_stats=V_stats, C_stats=C_stats,
         motion=motion, seg_areas=seg_areas, seg_cy=seg_cy, seg_cx=seg_cx,
         heart_mask=heart_mask)
print(f"  -> 漂移检验_统计量.npz")
print("\n完成!")
