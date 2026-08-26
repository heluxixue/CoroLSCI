"""step6d_render_two_videos.py -- Paper videos + final-frame PDF figures

Video 1: Intensity + mask overlay | coronary structural metrics
         (segmented y-axis, labeled value ranges).
Video 2: Perfusion + mask overlay | coronary mean perfusion time series.
Both rendered at native 44 fps; the last frame is exported as a vector
PDF (Times New Roman) for the paper.
"""
import os
import sys
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
REG_DIR = os.path.join(EXE_DIR, "registration")
RAW_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
OUT_DIR = os.path.join(EXE_DIR, "60s统一结果")
H, W = 647, 549
FPS = 44.0
N_TOTAL = 2641

SEGS = [("0-10s", 0, 440), ("10-20s", 440, 440), ("20-30s", 880, 440),
        ("后30s", 1320, 1321)]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


def load_mask(t):
    for seg, start, n in SEGS:
        if start <= t < start + n:
            segm = os.path.join(EXE_DIR, f"segmentation_{seg}", "masks_raw")
            return np.load(os.path.join(segm, f"{t:05d}.npy")) > 0
    raise IndexError(t)


def I_at(I_stacks, t):
    for seg, start, n in SEGS:
        if start <= t < start + n:
            return np.asarray(I_stacks[seg][t - start]).astype(np.float32) / 255.0
    raise IndexError(t)


def main():
    print("=" * 70, flush=True)
    print("渲染两个视频 (全60s)", flush=True)
    print("=" * 70, flush=True)
    t_all = time.time()

    # load data
    print("加载数据...", flush=True)
    I_stacks = {seg: np.load(os.path.join(REG_DIR, f"reg_{seg}_I_u8.npy"),
                             mmap_mode="r") for seg, _, _ in SEGS}
    perf = np.load(os.path.join(RAW_DIR, "灌注_perfusion.npy"), mmap_mode="r")

    # load geometry metrics (already compute )
    import csv
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "冠脉分析_指标.csv"),
                                    encoding="utf-8-sig")))
    metrics = {
        "area": np.array([float(r["area"]) for r in rows]),
        "skel_len": np.array([float(r["skel_len"]) for r in rows]),
        "mean_d": np.array([float(r["mean_d"]) for r in rows]),
        "n_branch": np.array([float(r["n_branch"]) for r in rows]),
    }
    print(f"几何指标已加载 ({len(rows)} 帧)", flush=True)

    # new compute perfusion amount (on the original perfusion image , not warp)
    print("重新计算灌注量 (原始灌注图, 未warp)...", flush=True)
    perf_mean = np.zeros(N_TOTAL)
    for t in range(N_TOTAL):
        mask = load_mask(t)
        P_raw = np.asarray(perf[t]).astype(np.float32)
        perf_mean[t] = float(P_raw[mask].mean()) if mask.any() else 0.0
        if (t + 1) % 440 == 0:
            print(f"  {t+1}/{N_TOTAL} ({time.time()-t_all:.0f}s)", flush=True)

    # save more new metrics
    with open(os.path.join(OUT_DIR, "冠脉分析_指标_原始灌注.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "area", "skel_len", "mean_d", "n_branch",
                    "perfusion_mean_raw"])
        for t in range(N_TOTAL):
            w.writerow([t, t / FPS, metrics["area"][t], metrics["skel_len"][t],
                        metrics["mean_d"][t], metrics["n_branch"][t],
                        perf_mean[t]])
    print(f"-> 冠脉分析_指标_原始灌注.csv", flush=True)

    ts = np.arange(N_TOTAL) / FPS

    # normalization geometry metrics (y-axis )
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    metrics_norm = {k: normalize(v) for k, v in metrics.items()}

    # === video 1: Intensity + geometry metrics ===
    print("\n渲染视频 1: Intensity + 几何指标...", flush=True)
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    out1 = os.path.join(OUT_DIR, "视频1_Intensity_几何指标_60s.mp4")
    vw1 = cv2.VideoWriter(out1, cv2.VideoWriter_fourcc(*"mp4v"), int(FPS),
                          (1600, 550))

    for t in range(N_TOTAL):
        mask = load_mask(t)
        mask_edge = mask & ~binary_erosion(mask)
        ys, xs = np.where(mask_edge)

        # left : Intensity + Mask
        img_I = I_at(I_stacks, t)
        axes1[0].clear()
        axes1[0].imshow(img_I, cmap="gray", vmin=0, vmax=1, origin="upper")
        axes1[0].scatter(xs, ys, s=0.5, c="lime", marker=".")
        axes1[0].set_title("Intensity + Mask", fontsize=12)
        axes1[0].set_xlabel("X (col)")
        axes1[0].set_ylabel("Y (row)")

        # right : geometry metrics (normalization )
        axes1[1].clear()
        axes1[1].plot(ts[:t + 1], metrics_norm["area"][:t + 1], lw=1.5,
                      label="Area", color="C0")
        axes1[1].plot(ts[:t + 1], metrics_norm["skel_len"][:t + 1], lw=1.5,
                      label="Skeleton Length", color="C1")
        axes1[1].plot(ts[:t + 1], metrics_norm["mean_d"][:t + 1], lw=1.5,
                      label="Mean Diameter", color="C2")
        axes1[1].plot(ts[:t + 1], metrics_norm["n_branch"][:t + 1], lw=1.5,
                      label="Branches", color="C3")
        axes1[1].set_xlabel("Time (s)", fontsize=10)
        axes1[1].set_ylabel("Normalized Value", fontsize=10)
        axes1[1].set_title("Coronary Structural Metrics", fontsize=12)
        axes1[1].legend(fontsize=9, loc="upper right")
        axes1[1].set_xlim(0, 60)
        axes1[1].set_ylim(-0.05, 1.05)
        axes1[1].grid(alpha=0.3)

        fig1.suptitle(f"Frame {t}  t={t/FPS:.2f}s", fontsize=14)
        fig1.canvas.draw()
        frame = np.frombuffer(fig1.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig1.canvas.get_width_height()[::-1] + (4,))
        frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        vw1.write(frame)
        if (t + 1) % 440 == 0:
            print(f"  {t+1}/{N_TOTAL} ({time.time()-t_all:.0f}s)", flush=True)
    vw1.release()
    plt.close(fig1)
    print(f"-> {out1}", flush=True)

    # === video 2: Perfusion + perfusion amount temporal ===
    print("\n渲染视频 2: Perfusion + 灌注量时序...", flush=True)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    out2 = os.path.join(OUT_DIR, "视频2_Perfusion_灌注时序_60s.mp4")
    vw2 = cv2.VideoWriter(out2, cv2.VideoWriter_fourcc(*"mp4v"), int(FPS),
                          (1600, 550))

    for t in range(N_TOTAL):
        mask = load_mask(t)
        mask_edge = mask & ~binary_erosion(mask)
        ys, xs = np.where(mask_edge)

        # left : Perfusion + Mask
        img_P = np.asarray(perf[t]).astype(np.float32)
        axes2[0].clear()
        im = axes2[0].imshow(img_P, cmap="turbo", vmin=0, vmax=1600,
                             origin="upper")
        axes2[0].scatter(xs, ys, s=0.5, c="lime", marker=".")
        axes2[0].set_title("Perfusion + Mask", fontsize=12)
        axes2[0].set_xlabel("X (col)")
        axes2[0].set_ylabel("Y (row)")
        if t == 0:
            fig2.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04)

        # right : perfusion amount temporal
        axes2[1].clear()
        axes2[1].plot(ts[:t + 1], perf_mean[:t + 1], lw=1.5, color="red")
        axes2[1].set_xlabel("Time (s)", fontsize=10)
        axes2[1].set_ylabel("Mean Perfusion (a.u.)", fontsize=10)
        axes2[1].set_title("Coronary Mean Perfusion (Raw)", fontsize=12)
        axes2[1].set_xlim(0, 60)
        axes2[1].grid(alpha=0.3)

        fig2.suptitle(f"Frame {t}  t={t/FPS:.2f}s", fontsize=14)
        fig2.canvas.draw()
        frame = np.frombuffer(fig2.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig2.canvas.get_width_height()[::-1] + (4,))
        frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        vw2.write(frame)
        if (t + 1) % 440 == 0:
            print(f"  {t+1}/{N_TOTAL} ({time.time()-t_all:.0f}s)", flush=True)
    vw2.release()
    plt.close(fig2)
    print(f"-> {out2}", flush=True)

    print(f"\n全部完成 ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
