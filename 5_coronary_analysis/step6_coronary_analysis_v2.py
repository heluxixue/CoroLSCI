"""step6_coronary_analysis_v2.py -- Coronary geometry + perfusion analysis (full 60 s)

Structural metrics from the adopted masks: area, skeleton length, mean
diameter (distance transform on skeleton), branch count.
Functional metric: mask-averaged perfusion time series computed directly
on the ORIGINAL perfusion stack (perfusion data is only sampled --
never used for registration training), 44 fps.
"""
import os
import sys
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reg_common import warp_image

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

# Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


def load_fields(config="I_maskw_s2"):
    return {seg: np.load(os.path.join(REG_DIR, f"fields_{seg}_{config}.npy"),
                         mmap_mode="r") for seg, _, _ in SEGS}


def field_at(fields, t):
    for seg, start, n in SEGS:
        if start <= t < start + n:
            return fields[seg][t - start].astype(np.float32)
    raise IndexError(t)


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


def compute_structural_metrics(mask):
    from skimage.morphology import skeletonize
    area = int(mask.sum())
    if area == 0:
        return {"area": 0, "skel_len": 0, "mean_d": 0, "n_branch": 0}
    skel = skeletonize(mask)
    skel_len = int(skel.sum())
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    mean_d = float(dist[skel].mean() * 2) if skel.any() else 0.0
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbors = convolve(skel.astype(int), kernel, mode="constant")
    n_branch = int(((neighbors > 2) & skel).sum())
    return {"area": area, "skel_len": skel_len, "mean_d": mean_d,
            "n_branch": n_branch}


def main():
    print("=" * 70, flush=True)
    print("冠脉几何指标 + 灌注量时序可视化 (改进版)", flush=True)
    print("=" * 70, flush=True)
    t_all = time.time()

    # load data
    print("加载数据...", flush=True)
    I_stacks = {seg: np.load(os.path.join(REG_DIR, f"reg_{seg}_I_u8.npy"),
                             mmap_mode="r") for seg, _, _ in SEGS}
    perf = np.load(os.path.join(RAW_DIR, "灌注_perfusion.npy"), mmap_mode="r")
    fields = load_fields()

    # compute
    print("预计算几何指标和灌注量...", flush=True)
    metrics = {"area": [], "skel_len": [], "mean_d": [], "n_branch": []}
    perf_mean = []
    for t in range(N_TOTAL):
        mask = load_mask(t)
        m = compute_structural_metrics(mask)
        for k in metrics:
            metrics[k].append(m[k])
        P_w = warp_image(np.asarray(perf[t]).astype(np.float32),
                         field_at(fields, t))
        perf_mean.append(float(P_w[mask].mean()) if mask.any() else 0.0)
        if (t + 1) % 440 == 0:
            print(f"  {t+1}/{N_TOTAL} ({time.time()-t_all:.0f}s)", flush=True)

    # save metrics
    import csv
    with open(os.path.join(OUT_DIR, "冠脉分析_指标.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "area", "skel_len", "mean_d", "n_branch",
                    "perfusion_mean"])
        for t in range(N_TOTAL):
            w.writerow([t, t / FPS, metrics["area"][t], metrics["skel_len"][t],
                        metrics["mean_d"][t], metrics["n_branch"][t],
                        perf_mean[t]])
    print(f"-> 冠脉分析_指标.csv", flush=True)

    # video render
    print("\n渲染视频...", flush=True)
    ts = np.arange(N_TOTAL) / FPS

    # : on line 2 image , down line 4+1 image (metrics each one )
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="white")
    # on line : 2
    gs_top = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.25,
                              wspace=0.15, left=0.05, right=0.95, top=0.95,
                              bottom=0.05)
    ax_img = fig.add_subplot(gs_top[0, 0])
    ax_perf = fig.add_subplot(gs_top[0, 1])

    # down line left : 4 image (2×2 grid )
    gs_struct = gs_top[1, 0].subgridspec(2, 2, hspace=0.4, wspace=0.3)
    ax_area = fig.add_subplot(gs_struct[0, 0])
    ax_skel = fig.add_subplot(gs_struct[0, 1])
    ax_diam = fig.add_subplot(gs_struct[1, 0])
    ax_branch = fig.add_subplot(gs_struct[1, 1])

    # down line right : perfusion amount temporal
    ax_perf_curve = fig.add_subplot(gs_top[1, 1])

    out_path = os.path.join(OUT_DIR, "冠脉分析_60s_v2.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), int(FPS),
                         (1600, 1000))

    for t in range(N_TOTAL):
        # on line left : Intensity + Mask
        img = I_at(I_stacks, t)
        mask = load_mask(t)
        ax_img.clear()
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
        # mask ( )
        mask_edge = mask & ~binary_erosion(mask)
        ys, xs = np.where(mask_edge)
        ax_img.scatter(xs, ys, s=0.5, c="lime", marker=".")
        ax_img.set_title(f"Intensity + Mask  t={t/FPS:.2f}s", fontsize=12)
        ax_img.set_xlabel("X (col)", fontsize=10)
        ax_img.set_ylabel("Y (row)", fontsize=10)

        # on line right : Perfusion + perfusion region mask (perfusion value > value )
        P_w = warp_image(np.asarray(perf[t]).astype(np.float32),
                         field_at(fields, t))
        ax_perf.clear()
        im = ax_perf.imshow(P_w, cmap="turbo", vmin=0, vmax=1600,
                            origin="upper")
        # perfusion region : perfusion value > 0 ( except no. region )
        perf_mask = P_w > 0
        perf_edge = perf_mask & ~binary_erosion(perf_mask)
        ys_p, xs_p = np.where(perf_edge)
        ax_perf.scatter(xs_p, ys_p, s=0.5, c="lime", marker=".")
        ax_perf.set_title("Perfusion + Valid Region", fontsize=12)
        ax_perf.set_xlabel("X (col)", fontsize=10)
        ax_perf.set_ylabel("Y (row)", fontsize=10)
        if t == 0:
            fig.colorbar(im, ax=ax_perf, fraction=0.046, pad=0.04)

        # down line left : 4 metrics image
        ax_area.clear()
        ax_area.plot(ts[:t + 1], metrics["area"][:t + 1], lw=1.5, color="C0")
        ax_area.set_title("Area (px)", fontsize=10)
        ax_area.set_xlim(0, 60)
        ax_area.grid(alpha=0.3)

        ax_skel.clear()
        ax_skel.plot(ts[:t + 1], metrics["skel_len"][:t + 1], lw=1.5, color="C1")
        ax_skel.set_title("Skeleton Length (px)", fontsize=10)
        ax_skel.set_xlim(0, 60)
        ax_skel.grid(alpha=0.3)

        ax_diam.clear()
        ax_diam.plot(ts[:t + 1], metrics["mean_d"][:t + 1], lw=1.5, color="C2")
        ax_diam.set_title("Mean Diameter (px)", fontsize=10)
        ax_diam.set_xlim(0, 60)
        ax_diam.grid(alpha=0.3)

        ax_branch.clear()
        ax_branch.plot(ts[:t + 1], metrics["n_branch"][:t + 1], lw=1.5, color="C3")
        ax_branch.set_title("Branches", fontsize=10)
        ax_branch.set_xlim(0, 60)
        ax_branch.grid(alpha=0.3)

        # down line right : perfusion amount temporal
        ax_perf_curve.clear()
        ax_perf_curve.plot(ts[:t + 1], perf_mean[:t + 1], lw=1.5, color="red")
        ax_perf_curve.set_xlabel("Time (s)", fontsize=10)
        ax_perf_curve.set_ylabel("Mean Perfusion (a.u.)", fontsize=10)
        ax_perf_curve.set_title("Coronary Mean Perfusion", fontsize=12)
        ax_perf_curve.set_xlim(0, 60)
        ax_perf_curve.grid(alpha=0.3)

        # render
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        vw.write(frame)
        if (t + 1) % 440 == 0:
            print(f"  {t+1}/{N_TOTAL} ({time.time()-t_all:.0f}s)", flush=True)
    vw.release()
    print(f"-> {out_path}", flush=True)

    # most after one frame PDF
    print("\n保存最后一帧 PDF...", flush=True)
    pdf_path = os.path.join(OUT_DIR, "冠脉分析_最后一帧_v2.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
    print(f"-> {pdf_path}", flush=True)

    plt.close(fig)
    print(f"\n完成 ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
