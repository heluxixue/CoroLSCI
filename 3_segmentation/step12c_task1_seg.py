"""step12c_task1_seg.py -- Full-60s segmentation inference

Runs the adopted segmentation model (ResUNet, Intensity, 3-seed ensemble)
over all 2641 frames, producing per-frame masks and structural metrics.
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_common import (ResUNet, PAD_H, PAD_W)
from step7_train_gated import GatedFusionNet

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧_预处理")
EXE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
SEG_BASE = os.path.join(EXE_DIR, "segmentation")

H_RAW, W_RAW = 647, 549


def build_channels(images, config):
    I, V, C = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    if config == "I":
        return torch.cat([I, I, I], 1)
    if config == "IV":
        return torch.cat([I, V, V], 1)
    if config == "IC":
        return torch.cat([I, C, C], 1)
    return images  # IVC


# ============================================================
# structure metrics (all in raw frame mask on )
# ============================================================
def structural_metrics(mask):
    from scipy.ndimage import (distance_transform_edt, label, convolve,
                               binary_dilation)
    from skimage.morphology import skeletonize
    if mask.sum() == 0:
        return {"area": 0, "skel_len": 0, "mean_d": 0.0, "min_d": 0.0,
                "n_branch": 0, "branch_pts": [], "n_comp": 0, "n_break": 0,
                "tortuosity": 0.0}
    area = int(mask.sum())
    skel = skeletonize(mask)
    skel_len = int(skel.sum())

    # local diameter = skeleton point place 2×distance off
    dt = distance_transform_edt(mask)
    d_skel = dt[skel]
    mean_d = float(np.mean(d_skel)) * 2
    min_d = float(np.min(d_skel)) * 2

    # part cross point : 8neighborhood ≥3 skeleton
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    nbr = convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    branch_pts = [(int(y), int(x)) for y, x in zip(*np.where((skel & (nbr >= 3))))]
    n_branch = len(branch_pts)

    # connected component (mask and skeleton )
    _, n_comp = label(mask)
    lab_skel, n_comp_skel = label(skel)
    # point = skeleton connected component count - 1 ( ; as 1)
    n_break = max(n_comp_skel - 1, 0)

    # : each skeleton connected component L / point between distance off (by length weighted mean )
    torts, wts = [], []
    for i in range(1, n_comp_skel + 1):
        comp = lab_skel == i
        L = float(comp.sum())
        if L < 10:
            continue
        ys, xs = np.where(comp)
        # point = the connected component inside nbr==1 skeleton point ; take most two point distance off
        eps = np.where(comp & (nbr == 1))
        if len(eps[0]) >= 2:
            p0 = np.array([eps[0][0], eps[1][0]], dtype=np.float64)
            p1 = np.array([eps[0][-1], eps[1][-1]], dtype=np.float64)
            C = max(np.linalg.norm(p0 - p1), 1.0)
        else:
            C = max(np.linalg.norm([ys.max() - ys.min(), xs.max() - xs.min()]), 1.0)
        torts.append(L / C)
        wts.append(L)
    tortuosity = float(np.average(torts, weights=wts)) if torts else 0.0

    return {"area": area, "skel_len": skel_len, "mean_d": mean_d, "min_d": min_d,
            "n_branch": n_branch, "branch_pts": branch_pts, "n_comp": n_comp,
            "n_break": n_break, "tortuosity": tortuosity}


def run(model_path, config, start, n, segname, batch, device):
    print(f"\n{'='*60}\n任务一 {segname} 结构量化 (模型: {os.path.basename(model_path)})\n{'='*60}",
          flush=True)
    out_dir = os.path.join(EXE_DIR, f"segmentation_{segname}")
    mask_dir = os.path.join(out_dir, "masks_raw")
    prob_dir = os.path.join(out_dir, "probability_maps")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)

    if config == "gated":
        model = GatedFusionNet().to(device)
    else:
        model = ResUNet().to(device)
    model.load_state_dict(torch.load(os.path.join(SEG_BASE, model_path),
                                     map_location=device))
    model.eval()
    print(f"  参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    I = np.load(os.path.join(PREP_DIR, "光强_intensity_p1p99.npy"), mmap_mode="r")
    V = np.load(os.path.join(PREP_DIR, "方差_variance_p1p99.npy"), mmap_mode="r")
    C = np.load(os.path.join(PREP_DIR, "散斑对比度_contrast_p1p99.npy"), mmap_mode="r")

    rows, branch_pos = [], {}
    t0 = time.time()
    for t0f in range(0, n, batch):
        fids = [start + t for t in range(t0f, min(t0f + batch, n))]
        x = np.zeros((len(fids), 3, PAD_H, PAD_W), dtype=np.float32)
        for b, fi in enumerate(fids):
            x[b, 0, :H_RAW, :W_RAW] = I[fi]
            x[b, 1, :H_RAW, :W_RAW] = V[fi]
            x[b, 2, :H_RAW, :W_RAW] = C[fi]
        with torch.no_grad(), torch.amp.autocast("cuda"):
            xt = build_channels(torch.from_numpy(x).to(device), config)
            if config == "gated":
                probs = model(xt)[0].float()
            else:
                probs = torch.sigmoid(model(xt).float())
            probs = probs.cpu().numpy()[:, 0, :H_RAW, :W_RAW]

        for b, fi in enumerate(fids):
            p = probs[b]
            m = p > 0.5
            met = structural_metrics(m)
            np.save(os.path.join(mask_dir, f"{fi:05d}.npy"), m)
            np.save(os.path.join(prob_dir, f"{fi:05d}.npy"), p.astype(np.float32))
            branch_pos[str(fi)] = [list(pt) for pt in met.pop("branch_pts")]
            rows.append({"frame_id": fi, "timestamp_s": round(fi / 44.0, 4), **met})
        if (t0f + batch) % 88 == 0:
            print(f"  {min(t0f + batch, n)}/{n} | {time.time()-t0:.0f}s", flush=True)

    csv_path = os.path.join(out_dir, "structural_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "branch_positions.json"), "w", encoding="utf-8") as f:
        json.dump(branch_pos, f, ensure_ascii=False)

    # temporal image (area /skeleton length /mean diameter /part cross count / )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    frames = [r["frame_id"] for r in rows]
    ts = [r["timestamp_s"] for r in rows]
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for ax, key, label in [
            (axes[0], "area", "Mask面积 (px)"),
            (axes[1], "skel_len", "骨架总长 (px)"),
            (axes[2], "mean_d", "平均直径 (px)"),
            (axes[3], "n_branch", "分叉点数"),
            (axes[4], "tortuosity", "迂曲度")]:
        ax.plot(ts, [r[key] for r in rows], lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("时间 (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "结构指标时序图.png"), dpi=110)
    plt.close()

    print(f"  面积 mean={np.mean([r['area'] for r in rows]):.0f}px  "
          f"骨架长 mean={np.mean([r['skel_len'] for r in rows]):.0f}px  "
          f"直径 mean={np.mean([r['mean_d'] for r in rows]):.2f}px  "
          f"分叉 mean={np.mean([r['n_branch'] for r in rows]):.1f}", flush=True)
    print(f"  -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="IVC",
                    choices=["I", "IV", "IC", "IVC", "gated"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=440)
    ap.add_argument("--segname", default="0-10s")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    run(args.model, args.config, args.start, args.n, args.segname, args.batch,
        device)
