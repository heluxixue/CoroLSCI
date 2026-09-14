"""step12b_maskw_exp.py -- Soft-mask weighted DIS registration

ABANDONED PRELIMINARY WORK -- NOT part of the published indicator path.
This stage was evaluated during development and then abandoned (see
4_registration/README.md and paper Sec. II-F): brightness-matching
registration exhibits a residual-motion floor of ~6 px under speckle
decorrelation, and warping would resample the measurand itself (areas,
diameters, speckle statistics). The code is retained for auditability.
The published pipeline computes every indicator in original coordinates
without any registration stage.

Historical description: multiplies the DIS input by a coronary soft mask
(Gaussian-blurred prediction; sigma=2 after the innovation sweep) so the
flow is driven by coronary structure.
"""
import argparse, json, os, sys, time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reg_common import (H, W, dis_flow, apply_affine, compose, compose_affine_dense,
                        field_stats, ncc, warp_image)
from step9_10_registration import _init_affine_worker, _task_affine, _G
from scipy.ndimage import gaussian_filter

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
OUT_DIR = os.path.join(EXE_DIR, "registration")

SOFT_SIGMA = 4.0


# ============================================================
# module-level worker (Windows spawn not pickle closure )
# ============================================================
def _init_maskw_worker(mod_paths, cyc_bounds, loc_ref, M_all, anchors, temp_path,
                       mod, soft_path):
    import cv2
    cv2.setNumThreads(1)
    _G["imgs"] = {k: np.load(p, mmap_mode="r") for k, p in mod_paths.items()}
    _G["bounds"] = cyc_bounds
    _G["loc_ref"] = loc_ref
    _G["M"] = M_all
    _G["anchors"] = anchors
    _G["temp"] = np.load(temp_path, mmap_mode="r+")
    _G["mod"] = mod
    _G["soft"] = np.load(soft_path, mmap_mode="r")
    _G["ref_masked"] = (np.asarray(_G["imgs"][mod][loc_ref]).astype(np.float32)
                        / 255.0) * _G["soft"][loc_ref]


def _task_cycle_maskw(cyc_idx):
    """L2+L3 DIS, image multiply coronary soft mask (its residual and step9_10._task_cycle fully consistent )"""
    bounds = _G["bounds"]
    lo, hi = int(bounds[cyc_idx]), int(bounds[cyc_idx + 1])
    anchor = int(_G["anchors"][cyc_idx])
    mod = _G["mod"]
    temp = _G["temp"]
    loc_ref = _G["loc_ref"]
    soft = _G["soft"]

    anchor_base = apply_affine(np.asarray(_G["imgs"][mod][anchor]).astype(np.float32)
                               / 255.0, _G["M"][anchor])
    anchor_masked = anchor_base * soft[anchor]
    f2 = dis_flow(anchor_masked, _G["ref_masked"], preset="ULTRAFAST")
    f2 = dis_flow(anchor_masked, _G["ref_masked"], prev=f2, preset="MEDIUM")

    prev = np.zeros((H, W, 2), dtype=np.float32)
    prev_valid = False
    for t in range(anchor, hi):
        base = apply_affine(np.asarray(_G["imgs"][mod][t]).astype(np.float32) / 255.0,
                            _G["M"][t])
        img = base * soft[t]
        if not prev_valid:
            f1 = dis_flow(img, anchor_masked, preset="ULTRAFAST")
            f1 = dis_flow(img, anchor_masked, prev=f1, preset="MEDIUM")
        else:
            f1 = dis_flow(img, anchor_masked, prev=prev)
        temp[t] = f1.astype(np.float16)
        prev = f1
        prev_valid = True

    prev = np.zeros((H, W, 2), dtype=np.float32)
    prev_valid = False
    for t in range(anchor - 1, lo - 1, -1):
        base = apply_affine(np.asarray(_G["imgs"][mod][t]).astype(np.float32) / 255.0,
                            _G["M"][t])
        img = base * soft[t]
        if not prev_valid:
            f1 = dis_flow(img, anchor_masked, preset="ULTRAFAST")
            f1 = dis_flow(img, anchor_masked, prev=f1, preset="MEDIUM")
        else:
            f1 = dis_flow(img, anchor_masked, prev=prev)
        temp[t] = f1.astype(np.float16)
        prev = f1
        prev_valid = True

    temp[anchor] = np.zeros((H, W, 2), dtype=np.float16)
    return cyc_idx, f2


def build_soft_masks(start, n, segname, ref_global, loc_ref):
    """segment inside each frame mask + reference frame manual M_ref → soft mask stack (n+1, H, W)
    M_ref soft mask in loc_ref (reference frame in segment inside then the frame soft mask , in outside then )"""
    seg_masks = os.path.join(EXE_DIR, f"segmentation_{segname}", "masks_raw")
    path = os.path.join(OUT_DIR, f"softmasks_{segname}.npy")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n + 1, H, W))
    for t in range(n):
        m = np.load(os.path.join(seg_masks, f"{start + t:05d}.npy"))
        s = gaussian_filter(m.astype(np.float32), SOFT_SIGMA)
        out[t] = s / max(float(s.max()), 1e-6)
    data = np.load(os.path.join(EXE_DIR, "segmentation", "annotated_200.npz"))
    M_ref = data["masks"][[int(f) for f in data["frame_ids"]].index(ref_global)] > 0
    s = gaussian_filter(M_ref.astype(np.float32), SOFT_SIGMA)
    out[loc_ref] = s / max(float(s.max()), 1e-6)
    out.flush()
    return path


def run(start, n, segname, config, workers, reg_stack=None):
    print(f"\n{'='*60}\n受控实验 mask加权DIS {segname} (R-{config})  基线: 冻结R-I\n{'='*60}",
          flush=True)
    t_all = time.time()
    mod_paths = {"I": reg_stack or os.path.join(OUT_DIR, f"reg_{segname}_I_u8.npy")}
    with open(os.path.join(EXE_DIR, "reference", "reference_frame.json"),
              encoding="utf-8") as f:
        ref_global = int(json.load(f)["frame_id"])
    # reference frame in segment inside → use its local (last-30s : 2562 in 1320-2640 inside , u8stack this );
    # in segment outside → segment stack tail this (loc_ref = n)
    loc_ref = ref_global - start if start <= ref_global < start + n else n

    cyc = np.load(os.path.join(EXE_DIR, f"{segname}_周期检测.npz"))
    bounds = cyc["bounds"].astype(np.int64)
    drift = np.load(os.path.join(SCRIPT_DIR, "DAT解析结果", "第0.5步_漂移检验",
                                 "漂移检验_统计量.npz"))
    heart_mask = drift["heart_mask"]
    I = np.load(mod_paths["I"], mmap_mode="r")

    print("0. 软掩膜栈 (预测mask×gauss4 + 人工M_ref)", flush=True)
    soft_path = build_soft_masks(start, n, segname, ref_global, loc_ref)

    print("1. L1 保护式平移 (与基线相同)", flush=True)
    M_all = np.zeros((n, 2, 3), dtype=np.float32)
    n_accept = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_affine_worker,
                 initargs=(mod_paths["I"], loc_ref,
                           os.path.join(SCRIPT_DIR, "DAT解析结果", "第0.5步_漂移检验",
                                        "漂移检验_统计量.npz"))) as pool:
        for loc, M, ok in pool.imap_unordered(_task_affine, range(n), chunksize=16):
            M_all[loc] = M
            n_accept += int(ok)
    print(f"  {time.time()-t0:.0f}s  接受平移: {n_accept}/{n}", flush=True)

    print("2. 锚点选择 (与基线相同)", flush=True)
    ref_img = np.asarray(I[loc_ref]).astype(np.float32) / 255.0
    anchors = np.zeros(len(bounds) - 1, dtype=np.int64)
    for ci in range(len(bounds) - 1):
        lo, hi = int(bounds[ci]), int(bounds[ci + 1])
        best_ncc, best_t = -2.0, lo
        for t in range(lo, min(hi, n)):
            img = apply_affine(np.asarray(I[t]).astype(np.float32) / 255.0, M_all[t])
            nc = ncc(img, ref_img, mask=heart_mask)
            if nc > best_ncc:
                best_ncc, best_t = nc, t
        anchors[ci] = best_t

    print("3. mask加权 L2+L3 DIS (I)", flush=True)
    temp_path = os.path.join(OUT_DIR, f"_temp_{segname}_maskw_{config}.npy")
    temp = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float16,
                                     shape=(n, H, W, 2))
    F2_all = np.zeros((len(bounds) - 1, H, W, 2), dtype=np.float32)
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_maskw_worker,
                 initargs=(mod_paths, bounds, loc_ref, M_all, anchors, temp_path,
                           "I", soft_path)) as pool:
        for ci, f2 in pool.imap_unordered(_task_cycle_maskw, range(len(bounds) - 1),
                                          chunksize=1):
            F2_all[ci] = f2
            if (ci + 1) % 5 == 0:
                print(f"    周期 {ci+1}/{len(bounds)-1} | {time.time()-t0:.0f}s",
                      flush=True)
    mean_flow = temp.astype(np.float32)
    del temp
    os.remove(temp_path)
    print(f"  DIS 完成 {time.time()-t_all:.0f}s", flush=True)

    print("4. 场合成 (Φ = 仿射∘L3∘L2)", flush=True)
    fields_path = os.path.join(OUT_DIR, f"fields_{segname}_{config}_maskw.npy")
    fields = np.lib.format.open_memmap(fields_path, mode="w+", dtype=np.float16,
                                       shape=(n, H, W, 2))
    stats = []
    for t in range(n):
        ci = min(int(np.searchsorted(bounds, t, side="right") - 1), len(bounds) - 2)
        F_res = compose(mean_flow[t], F2_all[ci])
        phi = compose_affine_dense(M_all[t], F_res)
        fields[t] = phi.astype(np.float16)
        stats.append({"frame": start + t, **field_stats(phi)})
    fields.flush()

    print("5. QC (与基线同协议)", flush=True)
    rows = []
    for t in range(n):
        phi = fields[t].astype(np.float32)
        cc = ncc(warp_image(np.asarray(I[t]).astype(np.float32) / 255.0, phi),
                 ref_img, mask=heart_mask)
        rows.append({"frame": start + t, "cc": float(cc),
                     "mag_mean_mag": stats[t]["mean_mag"],
                     "mag_p99_mag": stats[t]["p99_mag"]})
    closure = []
    for ci in range(len(bounds) - 1):
        a, b = int(bounds[ci]), int(bounds[ci + 1]) - 1
        if b < n and b != a:
            d = np.abs(fields[a].astype(np.float32) - fields[b].astype(np.float32))
            closure.append({"cycle": ci, "first": start + a, "last": start + b,
                            "closure_mean_px": float(d.mean()),
                            "closure_p99_px": float(np.percentile(d, 99))})
    qc = {
        "segment": segname, "config": f"{config}+maskw", "modalities": ["I"],
        "ref_frame": ref_global, "n_frames": n, "n_cycles": len(closure),
        "cc_mean": float(np.mean([r["cc"] for r in rows])),
        "cc_p05": float(np.percentile([r["cc"] for r in rows], 5)),
        "n_affine_accept": n_accept,
        "flow_mean_mag": float(np.mean([s["mean_mag"] for s in stats])),
        "fold_ratio_mean": float(np.mean([s["fold_ratio"] for s in stats])),
        "closure_mean_px": float(np.mean([c["closure_mean_px"] for c in closure])),
    }
    with open(os.path.join(OUT_DIR, f"reg_{segname}_{config}_maskw_QC.json"), "w",
              encoding="utf-8") as f:
        json.dump({"qc": qc, "per_frame": rows, "closure": closure}, f, indent=1,
                  ensure_ascii=False)
    print(f"  mean CC={qc['cc_mean']:.4f} (p05={qc['cc_p05']:.4f})  "
          f"平均|Φ|={qc['flow_mean_mag']:.2f}px  折叠率={qc['fold_ratio_mean']*100:.2f}%  "
          f"周期闭合={qc['closure_mean_px']:.2f}px", flush=True)
    print(f"-> {fields_path} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=440)
    ap.add_argument("--segname", default="0-10s")
    ap.add_argument("--config", default="I")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reg_stack", default=None,
                    help="覆盖 reg u8 栈路径 (后30s 用 reg_I_u8.npy, step9_10 命名)")
    args = ap.parse_args()
    mp.freeze_support()
    run(args.start, args.n, args.segname, args.config, args.workers, args.reg_stack)
