"""step12b_reg_segment.py -- Per-segment registration pipeline (L1+L2+L3)

L1: guarded global translation (frame -> reference).
L2: intra-cycle DIS frame -> cycle anchor.
L3: anchor -> global reference DIS.
Composes into per-frame displacement fields with QC statistics.
"""
import argparse, csv, json, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reg_common import (H, W, warp_image, compose, compose_affine_dense, ncc,
                        field_stats)
from step9_10_registration import (_init_worker, _task_cycle, _init_affine_worker,
                                   _task_affine, CONFIG_MODS, _G)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
EXE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行")
OUT_DIR = os.path.join(EXE_DIR, "registration")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_FILES = {"I": "光强_intensity.npy", "V": "方差_variance.npy",
             "C": "散斑对比度_contrast.npy"}


def build_segment_stacks(start, n, segname):
    """segment frame + reference frame (tail ) → global consistent normalization u8 stack """
    with open(os.path.join(OUT_DIR, "reg_scaling.json"), encoding="utf-8") as f:
        scaling = json.load(f)
    with open(os.path.join(EXE_DIR, "reference", "reference_frame.json"),
              encoding="utf-8") as f:
        ref_global = int(json.load(f)["frame_id"])

    paths = {}
    for mod, fn in RAW_FILES.items():
        arr = np.load(os.path.join(RAW_DIR, fn), mmap_mode="r")
        lo, hi = scaling[mod]["clip_low"], scaling[mod]["clip_high"]
        out = np.lib.format.open_memmap(os.path.join(OUT_DIR, f"reg_{segname}_{mod}_u8.npy"),
                                        mode="w+", dtype=np.uint8,
                                        shape=(n + 1, H, W))
        for t in range(n):
            f = np.asarray(arr[start + t]).astype(np.float64)
            out[t] = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        f = np.asarray(arr[ref_global]).astype(np.float64)
        out[n] = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        out.flush()
        paths[mod] = os.path.join(OUT_DIR, f"reg_{segname}_{mod}_u8.npy")
        print(f"  -> {os.path.basename(paths[mod])} ({n}+1帧)", flush=True)
    return paths, ref_global


def run(start, n, segname, config, workers):
    print(f"\n{'='*60}\n配准 {segname} → 参考帧 (R-{config})\n{'='*60}", flush=True)
    t_all = time.time()

    print("0. 归一化 u8 栈 (段 + 参考帧)", flush=True)
    mod_paths, ref_global = build_segment_stacks(start, n, segname)
    loc_ref = n  # reference frame in segment stack tail

    cyc = np.load(os.path.join(EXE_DIR, f"{segname}_周期检测.npz"))
    bounds = cyc["bounds"].astype(np.int64)      # segment inside local frame no.
    print(f"  参考帧: {ref_global}  周期数: {len(bounds)-1}", flush=True)

    drift = np.load(os.path.join(SCRIPT_DIR, "DAT解析结果", "第0.5步_漂移检验",
                                 "漂移检验_统计量.npz"))
    heart_mask = drift["heart_mask"]

    I = np.load(mod_paths["I"], mmap_mode="r")

    # ============================================================
    print("1. L1 全局平移 (每帧→参考, Intensity 驱动)", flush=True)
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
    print(f"  {time.time()-t0:.0f}s  接受平移: {n_accept}/{n} ({n_accept/n*100:.1f}%)",
          flush=True)

    # ============================================================
    print("2. 周期锚点选择 (仿射校正后与参考帧 NCC 最高)", flush=True)
    ref_img = np.asarray(I[loc_ref]).astype(np.float32) / 255.0
    anchors = np.zeros(len(bounds) - 1, dtype=np.int64)
    for ci in range(len(bounds) - 1):
        lo, hi = int(bounds[ci]), int(bounds[ci + 1])
        best_ncc, best_t = -2.0, lo
        for t in range(lo, min(hi, n)):
            img = apply_affine_local(np.asarray(I[t]).astype(np.float32) / 255.0,
                                     M_all[t])
            nc = ncc(img, ref_img, mask=heart_mask)
            if nc > best_ncc:
                best_ncc, best_t = nc, t
        anchors[ci] = best_t
    print(f"  锚点相位分布: {np.mean([(anchors[ci]-bounds[ci])/max(bounds[ci+1]-bounds[ci],1) for ci in range(len(anchors))]):.2f}",
          flush=True)

    # ============================================================
    mods = CONFIG_MODS[config]
    print(f"3. L2+L3 DIS 稠密光流 (模态: {mods})", flush=True)
    mean_flow = np.zeros((n, H, W, 2), dtype=np.float32)
    F2_all = np.zeros((len(bounds) - 1, H, W, 2), dtype=np.float32)

    for mod in mods:
        print(f"  --- 模态 {mod} ---", flush=True)
        temp_path = os.path.join(OUT_DIR, f"_temp_{segname}_{config}_{mod}.npy")
        temp = np.lib.format.open_memmap(temp_path, mode="w+",
                                         dtype=np.float16, shape=(n, H, W, 2))
        t0 = time.time()
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(mod_paths, bounds, loc_ref, M_all, anchors,
                               temp_path, mod)) as pool:
            for ci, f2 in pool.imap_unordered(_task_cycle, range(len(bounds) - 1),
                                              chunksize=1):
                F2_all[ci] = f2
                if (ci + 1) % 5 == 0:
                    print(f"    周期 {ci+1}/{len(bounds)-1} | {time.time()-t0:.0f}s",
                          flush=True)
        mean_flow += temp.astype(np.float32) / len(mods)
        del temp
        os.remove(temp_path)
    F2_all /= len(mods)
    print(f"  DIS 完成 {time.time()-t_all:.0f}s", flush=True)

    # ============================================================
    print("4. 场合成 (Φ = 仿射∘L3∘L2) 并保存", flush=True)
    fields_path = os.path.join(OUT_DIR, f"fields_{segname}_{config}.npy")
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

    # ============================================================
    print("5. QC: warped-NCC / 周期闭合 / 折叠率", flush=True)
    rows = []
    for t in range(n):
        phi = fields[t].astype(np.float32)
        cc = ncc(warp_image(np.asarray(I[t]).astype(np.float32) / 255.0, phi),
                 ref_img, mask=heart_mask)
        rows.append({"frame": start + t, "cc": float(cc),
                     **{f"mag_{k}": stats[t][k] for k in ("mean_mag", "p99_mag")}})

    # cycle closure : same cycle first frame displacement field (heart to )
    closure = []
    for ci in range(len(bounds) - 1):
        a, b = int(bounds[ci]), int(bounds[ci + 1]) - 1
        if b < n and b != a:
            d = np.abs(fields[a].astype(np.float32) - fields[b].astype(np.float32))
            closure.append({"cycle": ci, "first": start + a, "last": start + b,
                            "closure_mean_px": float(d.mean()),
                            "closure_p99_px": float(np.percentile(d, 99))})

    cc_mean = float(np.mean([r["cc"] for r in rows]))
    qc = {
        "segment": segname, "config": config, "modalities": mods,
        "ref_frame": ref_global, "n_frames": n, "n_cycles": len(bounds) - 1,
        "cc_mean": cc_mean, "cc_p05": float(np.percentile([r["cc"] for r in rows], 5)),
        "n_affine_accept": n_accept,
        "flow_mean_mag": float(np.mean([s["mean_mag"] for s in stats])),
        "fold_ratio_mean": float(np.mean([s["fold_ratio"] for s in stats])),
        "closure_mean_px": float(np.mean([c["closure_mean_px"] for c in closure])),
    }
    with open(os.path.join(OUT_DIR, f"reg_{segname}_{config}_QC.json"), "w",
              encoding="utf-8") as f:
        json.dump({"qc": qc, "per_frame": rows, "closure": closure}, f, indent=1,
                  ensure_ascii=False)
    with open(os.path.join(OUT_DIR, f"reg_{segname}_{config}_QC.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(OUT_DIR, f"flowstats_{segname}_{config}.npy"),
            np.array([(s["frame"], s["mean_mag"], s["p99_mag"], s["fold_ratio"])
                      for s in stats]))
    print(f"  [QC] mean CC={cc_mean:.4f} (p05={qc['cc_p05']:.4f})  "
          f"平均|Φ|={qc['flow_mean_mag']:.2f}px  折叠率={qc['fold_ratio_mean']*100:.2f}%  "
          f"周期闭合={qc['closure_mean_px']:.2f}px", flush=True)
    print(f"  全部: {time.time()-t_all:.0f}s -> {OUT_DIR}", flush=True)
    return qc


def apply_affine_local(img, M):
    """this inside affine use (and reg_common.apply_affine same , )"""
    import cv2
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=440)
    ap.add_argument("--segname", required=True)
    ap.add_argument("--config", default="IVC", choices=["I", "IV", "IC", "IVC"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    mp.freeze_support()
    run(args.start, args.n, args.segname, args.config, args.workers)
