"""measure_paper_metrics.py -- paper measurement metrics v2

Fixes over v1:
- v1 FLOPs inflated: hooks were not removed inside timed(), MAC accumulated
  over 21 forward passes -> now measured over a single forward pass
- v1 decoding 46 ms was cold-page disk read: now separates (a) hot-cache
  pure computation (b) cold sequential streaming read (c) batched decoding
- inference timing uses the same convention as paper_seg_benchmark.timed
  (time.time mean, 3 warm-up + 20 timed) plus perf_counter/median reference
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"d:\Project\CC_Project\激光散斑pure")
from seg_common import ResUNet, PAD_H, PAD_W           # noqa: E402
from step5f_nnunet_arch import PlainConvUNet            # noqa: E402
from step5b_train_seg3d import ResUNet3D                # noqa: E402
from step5l_flowmatch_centerline import CenterlineFlowNet  # noqa: E402
from step7_train_gated import GatedFusionNet            # noqa: E402
import export_pimsoft_dat as dec                        # noqa: E402
from heart1_indicators_v2 import fill_small_holes, compute_structural_metrics  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SRC = r"d:\Project\CC_Project\激光散斑pure"
EXE = os.path.join(SRC, "DAT解析结果", "第三版执行")
SEG = os.path.join(EXE, "segmentation")
OUT60 = os.path.join(EXE, "60s统一结果")
DAT = os.path.join(SRC, "zhuxin.dat")
H, W = 647, 549
H4, W4 = 162, 137
DEV = "cuda" if torch.cuda.is_available() else "cpu"
results = {"device": DEV,
           "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
           "torch": torch.__version__,
           "tf32_cudnn": torch.backends.cudnn.allow_tf32 if DEV == "cuda" else None}


def params_m(model):
    return sum(p.numel() for p in model.parameters())


def attach_flop_hooks(model):
    """前向钩子按实际输出形状累计 MAC; FLOPs = 2×MAC (thop 口径, 偏置忽略)。"""
    stats = {"macs": 0}
    handles = []

    def make_hook():
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(mod, nn.Conv2d):
                cout, cin, kh, kw = mod.weight.shape
                stats["macs"] += int(o.shape[0] * o.shape[2] * o.shape[3] * cin * kh * kw * cout)
            elif isinstance(mod, nn.ConvTranspose2d):
                cin, cout, kh, kw = mod.weight.shape
                stats["macs"] += int(o.shape[0] * o.shape[2] * o.shape[3] * kh * kw * cin * cout)
            elif isinstance(mod, nn.Conv3d):
                cout, cin, kd, kh, kw = mod.weight.shape
                stats["macs"] += int(o.shape[0] * o.shape[2] * o.shape[3] * o.shape[4]
                                     * cin * kd * kh * kw * cout)
            elif isinstance(mod, nn.Linear):
                stats["macs"] += int(o.shape[0] * mod.in_features * mod.out_features)
        return h

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.Linear)):
            handles.append(m.register_forward_hook(make_hook()))
    return stats, handles


def timed_bench(fn, n_warm=3, n_rep=20):
    """与 paper_seg_benchmark.timed 完全同口径: time.time 均值。"""
    for _ in range(n_warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return float(np.mean(ts) * 1000)


def timed_pc(fn, n_warm=3, n_rep=20):
    """perf_counter 均值/中位 (参考口径)。"""
    for _ in range(n_warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts) * 1000), float(np.median(ts) * 1000)


def measure_model(name, model, ckpt, make_input, n_warm=3, n_rep=20):
    """顺序: 载权重 → 参数 → 推理计时(无钩子) → 单次前向 FLOPs+峰值显存。"""
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    model.eval().to(DEV)
    inp = make_input()
    with torch.no_grad():
        mean_bench = timed_bench(lambda: model(inp), n_warm=n_warm, n_rep=n_rep)
        mean_pc, med_pc = timed_pc(lambda: model(inp), n_warm=0, n_rep=n_rep)
        stats, handles = attach_flop_hooks(model)
        if DEV == "cuda":
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
        model(inp)
        if DEV == "cuda":
            torch.cuda.synchronize()
            peak_total = torch.cuda.max_memory_allocated() / 1024 ** 2
    for h in handles:
        h.remove()
    rec = {
        "params_M": round(params_m(model) / 1e6, 3),
        "GFLOPS": round(2 * stats["macs"] / 1e9, 3),
        "infer_mean_ms_bench": round(mean_bench, 2),
        "infer_mean_ms_pc": round(mean_pc, 2),
        "infer_median_ms_pc": round(med_pc, 2),
    }
    if DEV == "cuda":
        rec["peak_activation_MiB"] = round(peak_total - base, 1)
        rec["total_resident_MiB"] = round(peak_total, 1)
    results[name] = rec
    print(f"  {name:<28s} params={rec['params_M']:>6.3f}M  "
          f"FLOPs={rec['GFLOPS']:>7.3f}G  infer={rec['infer_mean_ms_bench']:>6.2f}ms")
    del model, inp, stats
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def main():
    print(f"device: {DEV} ({results['gpu']}), torch {results['torch']}, "
          f"cudnn.allow_tf32={results['tf32_cudnn']}")

    print("\n[1] 基准网络: 参数量 / FLOPs / 推理耗时 (benchmark 同口径)")
    x2d = lambda: torch.zeros(1, 3, PAD_H, PAD_W, device=DEV)              # noqa: E731
    x3d = lambda: torch.zeros(1, 3, 3, H4, W4, device=DEV)                 # noqa: E731

    measure_model("ResUNet-I (adopted)", ResUNet(base=32),
                  os.path.join(SEG, "models", "seg_I_seed0.pt"), x2d)
    measure_model("nnU-Net-style", PlainConvUNet(in_ch=3, base=16, n_stages=5),
                  os.path.join(OUT60, "nnunet_arch_best.pth"), x2d)
    measure_model("3D self-supervised", ResUNet3D(base=32),
                  os.path.join(OUT60, "seg3d_selfsup_best.pth"), x3d)

    # Flow Matching: 每帧 = 10 步积分 (FLOPs/耗时均按整帧口径)
    mcl = CenterlineFlowNet(base=16)
    mcl.load_state_dict(torch.load(os.path.join(OUT60, "flowmatch_cl_best.pth")))
    mcl.eval().to(DEV)
    cl_prev = torch.zeros((1, 1, H4, W4), device=DEV)
    img4 = torch.zeros((1, 1, H4, W4), device=DEV)

    def fm_once():
        phi = torch.zeros((1, 2, H4, W4), device=DEV)
        for i in range(10):
            t = torch.full((1,), i / 10.0, device=DEV)
            phi = phi + mcl(cl_prev, img4, t) / 10.0
        return phi

    with torch.no_grad():
        mean_bench = timed_bench(fm_once, n_warm=3, n_rep=10)
        stats, handles = attach_flop_hooks(mcl)
        fm_once()
    for h in handles:
        h.remove()
    rec = {"params_M": round(params_m(mcl) / 1e6, 3),
           "GFLOPS": round(2 * stats["macs"] / 1e9, 3),
           "infer_mean_ms_bench": round(mean_bench, 2),
           "note": "per frame = 10 integration steps"}
    results["Flow Matching (centerline)"] = rec
    print(f"  {'Flow Matching (centerline)':<28s} params={rec['params_M']:>6.3f}M  "
          f"FLOPs={rec['GFLOPS']:>7.3f}G  infer={rec['infer_mean_ms_bench']:>6.2f}ms")
    del mcl
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # Gated fusion (消融臂): 仅参数量 + FLOPs
    mg = GatedFusionNet(base=32).eval().to(DEV)
    with torch.no_grad():
        stats, handles = attach_flop_hooks(mg)
        mg(torch.zeros(1, 3, PAD_H, PAD_W, device=DEV))
        stats["macs"] = 0
        mg(torch.zeros(1, 3, PAD_H, PAD_W, device=DEV))
    for h in handles:
        h.remove()
    results["Gated fusion (ablation)"] = {
        "params_M": round(params_m(mg) / 1e6, 3),
        "GFLOPS": round(2 * stats["macs"] / 1e9, 3)}
    print(f"  {'Gated fusion (ablation)':<28s} params={results['Gated fusion (ablation)']['params_M']:>6.3f}M  "
          f"FLOPs={results['Gated fusion (ablation)']['GFLOPS']:>7.3f}G")
    del mg
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # ---- 2. 解码器 ----
    print("\n[2] 解码器: 原始 .dat 逐帧解析")
    header = dec.read_header(DAT)
    vmap, imap = dec.open_raw_data_maps(DAT, header)
    n_px = header["pixel_count_per_frame"]
    cf_, sg_ = header["coherence_factor"], header["signal_gain"]

    def decode_one(i):
        vf = dec.get_frame(vmap, i, header)
        inf_ = dec.get_frame(imap, i, header)
        c = dec.calculate_contrast(vf, inf_, cf_)
        p = dec.calculate_perfusion(vf, inf_, cf_, sg_)
        return p

    # (a) 热缓存纯计算: 先整段预读一遍, 再对同段计时
    hot = list(range(1050, 1250))
    for i in hot:
        decode_one(i)
    ts = []
    for i in hot:
        t0 = time.perf_counter()
        decode_one(i)
        ts.append(time.perf_counter() - t0)
    hot_ms = float(np.mean(ts) * 1000)

    # 纯读取分量 (memmap 拷贝, 不计算 C/P)
    ts = []
    for i in hot:
        t0 = time.perf_counter()
        dec.get_frame(vmap, i, header)
        dec.get_frame(imap, i, header)
        ts.append(time.perf_counter() - t0)
    copy_ms = float(np.mean(ts) * 1000)

    # (b) 冷顺序流式读: 全新区间, 单次通过 (含磁盘)
    cold = list(range(1500, 1700))
    ts = []
    for i in cold:
        t0 = time.perf_counter()
        decode_one(i)
        ts.append(time.perf_counter() - t0)
    cold_ms = float(np.mean(ts) * 1000)

    # (c) 批量解码: 一次读连续 N 帧, 向量化计算 C/P
    NB = 100
    b0 = 1750
    vblk = np.asarray(vmap[b0 * n_px:(b0 + NB) * n_px]).reshape(NB, H, W, order="F")
    iblk = np.asarray(imap[b0 * n_px:(b0 + NB) * n_px]).reshape(NB, H, W, order="F")
    with np.errstate(divide="ignore", invalid="ignore"):
        cblk = np.sign(vblk) * cf_ * np.sqrt(np.abs(vblk)) / iblk
        pblk = np.minimum(sg_ * (1.0 / cblk - 1.0), 3000.0)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        vb = np.asarray(vmap[b0 * n_px:(b0 + NB) * n_px]).reshape(NB, H, W, order="F")
        ib = np.asarray(imap[b0 * n_px:(b0 + NB) * n_px]).reshape(NB, H, W, order="F")
        with np.errstate(divide="ignore", invalid="ignore"):
            cb = np.sign(vb) * cf_ * np.sqrt(np.abs(vb)) / ib
            pb = np.minimum(sg_ * (1.0 / cb - 1.0), 3000.0)
        ts.append(time.perf_counter() - t0)
    batch_ms = float(np.mean(ts) * 1000 / NB)

    mp_per_frame = 4 * H * W / 1e6
    results["decoder"] = {
        "ms_per_frame_hot_cache": round(hot_ms, 2),
        "ms_per_frame_copy_only": round(copy_ms, 2),
        "ms_per_frame_cold_stream": round(cold_ms, 2),
        "ms_per_frame_batched": round(batch_ms, 2),
        "MP_per_frame": round(mp_per_frame, 3),
        "MB_per_frame": round(2 * H * W * 8 / 1e6, 2),
        "MBs_at_44fps": round(2 * H * W * 8 * 44 / 1e6, 1),
    }
    print(f"  热缓存纯计算: {hot_ms:.2f} ms/帧 (其中纯拷贝 {copy_ms:.2f} ms)"
          f" -> 计算可支撑 {1000 / hot_ms:.0f} fps")
    print(f"  冷顺序流式读: {cold_ms:.2f} ms/帧 (含磁盘, 单次通过)")
    print(f"  批量解码(100帧向量化): {batch_ms:.2f} ms/帧 -> {1000 / batch_ms:.0f} fps")
    print(f"  每帧 {mp_per_frame:.3f} MP (4 图); 原始流 {2 * H * W * 8 / 1e6:.2f} MB/帧"
          f" = {2 * H * W * 8 * 44 / 1e6:.0f} MB/s @44fps")

    # ---- 3. 指标计算逐帧耗时 (真实掩膜 + 真实灌注帧, 帧 170) ----
    print("\n[3] 指标计算: 真实掩膜 (帧 170, 647×549)")
    mask = np.load(os.path.join(EXE, "segmentation_0-10s", "masks_raw",
                                "00170.npy")) > 0
    pf_frame = decode_one(170)

    ms, _ = timed_pc(lambda: fill_small_holes(mask), n_warm=3, n_rep=20)
    results["indicators"] = {"fill_ms": round(ms, 2)}
    print(f"  <=10px 填孔: {ms:.2f} ms")

    ms, _ = timed_pc(lambda: compute_structural_metrics(mask), n_warm=3, n_rep=20)
    results["indicators"]["structural_ms"] = round(ms, 2)
    print(f"  骨架+距离变换+分支 (结构指标): {ms:.2f} ms")

    ms, _ = timed_pc(lambda: float(pf_frame[mask].mean()), n_warm=3, n_rep=20)
    results["indicators"]["perfusion_mean_ms"] = round(ms, 3)
    print(f"  掩膜均值灌注 (功能指标): {ms:.3f} ms")

    pp_ms = (results["indicators"]["fill_ms"]
             + results["indicators"]["structural_ms"])
    results["indicators"]["postprocess_total_ms"] = round(pp_ms, 2)
    print(f"  掩膜后处理合计: {pp_ms:.2f} ms/帧")

    out = os.path.join(SRC, "paper_metrics_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
