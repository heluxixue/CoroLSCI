"""prep_reg_input.py -- Registration input preparation

Writes globally-normalized u8 stacks (clip p0.5-p99.5 over pooled heart
region, then min-max) for I/V/C so that all frames share one intensity
scale, as required by DIS/ECC.
"""
import json, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "registration")
os.makedirs(OUT_DIR, exist_ok=True)

OFFSET = 1320
N_SEG = 1321

# heart region mask (and registration validation consistent )
drift = np.load(os.path.join(SCRIPT_DIR, "DAT解析结果", "第0.5步_漂移检验",
                             "漂移检验_统计量.npz"))
heart = drift["heart_mask"]

print("0. 心脏区域合并统计 clip p0.5/p99.5 (全部后30s帧池化)", flush=True)
scaling = {}
for name, fn in [("I", "光强_intensity.npy"), ("V", "方差_variance.npy"),
                 ("C", "散斑对比度_contrast.npy")]:
    arr = np.load(os.path.join(RAW_DIR, fn), mmap_mode="r")
    sample = []
    for t in range(OFFSET, OFFSET + N_SEG, 7):
        f = np.asarray(arr[t]).astype(np.float64)
        sample.append(f[heart][np.isfinite(f[heart])][::4])
    sample = np.concatenate(sample)
    lo, hi = float(np.percentile(sample, 0.5)), float(np.percentile(sample, 99.5))
    scaling[name] = {"clip_low": lo, "clip_high": hi}
    print(f"  {name}: clip [{lo:.2f}, {hi:.2f}]", flush=True)

print("1. 后30s 全局 min-max → uint8", flush=True)
for name, fn in [("I", "光强_intensity.npy"), ("V", "方差_variance.npy"),
                 ("C", "散斑对比度_contrast.npy")]:
    arr = np.load(os.path.join(RAW_DIR, fn), mmap_mode="r")
    lo, hi = scaling[name]["clip_low"], scaling[name]["clip_high"]
    out = np.lib.format.open_memmap(os.path.join(OUT_DIR, f"reg_{name}_u8.npy"),
                                    mode="w+", dtype=np.uint8,
                                    shape=(N_SEG, arr.shape[1], arr.shape[2]))
    for t in range(N_SEG):
        f = np.asarray(arr[OFFSET + t]).astype(np.float64)
        f = np.clip((f - lo) / (hi - lo), 0, 1)
        out[t] = (f * 255).astype(np.uint8)
        if (t + 1) % 300 == 0:
            print(f"  {name}: {t+1}/{N_SEG}", flush=True)
    out.flush()
    print(f"  -> reg_{name}_u8.npy", flush=True)

with open(os.path.join(OUT_DIR, "reg_scaling.json"), "w", encoding="utf-8") as f:
    json.dump(scaling, f, indent=1, ensure_ascii=False)
print("2. -> reg_scaling.json")
print("完成!")
