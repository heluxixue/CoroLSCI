# -*- coding: utf-8 -*-
"""block_summary_bootstrap.py -- two-stage block-summary bootstrap of the
longitudinal perfusion-surrogate decline (paper Sec. III-D/II-J)

Protocol (prespecified):
  1. For each geometry-matched session (T2..T6), the per-frame P_mfr series
     (manufacturer convention, Eq. 2) is summarized into twelve non-overlapping
     5-s blocks (each spanning ~6 cardiac cycles at the observed 1.27-Hz
     modulation) by block averaging.
  2. Block means are resampled with replacement within each session
     (4000 replicates) and the T2->T6 decline is re-estimated, so the reported
     interval reflects block-level variability rather than the spurious
     precision of thousands of autocorrelated frames.

Output: block_summary_bootstrap.json
  point estimate = observed session-mean decline (22.0% in the paper);
  bootstrap median + 95% CI (20.7-23.8% in the paper).
"""
import json
import os

import numpy as np

ROOT = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果"
T2_RAW = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果"
OUT = r"D:\Project\CC_Project\激光散斑pure\block_summary_bootstrap_rebuilt.json"
FPS = 44.0
BLOCK_S = 5
B = 4000
SEED = 2026

SESSIONS = [  # tag, rec dir or None for the original T2 decode
    ("T2", None),
    ("T3", "zhuxin1_4_20260709-7use"),
    ("T4", "zhuxin1_5_20260709-10"),
    ("T5", "zhuxin1_6_20260709-12"),
    ("T6", "zhuxin1_7_20260709-13"),
]

T2_MASK_DIRS = [
    os.path.join(T2_RAW, "第三版执行", f"segmentation_{s}", "masks_raw")
    for s in ("0-10s", "10-20s", "20-30s", "后30s")]


def load_masks(session_dir, n):
    if session_dir is None:  # T2: masks live in four segment folders
        files = []
        for d in T2_MASK_DIRS:
            files += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy"))
        return files[:n]
    d = os.path.join(session_dir, "masks")
    return [os.path.join(d, f"{t:05d}.npy") for t in range(n)]


def p_mfr_series(tag):
    """Per-20-frame sampled P_mfr series (133 samples per 60-s session),
    matching the session-level table values of the paper (T2 1033.9, T6 806.8)."""
    if tag == "T2":
        rec = T2_RAW
        h = json.load(open(os.path.join(rec, "文件头信息.json"),
                           encoding="utf-8"))
        I = np.load(os.path.join(rec, "整段数据_2641帧",
                                 "光强_intensity.npy"), mmap_mode="r")
        V = np.load(os.path.join(rec, "整段数据_2641帧",
                                 "方差_variance.npy"), mmap_mode="r")
        masks = load_masks(None, I.shape[0])
    else:
        rec = os.path.join(ROOT, SESSIONS[[s[0] for s in SESSIONS].index(tag)][1])
        h = json.load(open(os.path.join(rec, "文件头信息.json"), encoding="utf-8"))
        I = np.load(os.path.join(rec, "光强_intensity.npy"), mmap_mode="r")
        V = np.load(os.path.join(rec, "方差_variance.npy"), mmap_mode="r")
        masks = load_masks(rec, I.shape[0])
    gain, coh = h["signal_gain"], h["coherence_factor"]
    every = 20
    n_samp = I.shape[0] // every
    out = np.zeros(n_samp)
    for k in range(n_samp):
        t = k * every
        m = np.load(masks[t]) > 0
        if m.any():
            Im = np.asarray(I[t], np.float64)[m].mean()
            Vm = np.asarray(V[t], np.float64)[m].mean()
            C = coh * np.sqrt(abs(Vm)) / max(Im, 1e-9)
            out[k] = gain * (1.0 / C - 1.0)
        else:
            out[k] = np.nan
    return out


def block_summary(x, n_blocks=12):
    """Twelve non-overlapping blocks per session. Samples are divided evenly
    (floor division); a trailing remainder of at most n_blocks-1 samples is
    dropped so that every block spans the same nominal duration."""
    nb = len(x) // n_blocks  # samples per block (133 -> 11)
    return np.array([np.nanmean(x[b * nb:(b + 1) * nb]) for b in range(n_blocks)])


def main():
    rng = np.random.RandomState(SEED)
    series = {tag: p_mfr_series(tag) for tag, _ in SESSIONS}
    blocks = {tag: block_summary(series[tag]) for tag, _ in SESSIONS}

    # observed point estimate: session-mean decline
    sm = {tag: np.nanmean(blocks[tag]) for tag, _ in SESSIONS}
    point = 100 * (sm["T2"] - sm["T6"]) / sm["T2"]

    declines = np.zeros(B)
    for b in range(B):
        bm = [rng.choice(blocks[tag], size=len(blocks[tag]), replace=True).mean()
              for tag, _ in SESSIONS]
        declines[b] = 100 * (bm[0] - bm[-1]) / bm[0]

    out = {
        "session_block_means": {k: round(float(v), 1) for k, v in sm.items()},
        "decline_pct_point": round(float(point), 1),
        "decline_pct_median": round(float(np.median(declines)), 1),
        "decline_pct_CI95": [round(float(np.percentile(declines, 2.5)), 1),
                             round(float(np.percentile(declines, 97.5)), 1)],
        "n_blocks_per_session": len(blocks["T2"]),
        "block_seconds": BLOCK_S,
        "B": B, "seed": SEED,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
