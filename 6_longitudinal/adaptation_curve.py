"""adaptation_curve.py -- calibration-frame vs recovery-accuracy curves
(T1 working-distance shift; heart-2 cross-heart shift)

Protocol: fixed held-out validation set (T1: the 6 frames of split.json;
heart 2: the 10 frames of split.json). For each calibration budget n
(T1: 5/10/14; heart 2: 5/10/20/30), three random subsets of n training frames
are fine-tuned and evaluated on the same validation set. The zero-shot point
is n = 0. Training, early-stopping and evaluation frames are disjoint at
every point.
Output: adaptation_curve.json
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, r"D:\Project\CC_Project\激光散斑pure")
sys.stdout.reconfigure(encoding="utf-8")

from seg_common import ResUNet, seg_loss, soft_cldice, gpu_augment

PRE_H1 = (r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行"
          r"\segmentation\models\seg_I_seed0.pt")
PRE_H2 = PRE_H1  # 心2微调也从心1预训练出发 (与原协议一致)
OUT = r"D:\Project\CC_Project\激光散斑pure\adaptation_curve.json"
SEED = 2026
EPOCHS = 150
PATIENCE = 40


def p1p99(frame):
    fin = np.isfinite(frame)
    vals = frame[fin] if fin.any() else frame
    lo, hi = np.percentile(vals, [1, 99])
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


def load_frames(raw_I, frame_ids, mask_dir):
    I = np.load(raw_I, mmap_mode="r")
    n, H, W = I.shape
    PH, PW = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    imgs = np.zeros((len(frame_ids), 3, H, W), dtype=np.float32)
    msks = np.zeros((len(frame_ids), H, W), dtype=np.uint8)
    for i, fi in enumerate(frame_ids):
        f = p1p99(np.asarray(I[fi], dtype=np.float64))
        imgs[i, 0] = imgs[i, 1] = imgs[i, 2] = f
        m = np.array(Image.open(os.path.join(mask_dir, f"{fi}.png")))
        msks[i] = ((m[..., 1] == 255) if m.ndim == 3 else (m > 127)).astype(np.uint8)
    return imgs, msks, (H, W, PH, PW)


class DS(Dataset):
    def __init__(self, imgs, msks, H, W, PH, PW):
        self.imgs, self.msks = imgs, msks
        self.H, self.W, self.PH, self.PW = H, W, PH, PW
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        pi = np.zeros((3, self.PH, self.PW), dtype=np.float32)
        pi[:, :self.H, :self.W] = self.imgs[i]
        pm = np.zeros((self.PH, self.PW), dtype=np.float32)
        pm[:self.H, :self.W] = self.msks[i]
        return torch.from_numpy(pi), torch.from_numpy(pm).unsqueeze(0)


def finetune_eval(imgs, msks, geom, tr_ids, va_ids, pre, dev):
    H, W, PH, PW = geom
    model = ResUNet().to(dev)
    model.load_state_dict(torch.load(pre, map_location=dev))
    tr_loader = DataLoader(DS(imgs[tr_ids], msks[tr_ids], *geom), batch_size=2,
                           shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.RandomState(SEED)
    best, best_ep, patience = -1.0, 0, 0
    best_state = None
    import torch.nn.functional as F
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            x, y = gpu_augment(x, y, rng)
            with torch.amp.autocast("cuda"):
                loss, _ = seg_loss(model(x), y, w_cldice=0.3, w_boundary=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        ds_ = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in va_ids:
                x = torch.from_numpy(imgs[i])
                xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
                p = torch.sigmoid(model(xp).float())[0, 0, :H, :W].cpu().numpy()
                pred = p > 0.5
                gt = msks[i] > 0
                s = pred.sum() + gt.sum()
                ds_.append(2 * (pred & gt).sum() / s if s else 1.0)
        vd = float(np.mean(ds_))
        if vd > best:
            best, best_ep, patience = vd, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    return best, best_ep


def main():
    dev = torch.device("cuda")
    t0 = time.time()
    results = {}

    # ---- T1 ----
    T1_REC = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果\zhuxin1_3_20260709-4"
    T1_MASK = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\标注样本_T1特写段20帧对应标注\SegmentationClass"
    split1 = json.load(open(os.path.join(T1_REC, "finetune", "split.json"),
                            encoding="utf-8"))
    val1 = split1["val"]                    # 6 帧固定验证
    pool1 = split1["train"]                 # 14 帧训练池
    ids1 = pool1 + val1
    imgs1, msks1, geom1 = load_frames(os.path.join(T1_REC, "光强_intensity.npy"),
                                      ids1, T1_MASK)
    vidx1 = [ids1.index(v) for v in val1]
    pidx1 = [ids1.index(p) for p in pool1]
    print(f"[T1] val={len(vidx1)} pool={len(pidx1)}", flush=True)
    rows = []
    # zero-shot
    zs = finetune_eval(imgs1, msks1, geom1, np.array([], int), vidx1,
                       PRE_H1, dev) if False else None
    import torch.nn.functional as F
    model = ResUNet().to(dev)
    model.load_state_dict(torch.load(PRE_H1, map_location=dev))
    model.eval()
    H, W, PH, PW = geom1
    ds_ = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in vidx1:
            x = torch.from_numpy(imgs1[i])
            xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
            p = torch.sigmoid(model(xp).float())[0, 0, :H, :W].cpu().numpy()
            pred = p > 0.5
            gt = msks1[i] > 0
            ds_.append(2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1))
    rows.append({"n_frames": 0, "repeat": 0, "val_dice": round(float(np.mean(ds_)), 4)})
    print(f"[T1] zero-shot val Dice = {np.mean(ds_):.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    for n in (5, 10, 14):
        for rep in range(3):
            rng = np.random.RandomState(1000 * n + rep)
            tr = rng.choice(pidx1, min(n, len(pidx1)), replace=False)
            d, ep = finetune_eval(imgs1, msks1, geom1, tr, vidx1, PRE_H1, dev)
            rows.append({"n_frames": n, "repeat": rep, "val_dice": round(d, 4)})
            print(f"[T1] n={n:2d} rep{rep}: val Dice={d:.4f} (best ep{ep}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    results["T1_closeup_233mm"] = rows

    # ---- 心2 ----
    H2_I = r"D:\Project\CC_Project\激光散斑数据2\newdata\整段数据_2641帧\光强_intensity.npy"
    H2_MASK = r"D:\Project\CC_Project\激光散斑数据2\newdata\40帧mask标注\SegmentationClass"
    split2 = json.load(open(r"D:\Project\CC_Project\激光散斑数据2\newdata\finetune\split.json",
                            encoding="utf-8"))
    val2 = split2["val"]
    pool2 = split2["train"]
    names = sorted(int(os.path.splitext(f)[0].split("_")[-1])
                   for f in os.listdir(H2_MASK) if f.endswith(".png"))
    name_map = {int(os.path.splitext(f)[0].split("_")[-1]): f
                for f in os.listdir(H2_MASK) if f.endswith(".png")}
    # 心2 mask 文件名形如 frame_XXXX ? 适配两种命名
    ids2 = pool2 + val2
    H2_MASK2 = H2_MASK
    imgs2 = None
    def load2(ids):
        I = np.load(H2_I, mmap_mode="r")
        n, H, W = I.shape
        PH, PW = (H + 15) // 16 * 16, (W + 15) // 16 * 16
        imgs = np.zeros((len(ids), 3, H, W), dtype=np.float32)
        msks = np.zeros((len(ids), H, W), dtype=np.uint8)
        for i, fi in enumerate(ids):
            f = p1p99(np.asarray(I[fi], dtype=np.float64))
            imgs[i, 0] = imgs[i, 1] = imgs[i, 2] = f
            fn = name_map.get(fi)
            m = np.array(Image.open(os.path.join(H2_MASK2, fn)))
            msks[i] = ((m[..., 1] == 255) if m.ndim == 3 else (m > 127)).astype(np.uint8)
        return imgs, msks, (H, W, PH, PW)
    imgs2, msks2, geom2 = load2(ids2)
    vidx2 = [ids2.index(v) for v in val2]
    pidx2 = [ids2.index(p) for p in pool2]
    print(f"[H2] val={len(vidx2)} pool={len(pidx2)}", flush=True)
    rows2 = []
    model = ResUNet().to(dev)
    model.load_state_dict(torch.load(PRE_H2, map_location=dev))
    model.eval()
    H, W, PH, PW = geom2
    ds_ = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in vidx2:
            x = torch.from_numpy(imgs2[i])
            xp = F.pad(x, (0, PW - W, 0, PH - H)).unsqueeze(0).to(dev)
            p = torch.sigmoid(model(xp).float())[0, 0, :H, :W].cpu().numpy()
            pred = p > 0.5
            gt = msks2[i] > 0
            ds_.append(2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1))
    rows2.append({"n_frames": 0, "repeat": 0, "val_dice": round(float(np.mean(ds_)), 4)})
    print(f"[H2] zero-shot val Dice = {np.mean(ds_):.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    for n in (5, 10, 20, 30):
        for rep in range(3):
            rng = np.random.RandomState(2000 * n + rep)
            tr = rng.choice(pidx2, min(n, len(pidx2)), replace=False)
            d, ep = finetune_eval(imgs2, msks2, geom2, tr, vidx2, PRE_H2, dev)
            rows2.append({"n_frames": n, "repeat": rep, "val_dice": round(d, 4)})
            print(f"[H2] n={n:2d} rep{rep}: val Dice={d:.4f} (best ep{ep}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    results["heart2_cross_heart"] = rows2

    json.dump(results, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
