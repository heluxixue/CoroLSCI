"""t1_finetune.py -- T1 close-up session (20260709-4, 233 mm): 20-frame
fine-tune repairing the scale/brightness domain shift

Protocol identical to heart2_finetune.py (AdamW + cosine + mixed precision +
GPU augmentation + same loss), differences:
  - 20 frames, split 14 train / 6 held-out validation
  - dynamic padding to a multiple of 16 (T1 = 650x698 -> 656x704)
Output: seg_I_T1.pt + split.json + training curves
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

from seg_common import ResUNet, seg_loss, soft_cldice, evaluate_mask, gpu_augment

REC = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\解析结果\zhuxin1_3_20260709-4"
RAW_I = os.path.join(REC, "光强_intensity.npy")
MASK_DIR = r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\标注样本_T1特写段20帧对应标注\SegmentationClass"
PRETRAINED = (r"D:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行"
              r"\segmentation\models\seg_I_seed0.pt")
OUT_DIR = os.path.join(REC, "finetune")
SEED = 2026
EPOCHS = 150
BATCH = 2
LR = 3e-4


def p1p99(frame):
    fin = np.isfinite(frame)
    vals = frame[fin] if fin.any() else frame
    lo, hi = np.percentile(vals, [1, 99])
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda")
    print(f"设备 {dev} | finetune ResUNet-I (T1 特写段)", flush=True)
    t0 = time.time()

    I = np.load(RAW_I, mmap_mode="r")
    n, H_RAW, W_RAW = I.shape
    PAD_H, PAD_W = (H_RAW + 15) // 16 * 16, (W_RAW + 15) // 16 * 16
    print(f"T1: {n} 帧 {W_RAW}x{H_RAW} pad->{PAD_W}x{PAD_H}", flush=True)

    names = sorted(f for f in os.listdir(MASK_DIR) if f.endswith(".png"))
    frame_ids = [int(os.path.splitext(n)[0]) for n in names]
    assert len(names) == 20, f"expect 20 masks, got {len(names)}"
    N = 20
    images = np.zeros((N, 3, H_RAW, W_RAW), dtype=np.float32)
    masks = np.zeros((N, H_RAW, W_RAW), dtype=np.uint8)
    for i, (nm, fi) in enumerate(zip(names, frame_ids)):
        f = p1p99(np.asarray(I[fi], dtype=np.float64))
        images[i, 0] = images[i, 1] = images[i, 2] = f
        m = np.array(Image.open(os.path.join(MASK_DIR, nm)))
        masks[i] = ((m[..., 1] == 255) if m.ndim == 3 else (m > 127)).astype(np.uint8)
    print(f"数据: 20 帧 p1-p99 | 前景均值 {masks.sum(axis=(1,2)).mean():.0f}px", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N)
    tr_idx, va_idx = perm[:14], perm[14:]
    split = {"seed": SEED, "train": [frame_ids[i] for i in tr_idx],
             "val": [frame_ids[i] for i in va_idx]}
    json.dump(split, open(os.path.join(OUT_DIR, "split.json"), "w",
                          encoding="utf-8"), indent=2)
    print(f"划分: train={len(tr_idx)} val={len(va_idx)} (val 帧: {split['val']})",
          flush=True)

    class DS(Dataset):
        def __init__(self, idx):
            self.imgs, self.msks = images[idx], masks[idx]
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, i):
            pi = np.zeros((3, PAD_H, PAD_W), dtype=np.float32)
            pi[:, :H_RAW, :W_RAW] = self.imgs[i]
            pm = np.zeros((PAD_H, PAD_W), dtype=np.float32)
            pm[:H_RAW, :W_RAW] = self.msks[i]
            return torch.from_numpy(pi), torch.from_numpy(pm).unsqueeze(0)

    tr_loader = DataLoader(DS(tr_idx), batch_size=BATCH, shuffle=True,
                           num_workers=0, drop_last=True, pin_memory=True)
    va_loader = DataLoader(DS(va_idx), batch_size=4, shuffle=False, num_workers=0)

    model = ResUNet().to(dev)
    model.load_state_dict(torch.load(PRETRAINED, map_location=dev))
    model.eval()

    def evaluate(model, idx_list):
        ds_, cs_, hds = [], [], []
        import torch.nn.functional as F
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in idx_list:
                x = torch.from_numpy(images[i])
                xp = F.pad(x, (0, PAD_W - W_RAW, 0, PAD_H - H_RAW)
                           ).unsqueeze(0).to(dev)
                p = torch.sigmoid(model(xp).float())[0, 0, :H_RAW, :W_RAW].cpu().numpy()
                met = evaluate_mask(p > 0.5, masks[i] > 0)
                ds_.append(met["dice"]); cs_.append(met["cldice"]); hds.append(met["hd95"])
        return float(np.mean(ds_)), float(np.mean(cs_)), float(np.mean(hds))

    zd, zc, zh = evaluate(model, list(range(N)))
    print(f"[zero-shot] 20帧 Dice={zd:.4f} clDice={zc:.4f} HD95={zh:.2f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    aug_rng = np.random.RandomState(SEED)

    def fast_val():
        model.eval()
        ds_, cs_ = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y in va_loader:
                x, y = x.to(dev), y.to(dev)
                p = torch.sigmoid(model(x).float())
                pred = (p > 0.5).float()
                for b in range(p.shape[0]):
                    inter = (pred[b] * y[b]).sum()
                    den = pred[b].sum() + y[b].sum()
                    ds_.append(float(2 * inter / den) if den > 0 else 0.0)
                    cs_.append(float(soft_cldice(y[b:b + 1], p[b:b + 1])))
        return float(np.mean(ds_)), float(np.mean(cs_))

    log_rows, best, best_ep, patience = [], -1.0, 0, 0
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            x, y = gpu_augment(x, y, aug_rng)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss, _ = seg_loss(logits, y, w_cldice=0.3, w_boundary=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        v_d, v_cl = fast_val()
        score = 0.5 * v_d + 0.5 * v_cl
        log_rows.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                         "val_dice": v_d, "val_soft_cldice": v_cl})
        if score > best:
            best, best_ep, patience = score, ep, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "seg_I_T1.pt"))
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0:
            print(f"  ep {ep:3d} | loss {np.mean(losses):.4f} | valDice {v_d:.4f}"
                  f" valCl {v_cl:.4f} | best {best:.4f}@{best_ep} | "
                  f"{time.time()-t0:.0f}s", flush=True)
        if patience >= 40:
            print(f"  早停 @ ep{ep}", flush=True)
            break

    with open(os.path.join(OUT_DIR, "训练曲线.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "seg_I_T1.pt"),
                                     map_location=dev))
    model.eval()
    fd, fc, fh = evaluate(model, list(range(N)))
    vd, vc, vh = evaluate(model, list(va_idx))
    res = {"zero_shot_20f": {"dice": round(zd, 4), "cldice": round(zc, 4),
                              "hd95": round(zh, 2)},
           "finetuned_20f": {"dice": round(fd, 4), "cldice": round(fc, 4),
                              "hd95": round(fh, 2)},
           "finetuned_val6f": {"dice": round(vd, 4), "cldice": round(vc, 4),
                                "hd95": round(vh, 2)},
           "best_epoch": best_ep}
    json.dump(res, open(os.path.join(OUT_DIR, "评估.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[finetuned] 20帧 Dice={fd:.4f} clDice={fc:.4f} HD95={fh:.2f}")
    print(f"[finetuned] val6帧 Dice={vd:.4f} clDice={vc:.4f} HD95={vh:.2f}")
    print(f"DONE {time.time()-t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
