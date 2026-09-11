"""heart2_finetune.py -- second pig heart: ResUNet-I fine-tune with zero-shot
cross-heart control

Protocol identical to the first-heart training (step5_train_seg.py):
  input: per-frame p1-p99 normalized intensity -> [I,I,I] 3 channels -> pad
  loss: DiceCE + 0.3*(1-clDice) + 0.1*ring boundary BCE
  augmentation: GPU affine + flips + intensity jitter (same gpu_augment)
Fine-tune specifics:
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_common import (PAD_H, PAD_W, ResUNet, evaluate_mask, gpu_augment,
                        seg_loss, soft_cldice)

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\Project\CC_Project\激光散斑数据2\newdata"
RAW_I = os.path.join(ROOT, "整段数据_2641帧", "光强_intensity.npy")
MASK_DIR = os.path.join(ROOT, "40帧mask标注", "SegmentationClass")
PRETRAINED = (r"d:\Project\CC_Project\激光散斑pure\DAT解析结果\第三版执行"
              r"\segmentation\models\seg_I_seed0.pt")
OUT_DIR = os.path.join(ROOT, "finetune")
SEED = 2026
EPOCHS = 150
BATCH = 8
LR = 1e-4
H_RAW, W_RAW = 638, 528


def p1p99(frame):
    """原始强度 -> [0,1] (与第一颗心训练输入/显示变换一致)"""
    fin = np.isfinite(frame)
    lo, hi = np.percentile(frame[fin], [1, 99])
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


class SegDataset(Dataset):
    def __init__(self, images, masks, idx):
        self.images, self.masks = images[idx], masks[idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img3, msk = self.images[i], self.masks[i]
        pad_img = np.zeros((3, PAD_H, PAD_W), dtype=np.float32)
        pad_img[:, :img3.shape[1], :img3.shape[2]] = img3
        pad_msk = np.zeros((PAD_H, PAD_W), dtype=np.float32)
        pad_msk[:msk.shape[0], :msk.shape[1]] = msk
        return torch.from_numpy(pad_img), torch.from_numpy(pad_msk).unsqueeze(0)


def predict(model, x, dev):
    """x: (3,H,W) 0-1 -> pad 656x560 -> 裁回原始尺寸的概率图"""
    import torch.nn.functional as F
    xp = F.pad(x, (0, PAD_W - W_RAW, 0, PAD_H - H_RAW))
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p = torch.sigmoid(model(xp.unsqueeze(0).to(dev)).float())
    return p[0, 0, :H_RAW, :W_RAW].cpu().numpy()


def eval_frames(model, images, masks, dev):
    met = []
    for i in range(len(images)):
        pred = predict(model, torch.from_numpy(images[i]), dev) > 0.5
        met.append(evaluate_mask(pred, masks[i] > 0.5))
    return met


def agg(met):
    keys = ["dice", "cldice", "hd95"]
    out = {k: float(np.mean([m[k] for m in met])) for k in keys}
    out["dice_std"] = float(np.std([m["dice"] for m in met]))
    out["cldice_std"] = float(np.std([m["cldice"] for m in met]))
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {dev} | finetune ResUNet-I (heart2)", flush=True)
    t0 = time.time()

    # ---- 1. 构建 40 帧数据集 (p1-p99 强度 + VOC mask) ----
    names = sorted(f for f in os.listdir(MASK_DIR) if f.endswith(".png"))
    frame_ids = [int(n.split("_")[1].split(".")[0]) for n in names]
    assert len(names) == 40
    I = np.load(RAW_I, mmap_mode="r")
    images = np.zeros((40, 3, H_RAW, W_RAW), dtype=np.float32)
    masks = np.zeros((40, H_RAW, W_RAW), dtype=np.uint8)
    for i, (n, fi) in enumerate(zip(names, frame_ids)):
        f = p1p99(np.asarray(I[fi], dtype=np.float64))
        images[i, 0] = images[i, 1] = images[i, 2] = f
        m = np.array(Image.open(os.path.join(MASK_DIR, n)))
        masks[i] = ((m > 0) if m.ndim == 2 else (m.sum(-1) > 0)).astype(np.uint8)
    print(f"数据: 40 帧 p1-p99 归一化 | mask 前景均值 "
          f"{masks.sum(axis=(1,2)).mean():.0f}px", flush=True)

    # ---- 2. 固定划分 30/10 ----
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(40)
    tr_idx, va_idx = perm[:30], perm[30:]
    split = {"seed": SEED,
             "train": [frame_ids[i] for i in tr_idx],
             "val": [frame_ids[i] for i in va_idx]}
    with open(os.path.join(OUT_DIR, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"划分: train={len(tr_idx)} val={len(va_idx)} (val 帧: {split['val']})",
          flush=True)
    tr_ds = SegDataset(images, masks, tr_idx)
    va_ds = SegDataset(images, masks, va_idx)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=0, drop_last=True, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=0)

    # ---- 3. zero-shot: 第一颗心权重直接推理 ----
    model = ResUNet().to(dev)
    model.load_state_dict(torch.load(PRETRAINED))
    model.eval()
    zs_all = eval_frames(model, images, masks, dev)
    zs_val = [zs_all[i] for i in va_idx]
    print(f"\n[zero-shot] 全40帧  Dice={agg(zs_all)['dice']:.4f}±{agg(zs_all)['dice_std']:.4f}"
          f"  clDice={agg(zs_all)['cldice']:.4f}  HD95={agg(zs_all)['hd95']:.2f}", flush=True)
    print(f"[zero-shot] val10帧 Dice={agg(zs_val)['dice']:.4f}  "
          f"clDice={agg(zs_val)['cldice']:.4f}", flush=True)

    # ---- 4. finetune ----
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX := EPOCHS,
                                                       eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    aug_rng = np.random.RandomState(SEED)

    def fast_val():
        model.eval()
        ds, cs = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y in va_loader:
                x, y = x.to(dev), y.to(dev)
                p = torch.sigmoid(model(x).float())
                pred = (p > 0.5).float()
                for b in range(p.shape[0]):
                    inter = (pred[b] * y[b]).sum()
                    den = pred[b].sum() + y[b].sum()
                    ds.append(float(2 * inter / den) if den > 0 else 0.0)
                    cs.append(float(soft_cldice(y[b:b + 1], p[b:b + 1])))
        return float(np.mean(ds)), float(np.mean(cs))

    log_rows, best_score, best_epoch, patience = [], -1.0, 0, 0
    print(f"\n训练: epochs={EPOCHS} batch={BATCH} lr={LR} "
          f"(从 seg_I_seed0.pt 初始化)", flush=True)
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
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
                         "val_dice": v_d, "val_soft_cldice": v_cl,
                         "lr": float(sched.get_last_lr()[0])})
        if score > best_score:
            best_score, best_epoch, patience = score, ep, 0
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "seg_I_heart2.pt"))
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0:
            print(f"  ep {ep:3d} | loss {np.mean(losses):.4f} | valDice {v_d:.4f}"
                  f" valCl {v_cl:.4f} | best {best_score:.4f}@{best_epoch} | "
                  f"{time.time()-t0:.0f}s", flush=True)
        if patience >= 60:
            print(f"  早停 @ ep{ep}", flush=True)
            break

    with open(os.path.join(OUT_DIR, "训练曲线.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # ---- 5. 最终评估: finetune best vs zero-shot ----
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "seg_I_heart2.pt")))
    model.eval()
    ft_val = [eval_frames(model, images, masks, dev)[i] for i in va_idx]
    ft_all = eval_frames(model, images, masks, dev)
    res = {
        "protocol": {"init": "seg_I_seed0.pt (heart1)", "lr": LR,
                     "epochs": EPOCHS, "batch": BATCH, "seed": SEED,
                     "n_train": 30, "n_val": 10,
                     "input": "per-frame p1-p99 normalized intensity, [I,I,I]"},
        "best_epoch": best_epoch,
        "zero_shot_all40": agg(zs_all),
        "zero_shot_val10": agg(zs_val),
        "finetuned_val10": agg(ft_val),
        "finetuned_all40": agg(ft_all),
        "per_frame_val10": [{"frame": frame_ids[i],
                             **{k: ft_val[j][k] for k in
                                ("dice", "cldice", "hd95")}}
                            for j, i in enumerate(va_idx)],
    }
    with open(os.path.join(OUT_DIR, "评估.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n===== 结果汇总 =====")
    print(f"zero-shot  val10: Dice={res['zero_shot_val10']['dice']:.4f}  "
          f"clDice={res['zero_shot_val10']['cldice']:.4f}")
    print(f"finetuned  val10: Dice={res['finetuned_val10']['dice']:.4f}±"
          f"{res['finetuned_val10']['dice_std']:.4f}  "
          f"clDice={res['finetuned_val10']['cldice']:.4f}  "
          f"HD95={res['finetuned_val10']['hd95']:.2f}")
    print(f"zero-shot  all40: Dice={res['zero_shot_all40']['dice']:.4f}")
    print(f"finetuned  all40: Dice={res['finetuned_all40']['dice']:.4f}")
    print(f"best epoch {best_epoch} | 总耗时 {time.time()-t0:.0f}s")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
