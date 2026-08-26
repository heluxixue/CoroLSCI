"""step7_train_gated.py -- Gated multimodal fusion baseline

Trains a gated-fusion segmentation model as an additional multimodal
baseline for the modality ablation study.
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_common import (ConvBlock, ResUNet, seg_loss, gpu_augment, soft_cldice,
                        evaluate_mask, PAD_H, PAD_W)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "segmentation")
MODEL_DIR = os.path.join(BASE_DIR, "models")
METRIC_DIR = os.path.join(BASE_DIR, "metrics")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

H_RAW, W_RAW = 647, 549
W_GATE = 0.01  #  then weight


# ============================================================
# model : total shared encoder + 3 per-modality head + per- pixel softmax gated
# ============================================================
class GatedFusionNet(nn.Module):
    """structure and ResUNet same encode /decode , d1 place branch 3 head + gated """

    def __init__(self, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        self.e1 = ConvBlock(base, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d4 = ConvBlock(base * 12, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d3 = ConvBlock(base * 6, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d2 = ConvBlock(base * 3, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)

        # per-modality head (I / V / C)
        self.heads = nn.ModuleList([nn.Conv2d(base, 1, 1) for _ in range(3)])
        # gated head : d1 → 3 channel logits → per- pixel softmax
        self.gate = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.e1(x0)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))

        y = self.d4(torch.cat([self.up4(xb), x4], 1))
        y = self.d3(torch.cat([self.up3(y), x3], 1))
        y = self.d2(torch.cat([self.up2(y), x2], 1))
        y = self.d1(torch.cat([self.up1(y), x1], 1))

        zs = torch.cat([h(y) for h in self.heads], 1)      # (B,3,H,W) logits
        g = F.softmax(self.gate(y), dim=1)                  # (B,3,H,W) weight
        p_final = (g * torch.sigmoid(zs)).sum(1, keepdim=True)
        return p_final, g, zs


def fusion_loss(p_final, g, targets, w_cldice=0.3, w_boundary=0.1, w_gate=W_GATE):
    """in fusion probability on seg_loss ( logit autocast safe ) + then """
    p = p_final.float().clamp(1e-7, 1 - 1e-7)
    logits = torch.log(p / (1 - p))
    loss, parts = seg_loss(logits, targets, w_cldice=w_cldice, w_boundary=w_boundary)
    entropy = -(g * torch.log(g + 1e-7)).sum(1).mean()  # per- pixel ( small )
    loss = loss + w_gate * entropy
    parts["gate_entropy"] = float(entropy.detach())
    return loss, parts


# ============================================================
class SegDataset(Dataset):
    """and No. 5step consistent : , augmentation in GPU training ring in done """

    def __init__(self, images, masks, split_ids, split):
        idx = [i for i, s in enumerate(split_ids) if s == split]
        self.images = images[idx]
        self.masks = masks[idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img3, msk = self.images[i], self.masks[i]
        pad_img = np.zeros((3, PAD_H, PAD_W), dtype=np.float32)
        pad_img[:, :img3.shape[1], :img3.shape[2]] = img3
        pad_msk = np.zeros((PAD_H, PAD_W), dtype=np.float32)
        pad_msk[:msk.shape[0], :msk.shape[1]] = msk
        return torch.from_numpy(pad_img), torch.from_numpy(pad_msk).unsqueeze(0)


# ============================================================
def fast_val(model, val_loader, device):
    """GPU fast validation : per- image hard Dice + clDice (in fusion probability on )"""
    model.eval()
    dices, cldices = [], []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p, _, _ = model(x)
            p = p.float()
            pred = (p > 0.5).float()
            for b in range(p.shape[0]):
                pb, tb = pred[b], y[b]
                inter = (pb * tb).sum()
                denom = pb.sum() + tb.sum()
                dices.append(float(2 * inter / denom) if denom > 0 else 0.0)
                cldices.append(float(soft_cldice(tb[None], p[b:b + 1])))
    return float(np.mean(dices)), float(np.mean(cldices))


def hard_val(model, val_loader, device):
    """ hard metrics validation , 647×549"""
    model.eval()
    met = {"dice": [], "cldice": [], "hd95": []}
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)[0].float().cpu().numpy()
            ys = y.cpu().numpy()
            for b in range(probs.shape[0]):
                m = evaluate_mask(probs[b, 0, :H_RAW, :W_RAW] > 0.5,
                                  ys[b, 0, :H_RAW, :W_RAW] > 0.5)
                for k in met:
                    met[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in met.items()}


# ============================================================
def train_one_run(seed, epochs, batch_size, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}\n训练 GatedFusion seed={seed}\n{'='*60}", flush=True)
    data = np.load(os.path.join(BASE_DIR, "annotated_200.npz"))
    images, masks = data["images"], data["masks"]

    split_csv = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "dataset_split.csv")
    with open(split_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    splits = [r["split"] for r in rows]

    train_ds = SegDataset(images, masks, splits, "train")
    val_ds = SegDataset(images, masks, splits, "val")
    test_ds = SegDataset(images, masks, splits, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    model = GatedFusionNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数: {n_params/1e6:.2f}M  train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)}  λ_gate={W_GATE}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.RandomState(seed)

    log_rows = []
    best_score, best_epoch, patience = -1.0, 0, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        losses = {"total": [], "dice": [], "cldice": [], "gate_entropy": []}
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = gpu_augment(x, y, rng)
            with torch.amp.autocast("cuda"):
                p_final, g, _ = model(x)
                loss, parts = fusion_loss(p_final, g, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses["total"].append(float(loss.detach()))
            losses["dice"].append(parts["dice"])
            losses["cldice"].append(parts["cldice"])
            losses["gate_entropy"].append(parts["gate_entropy"])
        sched.step()

        v_d, v_cl = fast_val(model, val_loader, device)
        score = 0.5 * v_d + 0.5 * v_cl

        row = {
            "epoch": ep, "train_loss": np.mean(losses["total"]),
            "train_dice": np.mean(losses["dice"]), "train_cldice": np.mean(losses["cldice"]),
            "train_gate_entropy": np.mean(losses["gate_entropy"]),
            "val_dice": v_d, "val_soft_cldice": v_cl, "lr": float(sched.get_last_lr()[0]),
        }
        if ep % 5 == 0:
            h = hard_val(model, val_loader, device)
            row["val_hard_cldice"] = h["cldice"]
            row["val_hd95"] = h["hd95"]
            row["val_hard_dice"] = h["dice"]
        else:
            row["val_hard_cldice"] = ""
            row["val_hd95"] = ""
            row["val_hard_dice"] = ""
        log_rows.append(row)

        if score > best_score:
            best_score, best_epoch, patience = score, ep, 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"gated_seed{seed}.pt"))
        else:
            patience += 1

        if ep % 10 == 0 or patience == 0:
            print(f"  ep {ep:3d} | loss {np.mean(losses['total']):.4f} | "
                  f"valDice {v_d:.4f} valClDice {v_cl:.4f} | gateH "
                  f"{np.mean(losses['gate_entropy']):.3f} | "
                  f"best {best_score:.4f}@{best_epoch} | {time.time()-t0:.0f}s", flush=True)
        if patience >= 60:
            print(f"  早停 (patience 60) @ epoch {ep}", flush=True)
            break

    csv_path = os.path.join(METRIC_DIR, f"gated_seed{seed}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # ============ test set hard metrics evaluation ============
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"gated_seed{seed}.pt")))
    model.eval()
    per_frame = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)[0].float().cpu().numpy()
            ys = y.cpu().numpy()
            for b in range(probs.shape[0]):
                m = evaluate_mask(probs[b, 0, :H_RAW, :W_RAW] > 0.5,
                                  ys[b, 0, :H_RAW, :W_RAW] > 0.5)
                per_frame.append(m)

    agg = {k: float(np.nanmean([m[k] for m in per_frame])) for k in per_frame[0]}
    agg["dice_std"] = float(np.std([m["dice"] for m in per_frame]))
    agg["cldice_std"] = float(np.std([m["cldice"] for m in per_frame]))
    agg["best_epoch"] = best_epoch
    agg["n_params"] = n_params
    agg["config"] = "gated"
    agg["seed"] = seed
    eval_path = os.path.join(EVAL_DIR, f"gated_seed{seed}_test.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_frame": per_frame}, f, indent=1,
                  ensure_ascii=False)

    print(f"  [TEST] Dice={agg['dice']:.4f}±{agg['dice_std']:.4f}  "
          f"clDice={agg['cldice']:.4f}±{agg['cldice_std']:.4f}  HD95={agg['hd95']:.1f}",
          flush=True)
    print(f"  -> {eval_path}", flush=True)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    train_one_run(args.seed, args.epochs, args.batch, device)
