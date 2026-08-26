"""seg_common.py -- Segmentation commons: model / loss / augmentation / metrics

Design notes (paper Sec. 5):
  - ResUNet with fixed 3-channel input; modality ablation is done purely by
    channel replication so parameter counts are identical across arms.
  - Loss: L = DiceCE + 0.3 * (1 - clDice) + 0.1 * Boundary (ring-weighted BCE).
  - Augmentation: GPU batched affine (rotation/scale/translation) + flips,
    applied consistently across the three modalities; elastic deformation
    is deliberately disabled. Lightweight per-modality intensity jitter
    (gamma/brightness/noise).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

PAD_H, PAD_W = 656, 560  # 647,549 → 16 multiple


# ============================================================
# model : ResUNet (4layer , base 32)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        s = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + s, inplace=True)


class ResUNet(nn.Module):
    """input 3 channel → output 1 channel logits"""

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
        self.d4 = ConvBlock(base * 12, base * 4)   # 128(up) + 256(skip)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d3 = ConvBlock(base * 6, base * 2)    # 64(up) + 128(skip)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d2 = ConvBlock(base * 3, base)        # 32(up) + 64(skip)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)        # 32(up) + 32(skip)

        self.head = nn.Conv2d(base, 1, 1)

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
        return self.head(y)


# ============================================================
# skeleton + clDice (Shit et al. 2021)
# ============================================================
def soft_erode(img):
    if img.ndim == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    else:
        raise ValueError("soft_erode: 期望 4D 张量")


def soft_dilate(img):
    if img.ndim == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    else:
        raise ValueError("soft_dilate: 期望 4D 张量")


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=3):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(y_true, y_pred, iters=3):
    skel_pred = soft_skel(y_pred, iters)
    skel_true = soft_skel(y_true, iters)
    tprec = (torch.sum(skel_pred * y_true) + 1.0) / (torch.sum(skel_pred) + 1.0)
    tsens = (torch.sum(skel_true * y_pred) + 1.0) / (torch.sum(skel_true) + 1.0)
    return (2 * tprec * tsens) / (tprec + tsens)


def soft_dice(y_true, y_pred):
    inter = torch.sum(y_true * y_pred)
    smooth = 1.0
    return (2 * inter + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


# ============================================================
# loss (all GPU)
# ============================================================
def seg_loss(logits, targets, w_cldice=0.3, w_boundary=0.1):
    """logits: (B,1,H,W); targets: (B,1,H,W) 0/1
    BCE use logits form (autocast safe ); Dice/clDice in probability on compute """
    logits = logits.float()
    p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)

    dice_ce = 0.5 * (1.0 - soft_dice(targets, p)) \
        + 0.5 * F.binary_cross_entropy_with_logits(logits, targets)
    cldice_loss = 1.0 - soft_cldice(targets, p)

    # border boundary loss : target 2px ring weighted BCE ( height border boundary / )
    ring = F.max_pool2d(targets, 5, 1, 2) - targets
    w_map = 1.0 + 3.0 * ring
    boundary_loss = (F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none") * w_map).mean()

    loss = dice_ce + w_cldice * cldice_loss + w_boundary * boundary_loss
    with torch.no_grad():
        parts = {
            "dice": float(soft_dice(targets, (p > 0.5).float())),
            "cldice": float(soft_cldice(targets, p)),
            "dice_ce": float(dice_ce),
            "boundary": float(boundary_loss),
        }
    return loss, parts


# ============================================================
# GPU batch augmentation (three-modality consistent geometry + per-modality intensity jitter )
# ============================================================
def gpu_augment(x, y, rng):
    """x: (B,3,PAD_H,PAD_W) float32; y: (B,1,PAD_H,PAD_W) float32 (0/1)
    rng: numpy RandomState, ensure sequence reproducible
    """
    B, _, H, W = x.shape
    dev = x.device

    # random affine parameters
    ang = (rng.rand(B) * 30 - 15) * np.pi / 180.0
    scale = 0.9 + rng.rand(B) * 0.2
    tx = (rng.rand(B) * 2 - 1) * 0.05 * 2  # normalization translation
    ty = (rng.rand(B) * 2 - 1) * 0.05 * 2

    theta = np.zeros((B, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = np.cos(ang) * scale
    theta[:, 0, 1] = -np.sin(ang) * scale
    theta[:, 1, 0] = np.sin(ang) * scale
    theta[:, 1, 1] = np.cos(ang) * scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    theta = torch.from_numpy(theta).to(dev)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x_aug = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_aug = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # random flip (per- this , three-modality consistent )
    flip_h = rng.rand(B) < 0.5
    flip_v = rng.rand(B) < 0.5
    if flip_h.any():
        x_aug[flip_h] = torch.flip(x_aug[flip_h], dims=[3])
        y_aug[flip_h] = torch.flip(y_aug[flip_h], dims=[3])
    if flip_v.any():
        x_aug[flip_v] = torch.flip(x_aug[flip_v], dims=[2])
        y_aug[flip_v] = torch.flip(y_aug[flip_v], dims=[2])

    # per-modality intensity jitter (gamma/brightness /noise )
    for c in range(3):
        ch = x_aug[:, c:c + 1]
        g = torch.from_numpy(0.9 + rng.rand(B) * 0.2).to(dev).view(B, 1, 1, 1)
        b = torch.from_numpy(rng.rand(B) * 0.1 - 0.05).to(dev).view(B, 1, 1, 1)
        ch = torch.clamp(ch ** g + b, 0, 1)
        if rng.rand() < 0.5:
            ch = ch + torch.randn_like(ch, device=dev) * 0.01
        x_aug[:, c:c + 1] = torch.clamp(ch, 0, 1)

    return x_aug, (y_aug > 0.5).float()


# ============================================================
# hard metrics (evaluation use )
# ============================================================
def dice_coef(pred, target):
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return 2 * inter / denom if denom > 0 else 0.0


def hard_cldice(pred, target):
    """ skeleton clDice"""
    from skimage.morphology import skeletonize
    sp = skeletonize(pred)
    st = skeletonize(target)
    if sp.sum() == 0 and st.sum() == 0:
        return 1.0
    tprec = (sp & target).sum() / sp.sum() if sp.sum() > 0 else 0.0
    tsens = (st & pred).sum() / st.sum() if st.sum() > 0 else 0.0
    if tprec + tsens == 0:
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)


def hd95(pred, target):
    """95% Hausdorff distance"""
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    from scipy.ndimage import binary_erosion
    pb = pred ^ binary_erosion(pred)
    tb = target ^ binary_erosion(target)
    d2 = distance_transform_edt(~target)
    d1 = distance_transform_edt(~pred)
    # border boundary can as empty (mask is or fully ) → use mask
    if pb.sum() == 0:
        pb = pred
    if tb.sum() == 0:
        tb = target
    vals = np.concatenate([d2[pb].ravel(), d1[tb].ravel()])
    return float(np.percentile(vals, 95))


def connected_components_info(mask):
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0:
        return 0, 0.0
    sizes = np.array([(lab == i).sum() for i in range(1, n + 1)])
    return n, sizes.max() / mask.sum()


def evaluate_mask(pred, target):
    """pred/target: (H,W) bool"""
    if pred.sum() == 0:
        return {"dice": 0.0, "cldice": 0.0, "hd95": float("inf"),
                "n_comp_pred": 0, "n_comp_target": 0,
                "max_comp_ratio_pred": 0.0, "area_pred": 0,
                "area_target": int(target.sum())}
    n_comp_p, max_ratio_p = connected_components_info(pred)
    n_comp_t, _ = connected_components_info(target)
    return {
        "dice": dice_coef(pred, target),
        "cldice": hard_cldice(pred, target),
        "hd95": hd95(pred, target),
        "n_comp_pred": n_comp_p,
        "n_comp_target": n_comp_t,
        "max_comp_ratio_pred": max_ratio_p,
        "area_pred": int(pred.sum()),
        "area_target": int(target.sum()),
    }
