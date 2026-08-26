"""prep_seg_dataset.py -- Annotation dataset preparation

Packs the 200 expert-annotated frames (three-modality images + masks)
into annotated_200.npz with per-channel statistics.
"""
import os, csv
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "整段数据_2641帧_预处理")
MASK_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "200mask", "SegmentationClass")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "segmentation")
os.makedirs(OUT_DIR, exist_ok=True)

print("0. 读取 dataset_split.csv")
split_path = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "dataset_split.csv")
rows = []
with open(split_path, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows.append(row)
frame_ids = [int(r["frame_id"]) for r in rows]
print(f"  标注帧数: {len(frame_ids)}")

print("1. 提取三模态 (memmap)")
I = np.load(os.path.join(PREP_DIR, "光强_intensity_p1p99.npy"), mmap_mode="r")
V = np.load(os.path.join(PREP_DIR, "方差_variance_p1p99.npy"), mmap_mode="r")
C = np.load(os.path.join(PREP_DIR, "散斑对比度_contrast_p1p99.npy"), mmap_mode="r")

H, W = 647, 549
images = np.zeros((len(frame_ids), 3, H, W), dtype=np.float32)
masks = np.zeros((len(frame_ids), H, W), dtype=np.uint8)

for i, fi in enumerate(frame_ids):
    images[i, 0] = I[fi]
    images[i, 1] = V[fi]
    images[i, 2] = C[fi]

    m = np.array(Image.open(os.path.join(MASK_DIR, f"{fi}.png")))
    if m.ndim == 3:
        m_bin = (m.sum(axis=-1) > 0).astype(np.uint8)
    else:
        m_bin = (m > 0).astype(np.uint8)
    masks[i] = m_bin

print("2. 校验")
print(f"  images: {images.shape} {images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
print(f"  masks: {masks.shape}, 前景像素: {masks.sum()}, 每帧面积 mean={masks.sum(axis=(1,2)).mean():.0f}")
assert masks.sum() > 0, "mask 全空!"

# channel statistics ( training )
ch_means = images.mean(axis=(0, 2, 3))
ch_stds = images.std(axis=(0, 2, 3))
print(f"  通道均值: I={ch_means[0]:.4f} V={ch_means[1]:.4f} C={ch_means[2]:.4f}")
print(f"  通道std:  I={ch_stds[0]:.4f} V={ch_stds[1]:.4f} C={ch_stds[2]:.4f}")

np.savez_compressed(
    os.path.join(OUT_DIR, "annotated_200.npz"),
    images=images, masks=masks, frame_ids=np.array(frame_ids),
    ch_means=ch_means, ch_stds=ch_stds,
)
print(f"3. 保存 -> annotated_200.npz")
print("完成!")
