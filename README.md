# CoroLSCI: Coronary Laser Speckle Contrast Imaging Analysis Pipeline

Code release for the paper: **full-pipeline analysis of ex-vivo pig heart laser speckle contrast imaging (LSCI)** — from raw `.dat` decoding, through coronary segmentation and motion registration, to quantitative coronary geometry and perfusion time-series analysis.

## Pipeline Overview

```
zhuxin.dat  (raw Pimsoft LSCI acquisition)
    │
    ▼  [1] export_pimsoft_dat.py
Four-modality stacks (2641 frames @ 44 fps, 647x549)
    Intensity / Variance / Contrast / Perfusion  (.npy)
    │
    ├────────────────────────────────────────────────┐
    ▼ [2] Drift test & data preparation              │
heart_mask, frame index, cardiac cycle detection,    │
temporal split (70/15/15), reference frame (2562)    │
    │                                                │
    ▼ [3] Coronary segmentation                      │
2D ResUNet on Intensity (Dice 0.9218, clDice 0.9914)│
Modality ablation: I / IV / IC / IVC / Gated         │
    │  masks for all 2641 frames                     │
    ├────────────────────────────────────────────────┘
    ▼ [4] Motion registration (intensity-driven, mask-weighted)
L1 guarded translation → L2 intra-cycle DIS → L3 anchor→ref DIS
Soft-mask weighted (sigma=2), per-frame independent
(warped-NCC 0.841, mask-Dice 0.925, residual 6.2 px)
    │
    ▼ [5] Coronary quantitative analysis
Geometry: area / skeleton length / mean diameter / branch count
Perfusion: mask-averaged raw perfusion time series (44 fps)
Outputs: 60 s videos (44 fps) + vector PDF figures
```

## Environment & Dependencies

Developed and tested on Windows 11 (Anaconda), Python **3.11.15**, CUDA GPU required for training and inference.

| Package | Version |
|---|---|
| python | 3.11.15 |
| torch | 2.11.0+cu128 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| opencv-python (cv2) | 4.13.0 |
| scikit-image (skimage) | 0.26.0 |
| matplotlib | 3.11.0 |
| pandas | 2.3.3 |
| Pillow (PIL) | 12.2.0 |

Optional (only used by auxiliary experiments, not required by the main pipeline): `nibabel 5.4.2`, `SimpleITK 2.5.6` (nnU-Net data conversion).

Install:

```bash
conda create -n corolsci python=3.11
conda activate corolsci
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy opencv-python scikit-image matplotlib pandas pillow
```

Fonts: **Times New Roman** for all paper figures (set in the rendering scripts); no CJK font needed for the English code release.

## Repository Layout

- **`CoroLSCI/`** — main code (English comments, recommended)
- **`zh/`** — identical code with original Chinese comments

```
code/
├── CoroLSCI/  (or zh/)
│   ├── 1_dat_parsing/
│   │   ├── export_pimsoft_dat.py        # .dat → 4-modality .npy stacks
│   │   └── preprocess_first10s.py       # first-10s preprocessing
│   ├── 2_preparation/
│   │   ├── step0_5_drift_test.py        # global drift test, heart_mask
│   │   ├── step1_3_4_index_cycles_split.py  # indexing, cycle detection, 70/15/15 split
│   │   └── step8_select_reference.py    # reference frame selection (frame 2562)
│   ├── 3_segmentation/
│   │   ├── prep_seg_dataset.py          # annotation dataset (200 labeled frames)
│   │   ├── seg_common.py                # ResUNet, DiceCE+clDice+boundary loss, GPU aug
│   │   ├── step5_train_seg.py           # ResUNet training (I/IV/IC/IVC, 3 seeds)
│   │   ├── step7_train_gated.py         # gated multimodal fusion baseline
│   │   ├── step6_aggregate_seg.py       # 3-seed aggregation + ablation table
│   │   └── step12c_task1_seg.py         # full-60s inference (2641 masks)
│   ├── 4_registration/
│   │   ├── prep_reg_input.py            # u8 input stacks, global normalization
│   │   ├── reg_common.py                # DIS flow, guarded ECC, warp, compose, NCC
│   │   ├── step12b_reg_segment.py       # per-segment L1+L2+L3 pipeline
│   │   ├── step12b_maskw_exp.py         # soft-mask weighted DIS (adopted, sigma=2)
│   │   └── step10_aggregate_reg.py      # QC aggregation
│   └── 5_coronary_analysis/
│       ├── step6_coronary_analysis_v2.py   # geometry metrics + perfusion time series
│       └── step6d_render_two_videos.py     # paper videos + final-frame PDF figures
└── README.md
```

## Detailed Usage

All scripts are standalone and resolve data paths relative to the project root (`SCRIPT_DIR`, i.e. the parent of the stage folder — place the stage folders back into one project root, or run from the original repository layout). Run them **in stage order**. Run time is measured on a single RTX-class GPU.

### Stage 1 — Data decoding

**1.1 `export_pimsoft_dat.py`** — decode the raw acquisition.
- Input: `zhuxin.dat` (Pimsoft binary; edit `DAT_FILE` at the top of the script if the name differs).
- Output: `decoded_outputs/full_length_2641_frames/*.npy` — four modality stacks (intensity / variance / contrast / perfusion, 2641×647×549, float), header info JSON, preview figure. (Folder and file names on disk are Chinese; see the literal strings in `zh/1_dat_parsing/export_pimsoft_dat.py`.)
- Config: `PREVIEW_SECOND` (preview frame).
- No CLI args (edit constants at the top).

**1.2 `preprocess_first10s.py`** — first-10s development-segment preprocessing and quality checks.
- Input: the decoded stacks.
- Output: first-10s preprocessing folder with QC figures.

### Stage 2 — Preparation

**2.1 `step0_5_drift_test.py`** — global drift test; derives the `heart_mask` used by every later stage.
- Output: drift-test statistics `.npz` (contains `heart_mask`).

**2.2 `step1_3_4_index_cycles_split.py`** — unified frame indexing, per-segment **cardiac cycle detection** (cycle bounds used by cycle-aware registration and phase analysis), and the temporal **70/15/15 train/val/test split** (sealed test set).
- Output: per-segment cycle-detection `.npz` + split files.

**2.3 `step8_select_reference.py`** — selects the unified registration **reference frame (global frame 2562)** from stability/quality criteria.
- Output: `reference/reference_frame.json`.

### Stage 3 — Coronary segmentation

**3.1 `prep_seg_dataset.py`** — pack the 200 expert-annotated frames into the training dataset.
- Output: `segmentation/annotated_200.npz` (images (200,3,H,W), masks (200,H,W), channel statistics, split ids).

**3.2 `seg_common.py`** — shared module, not run directly. Provides the ResUNet (3.45 M params), loss `DiceCE + 0.3·(1−clDice) + 0.1·boundary-BCE`, GPU batched augmentation (affine + flips + per-modality jitter; elastic deformation disabled), and hard metrics (Dice / clDice / HD95 / connected components).

**3.3 `step5_train_seg.py`** — train one configuration/seed.
```bash
python step5_train_seg.py --config I --seed 0 --epochs 400 --batch 16
# --config in {I, IV, IC, IVC}: channel assembly (identical capacity, ablation by replication)
# --seed 0/1/2: repeat for 3 seeds per config
```
- Input: `annotated_200.npz`.
- Output: `segmentation/models/seg_{config}_seed{seed}.pt`, metrics curves CSV, test-set JSON (Dice/clDice/HD95 per config).
- ≈ 30 min per run on GPU.

**3.4 `step7_train_gated.py`** — gated multimodal-fusion baseline (same CLI as 3.3).

**3.5 `step6_aggregate_seg.py`** — aggregate all runs into the ablation summary table (CSV + report).

**3.6 `step12c_task1_seg.py`** — full-60s inference with the adopted model; writes per-frame masks `segmentation_{seg}/masks_raw/*.npy` + structural metrics CSV + probability maps. This is the segmentation deliverable used downstream.

### Stage 4 — Motion registration

**4.1 `prep_reg_input.py`** — build globally normalized u8 registration inputs (pooled heart-region clip p0.5–p99.5, shared across frames so intensities stay comparable).
- Output: `registration/reg_*_I_u8.npy` (+V/C) per segment.

**4.2 `reg_common.py`** — shared module, not run directly. Field convention `(H,W,2)`: `[...,0]=dx, [...,1]=dy`; sampling `output[x,y] = source[x−dy, y−dx]`. Provides DIS optical flow (`dis_flow`, with warm-start `prev`), guarded ECC translation (`affine_ecc_guarded`: σ=4 smoothing, translation-only, |shift|≤10 px or reject), `warp_image / warp_mask`, `compose`, `compose_affine_dense`, `ncc`, field statistics.

**4.3 `step12b_reg_segment.py`** — baseline per-segment L1+L2+L3 pipeline:
```bash
python step12b_reg_segment.py --start 0 --n 440 --segname 0-10s --config I --workers 8
# repeat with --start 440/880/1320, --n 440/440/1321, --segname 10-20s / 20-30s / last-30s
```
- L1: guarded global translation (frame→reference); L2: intra-cycle DIS (frame→cycle anchor); L3: anchor→reference DIS. Composes into per-frame fields + QC JSON.

**4.4 `step12b_maskw_exp.py`** — **adopted registration** (soft-mask weighted DIS, σ=2): coronary mask × image as DIS input. Same CLI as 4.3 (plus `--reg_stack` to override the input stack path).
- Output: `fields_{seg}_I_maskw*.npy` + per-frame QC (warped-NCC, folding ratio, cycle closure).

**4.5 `step10_aggregate_reg.py`** — aggregate QC across segments into the paper tables.

### Stage 5 — Coronary quantitative analysis

**5.1 `step6_coronary_analysis_v2.py`** — compute the two paper indicator families for all 2641 frames:
- **Geometry** (from the adopted masks): area, skeleton length, mean diameter (distance transform sampled on skeleton), branch count (skeleton pixels with >2 neighbors).
- **Perfusion** (functional): mask-averaged time series computed **directly on the original perfusion stack** (perfusion data is only sampled, never used in registration training).
- Output: coronary analysis metrics CSV (2641 rows × geometry + perfusion columns).

**5.2 `step6d_render_two_videos.py`** — render the two 60 s paper videos at native 44 fps:
- Video 1: Intensity + mask overlay | structural metrics (segmented y-axis with labeled value ranges).
- Video 2: Perfusion + mask overlay | mean perfusion time series.
- The last frame of each is exported as a **vector PDF (Times New Roman)** for the paper.
- ≈ 20 min per video (CPU rendering).

## Key Results

### Segmentation (test set, 30 frames)

| Method | Dice | clDice | Inference (ms) |
|---|---|---|---|
| **ResUNet, Intensity only (adopted)** | **0.9218** | **0.9914** | 10.7 |
| ResUNet, I+V+C | 0.9178 | 0.9914 | — |
| nnU-Net-style architecture | 0.8251 | 0.9784 | 5.3 |

Modality ablation (3 seeds): pure Intensity is best; adding Variance/Contrast consistently degrades Dice (speckle decorrelation amplifies noise in derived statistics).

### Registration (0-10s dev split)

| Method | mask-Dice | NCC | Residual motion (px) |
|---|---|---|---|
| **DIS + soft mask s2 (adopted)** | **0.9253** | 0.8407 | **6.23** |
| + edge refinement | 0.9414 | 0.8434 | 6.23 |
| + per-frame selection (final) | 0.9292 | 0.8396 | — |
| CNN + ConvLSTM (best DL) | 0.8674 | **0.8838** | 7.28 |
| 3D VoxelMorph variants | 0.63–0.88 | 0.55–0.86 | 7.4–17.5 |
| FFD / Farneback / Flow Matching | 0.63–0.86 | 0.76–0.81 | 13.9–20.6 |

Residual ~6 px motion is the physical limit of brightness-matching registration under speckle decorrelation: perfusion-driven intensity flicker is the signal itself, not noise.

### Coronary analysis (full 60 s, 2641 frames)

- **Geometry** (structural indicators): area, skeleton length, mean diameter, branch count — for stenosis / structural lesion screening.
- **Perfusion** (functional indicator): mask-averaged raw perfusion time series at 44 fps.

## Data & Paths

Scripts resolve paths relative to the project root (`SCRIPT_DIR`). Expected data layout (Chinese folder names are part of the on-disk contract and kept in the code):

```
decoded_outputs/
├── full_length_2641_frames/    # decoded 4-modality stacks
│   ├── intensity.npy
│   ├── variance.npy
│   ├── contrast.npy
│   └── perfusion.npy
├── drift_test/                 # drift stats, heart_mask
└── v3_execution/               # all experiment outputs
    ├── segmentation*/          # masks, metrics
    ├── registration/           # displacement fields, QC
    └── unified_60s_results/    # videos, CSV metrics, PDF figures
```

Folder and file names in the diagram are descriptive English; the actual on-disk names are Chinese (they are part of the data contract and appear as literal strings inside the scripts — see the `zh/` code for the exact strings).

Raw `.dat` file (`zhuxin.dat`) is not included in this repository (size/licensing); contact the authors for access.

## Citation

If you use this code, please cite the accompanying paper.
