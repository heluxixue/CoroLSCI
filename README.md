# CoroLSCI — Coronary Laser Speckle Contrast Imaging Analysis Pipeline

Code release for the manuscript:

> **CoroLSCI: Technical Feasibility of Longitudinal Coronary-Region
> Structure and Perfusion-Surrogate Monitoring During Ex Vivo Heart
> Perfusion** — submitted to IEEE Access.

This repository implements the full software measurement framework: raw
Pimsoft `.dat` decoding, per-frame coronary-region segmentation, session
adaptive normalization with quality control, and the closed-form dual
indicator family (structural geometry in millimeter units + an LSCI-derived
perfusion-surrogate index) computed frame by frame on unsampled data — plus
the adaptation, uncertainty, and longitudinal-analysis scripts that produce
every number in the paper.

## What the pipeline is (and is not)

The published indicator path is:

```
raw .dat  ->  decoding (4 synchronous modality stacks)  ->  session-adaptive
normalization + QC  ->  per-frame segmentation (ResUNet, 3.45 M)  ->  dual
indicators (Eq. 2) in original coordinates, frame by frame, unsampled
```

**No registration stage exists anywhere in the indicator path.** Stage 4 of
this repository (`4_registration/`) is *abandoned preliminary work*, retained
for auditability: brightness-matching registration shows a ~6-px residual
motion floor under speckle decorrelation and would resample the measurand
itself (paper Section II-F). See `4_registration/README.md`.

## Key results (sealed-test values; see `6_longitudinal/README.md` for the
script that produces each number)

| Quantity | Value |
|---|---|
| Internal sealed test set, heart-1 session T2 (temporally separated) | Dice 0.9207 ± 0.0056, clDice 0.9911 |
| Single-frame inference | 5.1 ms (196 frames/s; RTX 4090, fp32, batch 1) |
| End-to-end per-frame computation (decode 7.1 + inference 5.1 + indicators 7.1 ms) | 19.3 ms vs the 22.7-ms frame budget of the 44-fps acquisition |
| Parameters / FLOPs | 3.45 M / 100.5 G per forward pass |
| Inter-expert vs algorithm agreement (fresh 20-frame set) | E1–E2 0.9108; Algo–E1 0.9199; Algo–E2 0.9178; HD95 1.02–1.15 px |
| Five-frame calibration, working-distance shift (T1) | 0.810 (zero-shot) → 0.918 (held-out mean) |
| Five-frame calibration, cross-heart shift (heart 2) | 0.480 (zero-shot) → 0.887; 0.898 at 30 frames |
| Longitudinal, heart 1 (102 min, single heart, exploratory) | area within ±5% equivalence margin (Δ = −2.1%, 90% CI [−2.15, −1.98]%); perfusion-surrogate −22.0% (block-summary bootstrap 95% CI 20.7–23.8%) |
| Rig-log reference (CVR, independent sensor path) | ΔCVR +5.4% vs ΔP −22.0%; terminal flow drop −13.7%; Pearson −0.70, Spearman −0.10 (n = 5 sessions, descriptive) |

> Numbers that appear in older repository versions (Dice 0.9218, 10.7 ms
> inference) were development-stage measurements on a different evaluation
> basis. The values above are the paper's sealed-test values.

## Repository layout

```
├── 1_dat_parsing/          .dat -> 4-modality .npy stacks (manufacturer's note)
├── 2_preparation/          drift test, cardiac-cycle detection, temporal split
├── 3_segmentation/         ResUNet training/ablation, full-60s inference
├── 4_registration/         ABANDONED preliminary work (see its README)
├── 5_coronary_analysis/    geometry + perfusion indicator computation, videos
├── 6_longitudinal/         paper-figure & statistics scripts (P_mfr, bootstrap,
│                           CVR alignment, sensitivity, equivalence, fine-tunes)
├── expected_outputs/       archived JSON source data for every paper number
├── environment.txt         frozen environment
└── README.md
```

## Environment

Python 3.11.15, PyTorch 2.11.0+cu128 (CUDA GPU required for training and
inference), NumPy 2.4.4, SciPy 1.17.1, OpenCV 4.13.0, scikit-image 0.26.0,
Matplotlib 3.11.0, pandas 2.3.3, Pillow 12.2.0 (full list in
`environment.txt`). Optional, auxiliary experiments only: nibabel 5.4.2,
SimpleITK 2.5.6.

```bash
conda create -n corolsci python=3.11
conda activate corolsci
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy opencv-python scikit-image matplotlib pandas pillow
```

## Data

- Each Pimsoft `.dat` stores, per frame, two float64 images (variance and
  intensity) plus a versioned binary header carrying per-frame acquisition
  parameters, the instrument's coherence factor `β` and signal gain `k`, and
  the session's mm/px calibration. Decoding follows the manufacturer's
  technical note 44-00208-02.
- Modalities: intensity `I`, variance `V`, speckle contrast
  `C = β·√|V|/I·sgn(V)`, instrument-convention perfusion `P = k·(1/C − 1)`.
- Perfusion surrogate (Eq. 2, mask-averaged manufacturer convention):
  `P_mfr(t) = k·(1/C̄_t − 1)` with `C̄_t = β·√|V̄_t|/Ī_t`, where `Ī_t` and
  `V̄_t` are the mean intensity and variance **over the coronary mask** of
  frame `t` (not the mean of per-pixel P). Units are instrument-defined
  arbitrary units; `P_mfr` is not calibrated to absolute flow.
- Structural indicators (Eq. 3–5): area `A`, skeleton length `ℓ`, mean
  diameter `d̄`, branch count `n_br`, converted to physical units with the
  per-session header mm/px calibration.
- Data hierarchy: heart (independent biological unit) → session (repeated
  measurement) → cardiac-cycle/time block (correlated replicates) → frame
  (technical observation, not an independent n).
- A 17-frame raw sample is released at
  [`heluxixue/raw-data-for-CoroLSCI`](https://github.com/heluxixue/raw-data-for-CoroLSCI).
  The complete decoded stacks (>13 GB per modality) are available from the
  corresponding author on reasonable request.

## Reproducing the paper numbers

1. Decode the raw recordings (`1_dat_parsing/export_pimsoft_dat.py`).
2. Prepare splits/cycles (`2_preparation/`), train the segmentation network
   (`3_segmentation/step5_train_seg.py --config I --seed 0`), run full-session
   inference (`3_segmentation/step12c_task1_seg.py`).
3. Compute indicators (`5_coronary_analysis/step6_coronary_analysis_v2.py`)
   and run the longitudinal/statistics scripts in `6_longitudinal/`; each
   script names its expected output, archived in `expected_outputs/`.

Scripts resolve data paths relative to `SCRIPT_DIR`; the on-disk data layout
uses the authoring machine's folder names (some in Chinese, a historical
contract kept verbatim inside the scripts). Edit the `ROOT`/`OUT` constants
at the top of each script for your own layout.

## License and citation

If you use this code or the released data sample, please cite the
accompanying manuscript. Contact the corresponding author for data access.
