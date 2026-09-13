# 6_longitudinal — Scripts Reproducing the Paper's Core Numbers

These scripts reproduce the quantitative results of the manuscript
"CoroLSCI: Technical Feasibility of Longitudinal Coronary-Region Structure
and Perfusion-Surrogate Monitoring During Ex Vivo Heart Perfusion"
(IEEE Access submission). Hard-coded paths point to the local data layout of
the authoring machine (the raw recordings are not redistributable at full
size; the decoded stacks are available from the corresponding author and a
17-frame sample is released at `heluxixue/raw-data-for-CoroLSCI`).
Edit the `ROOT`/`OUT` constants to point at your copy of the decoded data.

Expected outputs are archived in `../expected_outputs/` so the paper's
numbers can be verified without re-running the (GPU-heavy) analyses.

| Paper number | Script | Expected output |
|---|---|---|
| Calibration curves (0.810→0.918 T1; 0.480→0.887/0.898 H2; Table 3) | `adaptation_curve.py` | `adaptation_curve.json` |
| Longitudinal decline −22.0%, block-summary bootstrap 95% CI 19.7–24.1% (canonical: `compute_review4.py`) | `block_summary_bootstrap.py` (legacy reconstruction) / `compute_review4.py` | `block_summary_bootstrap.json` (authoring-time) / `review4_stats.json` (canonical) |
| Session CVR table (Table 5) + ΔCVR +5.4% vs ΔP −22.0%, Pearson −0.70 / Spearman −0.10 (n=5) | `cvr_alignment.py` | `cvr_sessions.json` |
| Window-robustness of CVR (<0.001 mmHg·min/mL over ±30 s) + per-minute CF trajectory | `cvr_robustness.py` | (console table) |
| Area equivalence (Δ=−2.1%, 90% CI [−2.15,−1.98]% within ±5%) + HD95 + stratified Dice | `equivalence_hd95.py` | `equivalence_hd95.json` |
| Normalization sensitivity (C1/C2/C3 declines 22.0/22.0/22.9%) + background-referenced ratio | `sensitivity_analysis.py` | `sensitivity_analysis.json` |
| Cross-seed SD of session-level area (1–3 mm²) | `seed_variance.py` | `seed_variance.json` |
| Within-session repeatability coefficient (72–85 a.u.) + frame-level slope CIs | `bootstrap_dissociation.py` | `bootstrap_dissociation.json` |
| Fig. 6 / Fig. 9 (P_mfr series, manufacturer convention) | `fig69_mfr_fix.py` | `fig6_perfusion.pdf`, `fig9_perfusion_heart2.pdf` |
| Fig. 10 (longitudinal trajectory) | `figx_draft.py` | `fig10_longitudinal.pdf` |
| Fig. 11 (CVR reference) | `fig11_cvr.py` | `fig11_cvr.pdf` |
| Heart-2 fine-tune (30 frames; val Dice 0.893, zero-shot 0.480) | `heart2_finetune.py` | (model + eval JSON) |
| T1 close-up fine-tune (14 train / 6 held out; held-out 0.912) | `t1_finetune.py` | (model + eval JSON) |
| FLOPs / latency / memory (Table 3 companion numbers) | `measure_paper_metrics.py` | `paper_metrics_results.json` |
| Cross-session ±1-px perturbation, block-count sensitivity, invalid-pixel fractions, unrounded terminal decline (17.3%) | `compute_review3_stats.py` | `review3_stats.json` |
| T1 normalization×fine-tune 2×3 table + skeleton-anchored stratified Dice | `compute_review3_gpu.py` | `review3_gpu.json` |
| Round-4: per-session-calibration ±1-px perturbation (−2.6%/−1.3%), canonical bootstrap (12 blocks, CI 19.7–24.1), block sensitivity 6–33, provenance table | `compute_review4.py` | `review4_stats.json` |
| Round-4: contour-based HD95 (2.83/3.03/2.87 px), consensus coverage, extra-pixel split (23,758 disagreement-zone / 12,523 both-background) | `compute_review4_gpu.py` | `review4_gpu.json` |
| Fig. 6 / Fig. 10 geometry series in physical units (mm) | `fig58_units_fix.py` | `fig5_geometry.pdf`, `fig8_geometry_heart2.pdf` |
| Fig. 3 pipeline schematic (reference-style v4, full-width) | `fig1_pipeline_v4.py` | `fig1_pipeline.pdf` |

Notes on conventions:

- **P_mfr** is the manufacturer-convention, mask-averaged perfusion-surrogate
  index (Eq. 2 of the paper), in instrument-defined arbitrary units. It is
  NOT the per-pixel mean of the instrument's perfusion stack (that older
  quantity is negative-valued and was replaced in the published figures).
- `block_summary_bootstrap.py` is a protocol reconstruction (twelve 5-s
  blocks per session, 4000 within-session block resamples, seed 2026); it
  reproduces the point estimate (−22.0/−22.1%) and the archived protocol.
  The exact authoring-time run is archived in
  `expected_outputs/block_summary_bootstrap.json` (median 22.3%, CI
  [20.7, 23.8]); minor differences from a re-run reflect block-boundary and
  seed details of the archived run. The paper cites the archived numbers.
- The inter-expert three-way Dice values (0.9108 / 0.9199 / 0.9178) were
  computed on the fresh 20-frame set; the frame-level evaluation protocol is
  described in the paper (Section III-B) and the results are archived in
  `expected_outputs/p2_three_way.json`.
- The internal sealed-test Dice 0.9207±0.0056 and clDice 0.9911 come from
  the Stage-3 training outputs (`3_segmentation/step5_train_seg.py` +
  `step6_aggregate_seg.py`); the 0.9218/10.7 ms values that appear in older
  repository versions were development-stage measurements on a different
  evaluation basis and are superseded by the sealed-test values above.


Round-3 convention updates (2026-09-11):
- The stratified thick/thin analysis in `equivalence_hd95.json` used a distance-band
  definition (distance from the consensus centerline) that does not isolate vessel
  caliber; it is superseded by the skeleton-anchored local-radius stratification in
  `review3_gpu.json` (thin: radius <= 3 px; within-consensus agreement 0.981 thin /
  0.987 thick; residual error mass outside the consensus region).
- The terminal-interval drop is 17.3% (unrounded 17.255 from session means), replacing
  the earlier 17.2% display rounding.

Round-4 updates (2026-09-12):
- The ±1-px perturbation in `review3_stats.json` used one fixed mm/px value for all
  sessions; superseded by `review4_stats.json` (per-session header calibrations, full
  precision). Corrected endpoints: net area change −2.61% (erode) / −1.32% (dilate),
  both inside the ±5% tolerance; P_mfr decline 21.8–22.4%.
- The canonical decline CI is now generated by `compute_review4.py` (133 stride-20
  samples, contiguous blocks with remainder in the final block, B=4000, seed 2026):
  22.0% [19.7, 24.1]. The older archived interval [20.7, 23.8] came from the
  authoring-time run; the paper cites the reproducible version.
- HD95 values 1.02/1.15/1.08 px were foreground-percentile distances, not contour
  distances; superseded by contour-based 2.83/3.03/2.87 px in `review4_gpu.json`.
- Stratified within-consensus Dice renamed to consensus-region coverage (0.964 thin /
  0.974 thick); extra prediction pixels split 23,758 (expert-disagreement zone) +
  12,523 (both-experts-background).
