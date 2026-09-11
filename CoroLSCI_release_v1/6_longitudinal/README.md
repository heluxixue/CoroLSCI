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
| Longitudinal decline −22.0%, block-summary bootstrap 95% CI 20.7–23.8% | `block_summary_bootstrap.py` | `block_summary_bootstrap.json` |
| Session CVR table (Table 5) + ΔCVR +5.4% vs ΔP −22.0%, Pearson −0.70 / Spearman −0.10 (n=5) | `cvr_alignment.py` | `cvr_sessions.json` |
| Window-robustness of CVR (<0.001 mmHg·min/mL over ±30 s) + per-minute CF trajectory | `cvr_robustness.py` | (console table) |
| Area equivalence (Δ=−2.1%, 90% CI [−2.15,−1.98]% within ±5%) + HD95 + stratified Dice | `equivalence_hd95.py` | `equivalence_hd95.json` |
| Normalization sensitivity (C1/C2/C3 declines 22.0/22.0/22.9%) + background-referenced ratio | `sensitivity_analysis.py` | `sensitivity_analysis.json` |
| Cross-seed SD of session-level area (1–3 mm²) | `seed_variance.py` | `seed_variance.json` |
| Within-session repeatability coefficient (72–85 a.u.) + frame-level slope CIs | `bootstrap_dissociation.py` | `bootstrap_dissociation.json` |
| Fig. 1 (pipeline schematic) | `fig1_pipeline_v2.py` | `fig1_pipeline.pdf` |
| Fig. 6 / Fig. 9 (P_mfr series, manufacturer convention) | `fig69_mfr_fix.py` | `fig6_perfusion.pdf`, `fig9_perfusion_heart2.pdf` |
| Fig. 10 (longitudinal trajectory) | `figx_draft.py` | `fig10_longitudinal.pdf` |
| Fig. 11 (CVR reference) | `fig11_cvr.py` | `fig11_cvr.pdf` |
| Heart-2 fine-tune (30 frames; val Dice 0.893, zero-shot 0.480) | `heart2_finetune.py` | (model + eval JSON) |
| T1 close-up fine-tune (14 train / 6 held out; held-out 0.912) | `t1_finetune.py` | (model + eval JSON) |
| FLOPs / latency / memory (Table 3 companion numbers) | `measure_paper_metrics.py` | `paper_metrics_results.json` |

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
