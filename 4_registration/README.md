# Stage 4 — Motion Registration: ABANDONED PRELIMINARY WORK

**Status: not part of the published indicator path.**

This stage was fully implemented and benchmarked during development, and then
**abandoned** for the published pipeline. It is retained in the repository for
auditability only. The manuscript (Section II-F) reports the evaluation that
led to the abandonment:

- Brightness-matching registration (guarded ECC translation, DIS optical flow,
  soft-mask weighted DIS, plus FFD / Farneback / ConvLSTM / flow-matching
  alternatives) exhibits a **residual-motion floor of about 6 px** on this data
  class, because perfusion-driven intensity flicker is itself the signal and
  any brightness-matching method perceives it as irreducible apparent motion.
- Computing indicators in a warped space would additionally **resample the
  measurand itself** — deforming areas and diameters and mixing neighboring
  speckle statistics into perfusion values.

**Consequence for the published pipeline:** the mask and the four modality
stacks of each frame are acquired synchronously, so the structural and the
perfusion-surrogate indicators are intrinsically aligned and are computed in
**original coordinates** on unsampled data. No registration stage exists
anywhere in the indicator path.

## Historical content

| File | Purpose |
|---|---|
| `prep_reg_input.py` | Build globally normalized u8 registration inputs |
| `reg_common.py` | DIS flow, guarded ECC translation, warp/compose/NCC, field statistics |
| `step12b_reg_segment.py` | Baseline per-segment L1+L2+L3 pipeline |
| `step12b_maskw_exp.py` | Soft-mask weighted DIS (best abandoned variant, sigma=2) |
| `step10_aggregate_reg.py` | QC aggregation over segments |

Results reproduced here match the development reports (mask-Dice 0.9253,
warped-NCC 0.8407, residual 6.2 px on the 0-10 s development split).
These numbers are *preliminary diagnostics*, not paper metrics.
