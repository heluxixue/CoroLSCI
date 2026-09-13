"""fig11_cvr.py -- paper figure fig11_cvr.pdf (rig CVR reference, A/B/C)

Style consistent with fig10_longitudinal: Times New Roman serif + stix math;
INK=#0b0b0b (black = rig CVR), MUTED=#555 (grey = rig CF),
ACC=#2a78d6 (blue = LSCI optical chain). Data: perfusion-log CSVs plus the
session tables of the paper. No existing figure file is modified.
"""
import csv
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

INK, MUTED, ACC = "#0b0b0b", "#555555", "#2a78d6"

DATA = r"D:\Project\CC_Project\激光散斑心脏1和心脏2新增数据\CVR=AOPCF"
OUT = r"E:\2026年论文\激光散斑的ieee access\CoroLSCI_EN_clean\figures\fig11_cvr.pdf"

# ---- 会话数据 (表III P_mfr 与 表V CVR) ----
H1_MIN = np.array([0, 3, 11, 15, 44, 79, 113])          # +min (T0..T6)
H1_PM = np.array([1027.8, 986.5, 1033.9, 1010.6, 967.7, 975.0, 806.8])
H1_CVR = np.array([0.0871, 0.0849, 0.0811, 0.0789, 0.0779, 0.0762, 0.0855])
H1_AOPM, H1_AOPS = np.array([53.60, 53.65, 53.57, 53.50, 53.25, 51.60, 49.97]), np.array([.74, .73, .75, .72, .72, .58, .43])
H1_CFM, H1_CFS = np.array([615.1, 631.9, 660.9, 678.3, 684.0, 677.1, 584.5]), np.array([33.4, 26.4, 25.3, 22.1, 24.7, 22.8, 34.4])
H1_CVR_SD = H1_CVR * np.sqrt((H1_AOPS / H1_AOPM) ** 2 + (H1_CFS / H1_CFM) ** 2)  # 误差传播
H1_CLOSEUP = np.array([True, True, False, False, False, False, False])  # T0/T1 特写会话

H2_MIN = np.array([0, 6, 15])
H2_PM = np.array([1053.0, 1029.7, 997.9])
H2_CVR = np.array([0.1041, 0.1023, 0.1006])

T0_H1 = datetime(2026, 7, 9, 20, 1)   # 心1 T0 = +0 min
T5_MIN, T6_MIN = 79, 113


def per_minute_flow(path, t_lo, t_hi):
    """按分钟聚合 CF。"""
    mins, vals = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(rd)]
        it, iv = header.index("LocalTS"), header.index("CF")
        for row in rd:
            if not row or not row[it].strip():
                continue
            try:
                t = datetime.strptime(row[it].strip(), "%Y-%m-%d %H:%M:%S:%f")
                x = float(row[iv])
            except (ValueError, IndexError):
                continue
            if t_lo <= t <= t_hi:
                k = t.replace(second=0, microsecond=0)
                mins.setdefault(k, []).append(x)
    ks = sorted(mins)
    xm = np.array([(k - T0_H1).total_seconds() / 60.0 for k in ks])
    ym = np.array([np.mean(mins[k]) for k in ks])
    return xm, ym


print("loading heart-1 flow log ...")
xm, ym = per_minute_flow(DATA + r"\BloodFlowEntity-zhuxin1.csv",
                         T0_H1 - timedelta(minutes=6), T0_H1 + timedelta(minutes=130))
print(f"per-minute CF: n={len(xm)} min, range {xm.min():.0f}..{xm.max():.0f}")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

# ---------- (A) 心1 会话级双轴 ----------
ax = axes[0]
rel = 100.0 * H1_PM / H1_PM[2]  # T2=100
m_open = H1_CLOSEUP
m_fill = ~H1_CLOSEUP
ax.axvspan(T5_MIN, T6_MIN, color=ACC, alpha=0.10, lw=0, zorder=0)
ln1 = ax.plot(H1_MIN[m_fill], rel[m_fill], "o-", color=ACC, lw=1.6, ms=5.5, zorder=3,
              label="$P_{\\mathrm{mfr}}$ (LSCI, rel.\\ T2)")
ax.plot(H1_MIN[m_open], rel[m_open], "o", mfc="none", mec=ACC, ms=5.5, mew=1.4, zorder=3)
ax2 = ax.twinx()
ln2 = ax2.plot(H1_MIN[m_fill], 100 * H1_CVR[m_fill], "s--", color=INK, lw=1.6, ms=5.5, zorder=3,
               label="CVR (rig logs)")
ax2.plot(H1_MIN[m_open], 100 * H1_CVR[m_open], "s", mfc="none", mec=INK, ms=5.5, mew=1.4, zorder=3)
ax2.errorbar(H1_MIN, 100 * H1_CVR, yerr=100 * H1_CVR_SD, fmt="none", ecolor=INK,
             elinewidth=0.9, capsize=2.5, alpha=0.55, zorder=2)
ax.text((T5_MIN + T6_MIN) / 2, ax.get_ylim()[0] + 0.94 * (np.diff(ax.get_ylim())[0]),
        "T5–T6", ha="center", va="top", fontsize=10, color=MUTED)
ax.set_xlabel("Time from first session (min)", fontsize=11)
ax.set_ylabel("$P_{\\mathrm{mfr}}$ relative to T2 (%)", fontsize=11, color=ACC)
ax2.set_ylabel("CVR ($10^{-2}$ mmHg$\\cdot$min/mL)", fontsize=11, color=INK)
ax.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax.set_title("(A) Heart 1: session series", fontsize=12, color=INK)
ax.grid(alpha=0.25, lw=0.5)
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], loc="lower left", fontsize=9, framealpha=0.9)

# ---------- (B) 心1 逐分钟流量轨迹 ----------
ax = axes[1]
ax.axvspan(T5_MIN, T6_MIN, color=ACC, alpha=0.10, lw=0, zorder=0)
ax.plot(xm, ym, "-", color=MUTED, lw=1.5, zorder=2, label="coronary flow (per-minute)")
for mm, lab in [(0, "T0"), (3, "T1"), (11, "T2"), (15, "T3"), (44, "T4"), (79, "T5"), (113, "T6")]:
    ax.axvline(mm, color=INK, lw=0.5, alpha=0.30, zorder=1)
    ax.text(mm, 498, lab, fontsize=8.5, ha="center", va="center", color=INK)
ax2 = ax.twinx()
ax2.plot(H1_MIN[m_fill], 100 * H1_CVR[m_fill], "s--", color=INK, lw=1.6, ms=5.5,
         label="CVR (sessions)")
ax2.plot(H1_MIN[m_open], 100 * H1_CVR[m_open], "s", mfc="none", mec=INK, ms=5.5, mew=1.4)
ax.set_xlabel("Time from first session (min)", fontsize=11)
ax.set_ylabel("Coronary flow (mL/min)", fontsize=11, color=MUTED)
ax2.set_ylabel("CVR ($10^{-2}$ mmHg$\\cdot$min/mL)", fontsize=11, color=INK)
ax.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax.set_ylim(480, 740)
ax.set_title("(B) Heart 1: rig flow trajectory", fontsize=12, color=INK)
ax.grid(alpha=0.25, lw=0.5)

# ---------- (C) 心2 短时程复制 ----------
ax = axes[2]
rel2 = 100.0 * H2_PM / H2_PM[0]
ax.plot(H2_MIN, rel2, "o-", color=ACC, lw=1.6, ms=5.5, label="$P_{\\mathrm{mfr}}$ (rel.\\ T0)")
ax2 = ax.twinx()
ax2.plot(H2_MIN, 100 * H2_CVR, "s--", color=INK, lw=1.6, ms=5.5, label="CVR (rig logs)")
ax.set_xlabel("Time from first session (min)", fontsize=11)
ax.set_ylabel("$P_{\\mathrm{mfr}}$ relative to T0 (%)", fontsize=11, color=ACC)
ax2.set_ylabel("CVR ($10^{-2}$ mmHg$\\cdot$min/mL)", fontsize=11, color=INK)
ax.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax.set_title("(C) Heart 2: 15-min replication", fontsize=12, color=INK)
ax.grid(alpha=0.25, lw=0.5)

fig.tight_layout(pad=1.2)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("wrote", OUT)
