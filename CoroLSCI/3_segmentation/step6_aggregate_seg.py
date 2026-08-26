"""step6_aggregate_seg.py -- Aggregate segmentation runs

Aggregates the 3-seed x 5-config ablation results into the summary table
(Dice/clDice/HD95/area ratio) used in the paper.
"""
import csv, json, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "segmentation", "eval")
OUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果", "第三版执行", "segmentation", "summary")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [("S-I", "seg_I"), ("S-IV", "seg_IV"), ("S-IC", "seg_IC"),
           ("S-IVC", "seg_IVC"), ("Gated", "gated")]
SEEDS = [0, 1, 2]
KEYS = ["dice", "cldice", "hd95", "n_comp_pred", "area_pred", "area_target"]

rows = []
missing = []
for cfg, fn_prefix in CONFIGS:
    for sd in SEEDS:
        p = os.path.join(EVAL_DIR, f"{fn_prefix}_seed{sd}_test.json")
        if not os.path.exists(p):
            missing.append(f"{cfg}:{sd}")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        agg = d["aggregate"]
        rows.append({
            "config": cfg, "seed": sd,
            "dice": agg["dice"], "dice_std": agg["dice_std"],
            "cldice": agg["cldice"], "cldice_std": agg["cldice_std"],
            "hd95": agg["hd95"], "n_comp_pred": agg["n_comp_pred"],
            "area_pred": agg["area_pred"], "area_target": agg["area_target"],
            "area_ratio": agg["area_pred"] / max(agg["area_target"], 1),
            "best_epoch": agg["best_epoch"],
        })

if missing:
    print(f"缺少 {len(missing)} 个结果: {missing}")

# config summary
summary = []
for cfg, _ in CONFIGS:
    rs = [r for r in rows if r["config"] == cfg]
    if not rs:
        summary.append({"config": cfg, "n_seeds": 0})
        continue
    summary.append({
        "config": cfg, "n_seeds": len(rs),
        "dice_mean": np.mean([r["dice"] for r in rs]),
        "dice_sd": np.std([r["dice"] for r in rs]),
        "cldice_mean": np.mean([r["cldice"] for r in rs]),
        "cldice_sd": np.std([r["cldice"] for r in rs]),
        "hd95_mean": np.mean([r["hd95"] for r in rs]),
        "n_comp_mean": np.mean([r["n_comp_pred"] for r in rs]),
        "area_ratio_mean": np.mean([r["area_ratio"] for r in rs]),
        "best_epoch_mean": np.mean([r["best_epoch"] for r in rs]),
    })

# CSV
csv_path = os.path.join(OUT_DIR, "消融汇总表.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["config", "seed", "dice", "dice_std", "cldice",
                                      "cldice_std", "hd95", "n_comp_pred", "area_pred",
                                      "area_target", "area_ratio", "best_epoch"])
    w.writeheader()
    w.writerows(rows)
print(f"-> {csv_path}")

# Markdown report
md = ["# No. 6step : segmentation ablation summary (S-I / S-IV / S-IC / S-IVC × 3seed )\n",
      "测试集 = 30 帧 (时间组级独立, 已封存)\n",
      "| 配置 | 种子 | Dice | clDice | HD95 | 连通域数 | 面积比 | 最佳epoch |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['config']} | {r['seed']} | {r['dice']:.4f}±{r['dice_std']:.4f} | "
              f"{r['cldice']:.4f}±{r['cldice_std']:.4f} | {r['hd95']:.2f} | "
              f"{r['n_comp_pred']:.1f} | {r['area_ratio']:.3f} | {r['best_epoch']} |")
md.append("\n## config value ±seed between std\n")
md.append("| 配置 | Dice | clDice | HD95 | 连通域数 | 面积比 |")
md.append("|---|---|---|---|---|---|")
for s in summary:
    if s.get("n_seeds", 0) == 0:
        md.append(f"| {s['config']} | (缺) | | | | |")
        continue
    md.append(f"| {s['config']} | {s['dice_mean']:.4f}±{s['dice_sd']:.4f} | "
              f"{s['cldice_mean']:.4f}±{s['cldice_sd']:.4f} | {s['hd95_mean']:.2f} | "
              f"{s['n_comp_mean']:.1f} | {s['area_ratio_mean']:.3f} |")
md.append("\n注: 面积比 = 预测面积/标注面积, 1.0 为完全一致; 连通域数为预测mask的连通分量个数")
md.append(f"\n缺少数: {len(missing)} -> {missing}" if missing else "")

md_path = os.path.join(OUT_DIR, "消融汇总报告.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"-> {md_path}")

# control
print("\n===== 配置级汇总 =====")
for s in summary:
    if s.get("n_seeds", 0) == 0:
        print(f"{s['config']}: 缺")
        continue
    print(f"{s['config']}: Dice={s['dice_mean']:.4f}±{s['dice_sd']:.4f}  "
          f"clDice={s['cldice_mean']:.4f}±{s['cldice_sd']:.4f}  "
          f"HD95={s['hd95_mean']:.2f}  面积比={s['area_ratio_mean']:.3f}")
