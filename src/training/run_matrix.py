"""实验矩阵编排：批量运行、断点续跑、结果汇总。

每次运行写入独立目录 results/runs/<run_key>/，包含配置、逐流域指标、
训练历史与模型权重；代表性运行额外导出逐流域真实时序。所有运行共用
data/splits/ 下冻结的划分、归一化统计量与掩膜缓存。
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.metrics import aggregate, evaluate_run  # noqa: E402
from pipeline.coverage import load_eligibility  # noqa: E402
from pipeline.dataset import PreparedData  # noqa: E402
from pipeline.masking import get_hidden  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402
from training.trainer import TrainConfig, train_model  # noqa: E402

RUNS_DIR = RESULTS_ROOT / "runs"
SUMMARY_DIR = RESULTS_ROOT / "summary"

MULTITASK = ("dual_head", "capacity_matched", "wl2d")

# 缺失情景：名称 -> (目标比例, 机制, 是否季节匹配, 参与比较的模型)
SCENARIOS = {
    "complete":         ({},                              None,      True,  None),
    "q30_seg":          ({"flow": 0.30},                  "segment", True,  ("single_flow", "dual_head")),
    "q50_seg":          ({"flow": 0.50},                  "segment", True,  ("single_flow", "dual_head")),
    "q70_seg":          ({"flow": 0.70},                  "segment", True,  ("single_flow", "dual_head")),
    "h30_seg":          ({"waterlevel": 0.30},            "segment", True,  ("single_waterlevel", "dual_head")),
    "h50_seg":          ({"waterlevel": 0.50},            "segment", True,  ("single_waterlevel", "dual_head")),
    "h70_seg":          ({"waterlevel": 0.70},            "segment", True,  ("single_waterlevel", "dual_head")),
    "both50_seg":       ({"flow": 0.50, "waterlevel": 0.50}, "segment", True,
                         ("single_flow", "single_waterlevel", "dual_head")),
    "q50_mcar":         ({"flow": 0.50},                  "mcar",    True,  ("single_flow", "dual_head")),
    "q50_seg_uniform":  ({"flow": 0.50},                  "segment", False, ("single_flow", "dual_head")),
    # 按流域留出：论文主命题——某些流域根本没有径流观测，只有廉价的水位
    # 观测，问水位监督能否顶上。ratio 在此表示被留出的流域比例。
    "q_holdout30":      ({"flow": 0.30},          "basin_holdout", True,  ("single_flow", "dual_head")),
    "q_holdout50":      ({"flow": 0.50},          "basin_holdout", True,  ("single_flow", "dual_head")),
    "q_holdout70":      ({"flow": 0.70},          "basin_holdout", True,  ("single_flow", "dual_head")),
}


def build_matrix(scale: str = "full") -> list:
    """构造实验条目列表。scale='smoke' 用于小规模验证。"""
    entries = []

    def add(group, architecture, scenario, model_seed, mask_seed=0, **extra):
        entries.append({"group": group, "architecture": architecture,
                        "scenario": scenario, "model_seed": model_seed,
                        "mask_seed": mask_seed, **extra})

    if scale == "smoke":
        for arch in ("single_flow", "dual_head"):
            add("4A_main", arch, "complete", 1)
        add("4C_missing", "single_flow", "q50_seg", 1, mask_seed=42)
        add("4C_missing", "dual_head", "q50_seg", 1, mask_seed=42)
        return entries

    main_seeds = list(range(1, 11))
    miss_model_seeds = [1, 2, 3]
    miss_mask_seeds = [42, 123, 456]

    # 4-A 主实验：三个模型 × 10 个模型种子
    for arch in ("single_flow", "single_waterlevel", "dual_head"):
        for seed in main_seeds:
            add("4A_main", arch, "complete", seed, export_series=(seed <= 2))

    # 4-B 附录架构对照：与双头同种子配对
    for arch in ("wl2d", "capacity_matched"):
        for seed in main_seeds:
            add("4B_architecture", arch, "complete", seed)

    # 4-C 缺失标签：每个情景同步训练对应单任务与双头
    for scenario, (_, _, _, models) in SCENARIOS.items():
        if scenario == "complete":
            continue
        # 按流域留出是论文主命题，用满 3 个掩膜种子（决定哪些流域被留出）；
        # 机制对照与季节性对照用较少的掩膜种子
        mask_seeds = miss_mask_seeds if scenario in (
            "q30_seg", "q50_seg", "q70_seg", "h30_seg", "h50_seg", "h70_seg",
            "both50_seg", "q_holdout30", "q_holdout50", "q_holdout70") else miss_mask_seeds[:2]
        for arch in models:
            for ms in miss_model_seeds:
                for xs in mask_seeds:
                    add("4C_missing", arch, scenario, ms, mask_seed=xs)

    # 4-D 窗口敏感性：7 天与 60 天（21 天复用 4-A）
    for seq_length in (56, 480):
        for seed in main_seeds[:5]:
            add("4D_window", "dual_head", "complete", seed, seq_length=seq_length)

    # 4-E 损失归一化对照
    for arch in ("single_flow", "dual_head"):
        for ms in miss_model_seeds:
            for xs in miss_mask_seeds[:2]:
                add("4E_lossnorm", arch, "q50_seg", ms, mask_seed=xs, loss_norm="per_batch")

    # 4-F 筛选门槛敏感性：仅用测试段满一年且训练标签更充足的流域
    for arch in ("single_flow", "dual_head"):
        for seed in miss_model_seeds:
            add("4F_threshold", arch, "complete", seed, strict_basins=True)

    # 4-G 任务权重敏感性。默认 1:1 会让共享编码器被更易学的水位任务拽偏，
    # 不排除这一项就无法把"多任务无收益"与"任务权重没调"区分开。
    # w_h = 1.0 的情形已由 4-A 的 dual_head 覆盖，此处只跑更小的权重。
    for w_h in (0.1, 0.25, 0.5):
        for seed in miss_model_seeds:
            add("4G_taskweight", "dual_head", "complete", seed, waterlevel_weight=w_h)

    # 4-H 物理归一化：把"已知流量量级"这个前提也拿掉。
    # 默认的逐流域归一化用实测 Q 的均值方差，因此留出流域仍间接知道自己的流量
    # 量级与变幅——真正无资料流域拿不到这两个数。physical 模式下径流尺度改由
    # 「面积×降水」的 log–log 回归推出，回归只在仍有径流标签的流域上拟合。
    # 这是把主张从"已知量级"升级到"完全无资料"的关键对照。
    for arch in ("single_flow", "single_waterlevel", "dual_head"):
        for seed in main_seeds[:5]:
            add("4H_physical", arch, "complete", seed, target_scaling="physical")
    for scenario in ("q_holdout30", "q_holdout50", "q_holdout70"):
        for arch in ("single_flow", "dual_head"):
            for ms in miss_model_seeds:
                for xs in miss_mask_seeds:
                    add("4H_physical", arch, scenario, ms, mask_seed=xs,
                        target_scaling="physical")

    # 4-I 属性隔离：只换属性集，窗口与归一化不变，用于测出属性扩充的净效应。
    # 闸门——若径流 NSE 提升 <0.005 则不进行 4-J。
    for arch in ("single_flow", "single_waterlevel", "dual_head"):
        for seed in main_seeds[:5]:
            add("4I_attrs", arch, "complete", seed, attr_set="extended")

    # 4-J 最优配置：扩展属性 + 60 天窗口，重验主命题。
    for arch in ("single_flow", "single_waterlevel", "dual_head"):
        for seed in main_seeds:
            add("4J_best", arch, "complete", seed, attr_set="extended", seq_length=480)
    for scenario in ("q_holdout30", "q_holdout50", "q_holdout70"):
        for arch in ("single_flow", "dual_head"):
            for ms in miss_model_seeds:
                for xs in miss_mask_seeds:
                    add("4J_best", arch, scenario, ms, mask_seed=xs,
                        attr_set="extended", seq_length=480)

    return entries


def run_key(entry: dict) -> str:
    parts = [entry["group"], entry["scenario"], entry["architecture"],
             f"ms{entry['model_seed']}"]
    if entry.get("mask_seed"):
        parts.append(f"xs{entry['mask_seed']}")
    if entry.get("seq_length"):
        parts.append(f"L{entry['seq_length']}")
    if entry.get("loss_norm"):
        parts.append(entry["loss_norm"])
    if entry.get("strict_basins"):
        parts.append("strict")
    if entry.get("waterlevel_weight") is not None:
        parts.append(f"wh{entry['waterlevel_weight']:g}")
    if entry.get("target_scaling") and entry["target_scaling"] != "observed":
        parts.append(entry["target_scaling"])
    if entry.get("attr_set") and entry["attr_set"] != "base":
        parts.append(entry["attr_set"])
    return "_".join(parts)


def strict_basin_list(prepared) -> list:
    """更严格的流域筛选：排除测试段不足一年的流域。"""
    elig = load_eligibility()
    both = set(elig["flow"]) & set(elig["waterlevel"])
    keep = [b for b, v in prepared.splits["splits"].items()
            if b in both and not v.get("short_test", False)]
    return sorted(keep)


def run_one(prepared, entry: dict, force: bool = False, verbose: bool = False) -> dict:
    key = run_key(entry)
    out_dir = RUNS_DIR / key
    done_marker = out_dir / "run.json"
    if done_marker.exists() and not force:
        return {"run_key": key, "status": "skipped"}

    ratios, mechanism, seasonal, _ = SCENARIOS[entry["scenario"]]
    hidden = None
    mask_stats = None
    if ratios:
        hidden, mask_stats = get_hidden(prepared, ratios, mechanism=mechanism,
                                        mask_seed=entry["mask_seed"], seasonal=seasonal)

    basins = strict_basin_list(prepared) if entry.get("strict_basins") else None

    cfg_kwargs = {"architecture": entry["architecture"],
                  "model_seed": entry["model_seed"]}
    if entry.get("seq_length"):
        cfg_kwargs["seq_length"] = entry["seq_length"]
    if entry.get("loss_norm"):
        cfg_kwargs["loss_norm"] = entry["loss_norm"]
    if entry.get("waterlevel_weight") is not None:
        cfg_kwargs["task_weights"] = {"flow": 1.0,
                                      "waterlevel": entry["waterlevel_weight"]}
    # 注意不要用 key 作循环变量：函数开头的 key 是 run_key，会被覆盖
    for field in ("target_scaling", "attr_set"):
        if entry.get(field):
            cfg_kwargs[field] = entry[field]
    cfg = TrainConfig(**cfg_kwargs)

    t0 = time.time()
    result = train_model(prepared, cfg, hidden=hidden, basins=basins, verbose=verbose)
    metrics_df, series = evaluate_run(prepared, result["test_prediction"], result["tasks"],
                                      scaling=result.get("scaling"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.insert(0, "run_key", key)
    for col in ("group", "scenario", "architecture"):
        metrics_df.insert(1, col, entry[col])
    metrics_df["model_seed"] = entry["model_seed"]
    metrics_df["mask_seed"] = entry["mask_seed"]
    # 按流域留出时标出哪些流域的该任务训练标签被整段删除，
    # 后续分析要分开报告留出流域与保留流域的表现
    if mechanism == "basin_holdout" and mask_stats:
        held = {t: set(v.get("held_out_basins", []))
                for t, v in mask_stats["per_task"].items()}
        metrics_df["held_out"] = [
            row.basin in held.get(row.task, set())
            for row in metrics_df.itertuples()
        ]
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    aggregate(metrics_df).to_csv(out_dir / "aggregate.csv", index=False, encoding="utf-8-sig")

    if entry.get("export_series"):
        series_dir = out_dir / "timeseries"
        series_dir.mkdir(exist_ok=True)
        for basin, frame in series.items():
            frame.drop(columns=["target_pos"]).to_csv(
                series_dir / f"{basin}.csv", index=False, encoding="utf-8-sig")

    torch.save({"state_dict": result["state_dict"],
                "model_config": result["model_config"],
                "config": result["config"]}, out_dir / "model.pt")

    record = {
        "run_key": key, **{k: v for k, v in entry.items() if k != "export_series"},
        "config": result["config"], "model_config": result["model_config"],
        "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
        "hit_epoch_cap": result["hit_epoch_cap"],
        "best_val_score": result["best_val_score"],
        "n_train_samples": result["n_train_samples"],
        "n_valid_samples": result["n_valid_samples"],
        "n_test_samples": result["n_test_samples"],
        "train_label_counts": result["train_label_counts"],
        "valid_label_counts": result["valid_label_counts"],
        "test_label_counts": result["test_label_counts"],
        "n_updates_per_epoch": result["history"][0]["n_updates"] if result["history"] else 0,
        "total_updates": sum(h["n_updates"] for h in result["history"]),
        "mask_stats": (mask_stats["per_task"] if mask_stats else None),
        "scaling_info": result.get("scaling_info"),
        "elapsed_sec": time.time() - t0,
        "history": result["history"],
    }
    done_marker.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"run_key": key, "status": "done", "elapsed_sec": record["elapsed_sec"],
            "best_val_score": record["best_val_score"], "epochs": record["epochs_run"]}


def collect_summary() -> tuple:
    """汇总全部已完成运行为一张长表，供统计模块使用。"""
    metric_frames, records = [], []
    for run_dir in sorted(RUNS_DIR.glob("*")):
        mfile, rfile = run_dir / "metrics.csv", run_dir / "run.json"
        if not (mfile.exists() and rfile.exists()):
            continue
        metric_frames.append(pd.read_csv(mfile, dtype={"basin": str}))
        rec = json.loads(rfile.read_text(encoding="utf-8"))
        rec.pop("history", None)
        records.append(rec)
    if not metric_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="按实验矩阵批量重跑")
    parser.add_argument("--scale", choices=("full", "smoke"), default="full")
    parser.add_argument("--group", help="只跑某个分组，如 4A_main")
    parser.add_argument("--scenario", help="只跑某个缺失情景，如 q50_seg")
    parser.add_argument("--shard", metavar="i/n",
                        help="多机分片，如 1/2 表示两台机器中的第一台。"
                             "分片按条目序号取模，两台的划分与掩膜完全一致，"
                             "结果目录直接合并即可汇总")
    parser.add_argument("--limit", type=int, help="最多跑多少条")
    parser.add_argument("--force", action="store_true", help="重跑已完成的条目")
    parser.add_argument("--verbose", action="store_true", help="打印每轮训练进度")
    parser.add_argument("--summary-only", action="store_true", help="只汇总，不训练")
    args = parser.parse_args(argv)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        entries = build_matrix(args.scale)
        if args.group:
            entries = [e for e in entries if e["group"] == args.group]
        if args.scenario:
            entries = [e for e in entries if e["scenario"] == args.scenario]
        if args.shard:
            index, total = (int(v) for v in args.shard.split("/"))
            if not 1 <= index <= total:
                parser.error(f"--shard 取值应为 1/{total} 到 {total}/{total}")
            entries = [e for i, e in enumerate(entries) if i % total == index - 1]
            print(f"分片 {index}/{total}")
        if args.limit:
            entries = entries[:args.limit]

        print(f"实验矩阵共 {len(entries)} 条")
        # 不同 attr_set 需要各自的 PreparedData（属性矩阵维度不同），按需构造并缓存
        prepared_cache = {}

        def get_prepared(attr_set):
            if attr_set not in prepared_cache:
                prepared_cache[attr_set] = PreparedData(attr_set=attr_set)
            return prepared_cache[attr_set]

        t_start = time.time()
        done = 0
        for i, entry in enumerate(entries, 1):
            info = run_one(get_prepared(entry.get("attr_set", "base")), entry,
                           force=args.force, verbose=args.verbose)
            if info["status"] == "skipped":
                print(f"[{i}/{len(entries)}] 跳过（已完成） {info['run_key']}")
                continue
            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(entries) - i) / 60
            print(f"[{i}/{len(entries)}] {info['run_key']}  "
                  f"轮数 {info['epochs']}  验证分 {info['best_val_score']:.4f}  "
                  f"{info['elapsed_sec'] / 60:.1f} 分钟  预计剩余 {eta:.0f} 分钟")

    metrics, runs = collect_summary()
    if metrics.empty:
        print("暂无已完成的运行")
        return
    metrics.to_csv(SUMMARY_DIR / "all_metrics.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(SUMMARY_DIR / "all_runs.csv", index=False, encoding="utf-8-sig")
    print(f"\n汇总: {len(runs)} 次运行，{len(metrics)} 行逐流域指标")
    print(f"  {SUMMARY_DIR / 'all_metrics.csv'}")
    print(f"  {SUMMARY_DIR / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
