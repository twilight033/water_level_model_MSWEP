"""先运行B按Q/H选模，再运行B/C按Q选模；默认仅检查配置。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross_section_experiment.inference import infer
from cross_section_experiment.protocol import fingerprint, q_masks, validate_population, write_json
from cross_section_experiment.runner import make_train_config, save_prediction
from evaluation.metrics import basin_metrics
from pipeline.dataset import PreparedData, TargetScaling, WindowDataset
from training.trainer import train_model

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "original86_qh_train_valid_hidden_B_v1"
SELECTION = "0.5 * (mean_retained_validation_Q_NSE + mean_retained_validation_H_NSE)"
Q_SELECTION = "mean_retained_validation_Q_NSE"
Q_PROTOCOL = "original86_q_selected_BC_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_reference(source, scaling):
    """逐组核对C的完成记录；不调用随机抽样，也不按种子重新生成名单。"""
    source = Path(source)
    experiment = read_json(source / "experiment.json")
    settings, basins = experiment["settings"], experiment["basins"]
    if settings["target_scaling"] != scaling:
        raise ValueError("C来源的归一化条件与请求条件不一致")
    if not (source / "completed_both.json").is_file() or settings.get("smoke", False):
        raise ValueError("C来源须是已完成的正式实验")
    if experiment.get("protocol") != "original86_cross_section_train_valid_q_hidden_v1":
        raise ValueError("C来源不是原86训练与验证Q均屏蔽的协议")
    seen = set()
    references = []
    for scenario in experiment["scenarios"]:
        held, kept = scenario["heldout"], scenario["retained"]
        if (len(set(held)) != len(held) or len(set(kept)) != len(kept)
                or set(held) & set(kept) or set(held) | set(kept) != set(basins)):
            raise ValueError("C来源的留出/保留名单不是总体的互斥完整划分")
        if scenario["tag"] in seen:
            raise ValueError("C来源含重复情景")
        seen.add(scenario["tag"])
        for seed in settings["model_seeds"]:
            folder = source / "stage2" / scenario["tag"] / f"seed{seed}" / "dual_head"
            for name in ("done.json", "training_done.json", "training_history.json",
                         "model.pt", "prediction.npz", "metrics.csv"):
                if not (folder / name).is_file():
                    raise FileNotFoundError(folder / name)
            done, trained = read_json(folder / "done.json"), read_json(folder / "training_done.json")
            if (done["heldout"] != held or done["model_seed"] != seed
                    or done["variant"] != "dual_head"
                    or trained["model_config"]["architecture"] != "dual_head"
                    or done["run_fingerprint"] != trained["run_fingerprint"]):
                raise ValueError(f"C运行记录与情景不一致：{folder}")
            references.append({"scenario": copy.deepcopy(scenario), "model_seed": seed,
                               "c_folder": str(folder.resolve()),
                               "c_run_fingerprint": done["run_fingerprint"]})
    return experiment, references


def qh_masks(prepared, heldout):
    """两个任务共用同一目标站名单，训练/验证分别屏蔽，测试掩膜全为False。"""
    train, valid = q_masks(prepared, heldout)
    return ({"flow": train["flow"], "waterlevel": train["flow"].copy()},
            {"flow": valid["flow"], "waterlevel": valid["flow"].copy()})


def hidden_label_view(prepared, heldout, scaling, hide_waterlevel=True):
    """隔离隐藏标签和H尺度；observed仅明确保留原方案允许的Q尺度。"""
    if scaling not in ("physical", "observed"):
        raise ValueError("只支持physical或observed")
    view = copy.copy(prepared)
    view.stats = dict(prepared.stats)
    view.stats["flow"] = dict(prepared.stats["flow"])
    view.stats["waterlevel"] = dict(prepared.stats["waterlevel"])
    view.targets_raw = {task: values.copy() for task, values in prepared.targets_raw.items()}
    for basin in heldout:
        bi = prepared.basin_index[basin]
        # 留出站H完全不用，包括不报告其测试H技巧；0/1是占位尺度，不是估计值。
        if hide_waterlevel:
            view.targets_raw["waterlevel"][bi] = np.nan
            view.stats["waterlevel"][basin] = {"mean": 0.0, "std": 1.0, "n": 0}
        for split in ("train", "valid"):
            lo, hi = prepared.split_range(basin, split)
            view.targets_raw["flow"][bi, lo:hi + 1] = np.nan
        if scaling == "physical":
            view.stats["flow"][basin] = {"mean": 0.0, "std": 1.0, "n": 0}
    view.default_scaling = TargetScaling(view, "observed")
    view.target_mean, view.target_std = view.default_scaling.mean, view.default_scaling.std
    view.targets = view.default_scaling.normalized
    return view


def audit_datasets(view, scenario, settings, scaling, group="B", selection=SELECTION):
    """预检检查实际样本索引，保证目标站不进入任一训练损失或验证分数。"""
    hidden, hidden_valid = qh_masks(view, scenario["heldout"])
    if group == "C":
        hidden = {"flow": hidden["flow"]}
    scale = view.make_scaling(scaling, fit_basins=scenario["retained"])
    held_indices = [view.basin_index[b] for b in scenario["heldout"]]
    result = {}
    for split, masks in (("train", hidden), ("valid", hidden_valid)):
        dataset = WindowDataset(view, split, settings["sequence_length"],
                                settings[f"{split}_stride"], hidden=masks, scaling=scale)
        if (group == "B" or split == "valid") and np.isin(dataset.basin_idx, held_indices).any():
            raise AssertionError("B组目标站进入了训练或验证数据集")
        result[split] = {"n_samples": len(dataset), "label_counts": dataset.label_counts(),
                         "basins": [b for b, n in dataset.per_basin_counts.items() if n]}
    result.update(heldout_h_statistics_used=group == "C", heldout_q_labels_used=False,
                  heldout_q_statistics_used=scaling == "observed",
                  q_scale_fit_basins=scenario["retained"] if scaling == "physical" else [],
                  validation_selection=selection)
    return result


def train_b(view, settings, reference, scaling, device):
    """保持C的网络、种子、30轮与Q/H等权选模，仅改变目标站标签可用性。"""
    scenario = reference["scenario"]
    hidden, hidden_valid = qh_masks(view, scenario["heldout"])
    cfg = make_train_config(settings, "dual_head", reference["model_seed"], device)
    cfg.target_scaling = scaling
    return train_model(view, cfg, hidden=hidden, hidden_valid=hidden_valid,
                       fit_basins=scenario["retained"], track_selection_tasks=("flow",))


def train_c_q(view, settings, reference, scaling, device):
    """C保留目标站训练H；验证期Q/H均只来自保留站，且仅Q决定检查点。"""
    scenario = reference["scenario"]
    hidden, _ = q_masks(view, scenario["heldout"])
    _, hidden_valid = qh_masks(view, scenario["heldout"])
    cfg = make_train_config(settings, "dual_head", reference["model_seed"], device)
    cfg.target_scaling = scaling
    return train_model(view, cfg, hidden=hidden, hidden_valid=hidden_valid,
                       fit_basins=scenario["retained"], selection_tasks=("flow",))


def evaluate_q(prepared, prediction, q, scenario, seed, scaling, group="B"):
    """只在模型固定后的最终评价读取测试Q；推理时刻不依赖Q/H有效性。"""
    rows = []
    for bi, basin in enumerate(prepared.basins):
        selected = prediction["basin_idx"] == bi
        pos, pred = prediction["target_pos"][selected], q[selected]
        obs = prepared.targets_raw["flow"][bi, pos]
        ok = np.isfinite(obs) & np.isfinite(pred)
        metrics = {"n": int(ok.sum()), "nse": float("nan")}
        if ok.sum() >= 2:
            full = np.arange(pos.min(), pos.max() + 1)
            series_obs, series_pred = np.full(len(full), np.nan), np.full(len(full), np.nan)
            series_obs[pos - full[0]], series_pred[pos - full[0]] = obs, pred
            metrics = basin_metrics(obs[ok], pred[ok], series_obs, series_pred)
        rows.append({"basin": basin, "task": "flow", "group": group, "architecture": "dual_head",
                     "ratio": scenario["ratio"], "mask_seed": scenario["mask_seed"],
                     "model_seed": seed, "held_out": basin in scenario["heldout"],
                     "target_scaling": scaling, **metrics})
    return pd.DataFrame(rows)


def make_checkpoint(prepared, result, run_id, tracked=None):
    selected = result if tracked is None else tracked
    return {"state_dict": selected["state_dict"], "model_config": result["model_config"],
            "training_config": result["config"], "run_fingerprint": run_id,
            "selection_tasks": selected.get("selection_tasks", ["flow", "waterlevel"]),
            "best_epoch": selected["best_epoch"], "best_val_score": selected["best_val_score"],
            "target_mean": result["scaling"].mean, "target_std": result["scaling"].std,
            "attr_columns": list(prepared.attr_columns)}


def save_evaluation(prepared, view, checkpoint, settings, reference, scaling, device, folder, group):
    inference = infer(view, checkpoint["state_dict"], checkpoint["model_config"], "test",
                      settings["sequence_length"], settings["test_stride"], device)
    # 与C逐时刻配对；不能因为B没有目标站H就悄悄缩小测试时间范围。
    with np.load(Path(reference["c_folder"]) / "prediction.npz") as c_prediction:
        for key in ("basin_idx", "target_pos"):
            if not np.array_equal(inference[key], c_prediction[key]):
                raise ValueError(f"新运行与C测试时刻不一致：{key}")
    bi = inference["basin_idx"]
    q = (inference["pred"]["flow"] * checkpoint["target_std"]["flow"][bi]
         + checkpoint["target_mean"]["flow"][bi])
    evaluate_q(prepared, inference, q, reference["scenario"], reference["model_seed"], scaling,
               group=group).to_csv(folder / "metrics.csv", index=False, encoding="utf-8-sig")
    save_prediction(folder, bi, inference["target_pos"], q)


def finished(folder, run_id, additional=()):
    done = folder / "done.json"
    if not done.exists():
        return False
    if read_json(done)["run_fingerprint"] != run_id:
        raise ValueError(f"已有结果指纹不同，拒绝覆盖：{folder}")
    for name in ("model.pt", "training_history.json", "metrics.csv", "prediction.npz", *additional):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    return True


def execute(prepared, settings, reference, scaling, device, root, experiment_id):
    scenario, seed = reference["scenario"], reference["model_seed"]
    folder = root / scaling / scenario["tag"] / f"seed{seed}"
    run_id = fingerprint({"experiment": experiment_id, "reference": reference})
    done = folder / "done.json"
    if finished(folder, run_id, additional=("q_selected.pt",)):
        print(f"跳过已完成：{folder}", flush=True)
        return
    view = hidden_label_view(prepared, scenario["heldout"], scaling)
    audit = audit_datasets(view, scenario, settings, scaling)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "preflight.json", audit)
    print(f"开始B：{scaling}/{scenario['tag']}/seed{seed}", flush=True)
    result = train_b(view, settings, reference, scaling, device)
    tracked = result["tracked_checkpoint"]
    if tracked is None or tracked["best_epoch"] < 1:
        raise ValueError("B未保存按验证Q选模的权重，拒绝标记完成")
    checkpoint = make_checkpoint(prepared, result, run_id)
    torch.save(checkpoint, folder / "model.pt")
    torch.save(make_checkpoint(prepared, result, run_id, tracked), folder / "q_selected.pt")
    write_json(folder / "training_history.json", result["history"])
    save_evaluation(prepared, view, checkpoint, settings, reference, scaling, device, folder, "B")
    keys = ("best_epoch", "best_val_score", "epochs_run", "elapsed_sec", "n_train_samples",
            "n_valid_samples", "train_label_counts", "valid_label_counts", "scaling_info")
    write_json(done, {"protocol": PROTOCOL, "run_fingerprint": run_id, **reference,
                      "target_scaling": scaling, "validation_selection": SELECTION,
                      "q_selection": {key: tracked[key] for key in
                                      ("selection_tasks", "best_epoch", "best_val_score")},
                      **{key: result[key] for key in keys}})


def require_b_completed(references, root, experiment_id):
    """第二阶段开始前，所选范围的全部B主实验必须已完成并包含Q最优权重。"""
    for scaling, _, _, runs in references:
        for reference in runs:
            scenario, seed = reference["scenario"], reference["model_seed"]
            folder = root / scaling / scenario["tag"] / f"seed{seed}"
            run_id = fingerprint({"experiment": experiment_id, "reference": reference})
            if not finished(folder, run_id, additional=("q_selected.pt",)):
                raise FileNotFoundError(f"B主实验尚未完成，不能进入按Q选模阶段：{folder}")
            if read_json(folder / "done.json")["q_selection"]["selection_tasks"] != ["flow"]:
                raise ValueError("B旁路权重的选模任务不是Q")


def execute_q_selected(prepared, settings, reference, scaling, device, root,
                       experiment_id, b_root, b_experiment_id, group):
    """B重用同次训练的Q最优权重；C独立重训，两个损失仍同时优化Q/H。"""
    if group not in ("B", "C"):
        raise ValueError("按Q选模只支持B或C组")
    scenario, seed = reference["scenario"], reference["model_seed"]
    folder = root / group / scaling / scenario["tag"] / f"seed{seed}"
    run_id = fingerprint({"experiment": experiment_id, "reference": reference, "group": group})
    if finished(folder, run_id):
        print(f"跳过已完成：{folder}", flush=True)
        return
    view = hidden_label_view(prepared, scenario["heldout"], scaling, hide_waterlevel=group == "B")
    audit = audit_datasets(view, scenario, settings, scaling, group=group, selection=Q_SELECTION)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "preflight.json", audit)
    print(f"按Q选模：{group}/{scaling}/{scenario['tag']}/seed{seed}", flush=True)
    if group == "B":
        source = b_root / scaling / scenario["tag"] / f"seed{seed}"
        expected = fingerprint({"experiment": b_experiment_id, "reference": reference})
        if not finished(source, expected, additional=("q_selected.pt",)):
            raise FileNotFoundError("缺少已完成B的按Q选模权重")
        checkpoint = torch.load(source / "q_selected.pt", map_location="cpu", weights_only=False)
        if checkpoint["run_fingerprint"] != expected or checkpoint["selection_tasks"] != ["flow"]:
            raise ValueError("B旁路权重来源或选模任务不匹配")
        checkpoint["run_fingerprint"] = run_id
        history = read_json(source / "training_history.json")
        training_metadata = {"training_reused": True, "source_training_folder": str(source),
                             "epochs_run": len(history), "new_training_runs": 0}
    else:
        result = train_c_q(view, settings, reference, scaling, device)
        checkpoint = make_checkpoint(prepared, result, run_id)
        history = result["history"]
        training_metadata = {"training_reused": False, "epochs_run": result["epochs_run"],
                             "new_training_runs": 1, "n_train_samples": result["n_train_samples"],
                             "n_valid_samples": result["n_valid_samples"],
                             "train_label_counts": result["train_label_counts"],
                             "valid_label_counts": result["valid_label_counts"]}
    torch.save(checkpoint, folder / "model.pt")
    write_json(folder / "training_history.json", history)
    save_evaluation(prepared, view, checkpoint, settings, reference, scaling, device, folder, group)
    write_json(folder / "done.json", {"protocol": Q_PROTOCOL, "run_fingerprint": run_id,
                                      "group": group, **reference, "target_scaling": scaling,
                                      "validation_selection": Q_SELECTION,
                                      "best_epoch": checkpoint["best_epoch"],
                                      "best_val_score": checkpoint["best_val_score"],
                                      **training_metadata})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qh_holdout_b.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scaling", nargs="+", choices=("physical", "observed"))
    parser.add_argument("--stage", choices=("all", "b-qh", "q-bc"), default="all",
                        help="先B按Q/H选模，再B/C按Q选模；all按顺序执行")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="加载真实数据检查边界，不训练")
    mode.add_argument("--run", action="store_true", help="明确启动所选阶段；省略时只核对C运行清单")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if config["protocol"] != PROTOCOL:
        raise ValueError("B协议标记不一致")
    sources = [(scale, project_path(path)) for scale, path in config["sources"].items()]
    all_references = [(scale, source, *load_reference(source, scale)) for scale, source in sources]
    references = [item for item in all_references if args.scaling is None or item[0] in args.scaling]
    if not references:
        raise ValueError("配置中没有所请求尺度的C来源")
    count = sum(len(runs) for _, _, _, runs in references)
    if args.stage in ("all", "b-qh"):
        print(f"第一阶段：B按Q/H选模，{count}次训练，同时保存按Q选模的权重。", flush=True)
    if args.stage in ("all", "q-bc"):
        print(f"第二阶段：B按Q选模复用{count}次，C按Q选模新训{count}次。", flush=True)
    print("已核对C来源，名单、掩膜种子与模型种子直接复用。", flush=True)
    if not args.run and not args.preflight:
        print("未启动训练。--preflight仅检查数据；--run才启动训练。", flush=True)
        return 0
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    code_files = [Path(__file__), ROOT / "src/training/trainer.py", ROOT / "src/pipeline/dataset.py",
                  ROOT / "src/cross_section_experiment/runner.py", ROOT / "src/models/lstm_models.py"]
    manifest = {"protocol": PROTOCOL, "config": config, "validation_selection": SELECTION,
                "tracked_validation_selection": Q_SELECTION,
                "training_budget": "复用C的固定轮数和学习率；有效窗口与更新次数会因标签减少而变化",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "device": device},
                "sources": [{"scaling": s, "source": str(p), "experiment_fingerprint": e["experiment_fingerprint"],
                             "settings": e["settings"], "runs": r} for s, p, e, r in all_references],
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in code_files}}
    experiment_id = fingerprint(manifest)
    root = project_path(config["output_root"])
    q_root = project_path(config.get("q_output_root", str(root) + "_q_selected_bc"))
    if q_root.resolve() == root.resolve():
        raise ValueError("两种选模标准必须使用独立结果目录")
    q_manifest = {"protocol": Q_PROTOCOL, "config": config, "validation_selection": Q_SELECTION,
                  "b_experiment_fingerprint": experiment_id, "b_q_checkpoint_reused": True,
                  "c_training_h_retained": True, "c_heldout_validation_h_hidden": True,
                  "sources": manifest["sources"], "environment": manifest["environment"],
                  "code_sha256": manifest["code_sha256"]}
    q_experiment_id = fingerprint(q_manifest)
    # 默认模式与预检不创建结果目录，避免把“已准备”误当作“已运行”。
    stages = ("b-qh", "q-bc") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if args.run:
            if stage == "b-qh":
                save_manifest(root, manifest, experiment_id)
            else:
                require_b_completed(references, root, experiment_id)
                save_manifest(q_root, q_manifest, q_experiment_id)
        for scale, _, experiment, runs in references:
            settings = experiment["settings"]
            prepared = PreparedData(attr_set=settings["attribute_set"])
            validate_population(prepared, experiment["basins"])
            if args.preflight:
                for scenario in experiment["scenarios"]:
                    for group in (("B",) if stage == "b-qh" else ("B", "C")):
                        view = hidden_label_view(prepared, scenario["heldout"], scale,
                                                 hide_waterlevel=group == "B")
                        audit_datasets(view, scenario, settings, scale, group=group,
                                       selection=SELECTION if stage == "b-qh" else Q_SELECTION)
                print(f"{stage}/{scale}：全部数据边界检查通过，未训练。", flush=True)
            elif stage == "b-qh":
                for reference in runs:
                    execute(prepared, settings, reference, scale, device, root, experiment_id)
            else:
                for group in ("B", "C"):
                    for reference in runs:
                        execute_q_selected(prepared, settings, reference, scale, device, q_root,
                                           q_experiment_id, root, experiment_id, group)
            del prepared
        if args.run:
            if stage == "b-qh":
                mark_completed(root, all_references, experiment_id)
            else:
                mark_completed(q_root, all_references, q_experiment_id, groups=("B", "C"))
    return 0


def save_manifest(root, manifest, experiment_id):
    if (root / "experiment.json").exists():
        if read_json(root / "experiment.json")["experiment_fingerprint"] != experiment_id:
            raise ValueError("输出目录已有不同来源或代码指纹，拒绝覆盖")
    write_json(root / "experiment.json", {**manifest, "experiment_fingerprint": experiment_id})


def mark_completed(root, all_references, experiment_id, groups=None):
    """只在完整矩阵齐全时写全局完成标记；单尺度运行不冒充全部完成。"""
    count = 0
    for scale, _, _, runs in all_references:
        for reference in runs:
            for group in ((None,) if groups is None else groups):
                folder = root if group is None else root / group
                folder = folder / scale / reference["scenario"]["tag"] / f"seed{reference['model_seed']}"
                payload = {"experiment": experiment_id, "reference": reference}
                if group is not None:
                    payload["group"] = group
                if not finished(folder, fingerprint(payload),
                                additional=("q_selected.pt",) if group is None else ()):
                    return
                count += 1
    write_json(root / "completed.json", {"experiment_fingerprint": experiment_id,
                                         "n_completed": count, "selection_used_test_scores": False})
