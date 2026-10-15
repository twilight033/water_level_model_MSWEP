"""原86流域两个阶段的一键入口；所有新结果写入独立目录。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import pandas as pd
import torch

from cross_section_experiment.converters import fit_regional_converter, predict_converter
from cross_section_experiment.features import (
    FEATURE_COLUMNS, build_geometry, file_sha256, fit_geometry_scaler,
    shuffled_geometry, transform_geometry,
)
from cross_section_experiment.inference import infer
from cross_section_experiment.reference import audit_references, reference_folder, reuse_h_checkpoint
from cross_section_experiment.protocol import (
    PROTOCOL, WaterlevelOnlyPrepared, fingerprint, heldout_split, json_ready,
    q_masks, validate_population, with_attributes, write_json,
)
from evaluation.metrics import basin_metrics
from pipeline.dataset import PreparedData, WindowDataset
from pipeline.paths import load_basin_ids
from training.trainer import TrainConfig, train_model

DATA_FILES = (
    "86basin_ids.txt", "data/splits/basin_splits.json", "data/splits/norm_stats.json",
    "data/camelsh_exported/attributes_86.parquet",
    "data/camelsh_exported/attributes_86_extended.parquet",
    "data/camelsh_exported/flow_hourly_86.parquet",
    "data/camelsh_exported/waterlevel_hourly_86.parquet",
    "data/camelsh_exported/mswep_precip_3h_86.parquet",
    "data/camelsh_exported/forcing_temperature_mean_3h_86.parquet",
    "data/camelsh_exported/forcing_solar_radiation_3h_86.parquet",
)


def parser():
    p = argparse.ArgumentParser(description="原86流域：独立转换器＋断面输入单/双任务实验")
    p.add_argument("--config", type=Path, default=ROOT / "configs/cross_section86.json")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--profile-index", type=Path)
    p.add_argument("--geometry-mode", choices=("snapshot", "causal"))
    p.add_argument("--device", default="auto")
    p.add_argument("--stage", choices=("both", "stage1", "stage2"), default="both")
    p.add_argument("--preflight", action="store_true", help="完成原86、数据、断面与Q屏蔽检查，不训练")
    p.add_argument("--smoke", action="store_true", help="两阶段真实缓存烟雾测试，不作为论文结果")
    p.add_argument("--ratios", nargs="+", type=float)
    p.add_argument("--model-seeds", nargs="+", type=int)
    p.add_argument("--mask-seeds", nargs="+", type=int)
    p.add_argument("--no-sham-controls", action="store_true")
    return p


def settings_from_args(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in ("geometry_mode", "ratios", "model_seeds", "mask_seeds"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.no_sham_controls:
        cfg["sham_controls"] = False
    cfg["smoke"] = bool(args.smoke)
    if args.smoke:
        cfg.update(ratios=[0.5], model_seeds=[1], mask_seeds=[42],
                   lstm_epochs=1, hidden_size=4, sequence_length=8,
                   train_stride=4096, valid_stride=512, test_stride=512,
                   batch_size=64, converter_epochs=2, converter_widths=[4],
                   converter_max_pairs_per_basin=64)
        # 缩小后的烟雾模型不能冒充历史强模型，也不读取历史权重。
        cfg.pop("reference_run_templates", None)
        cfg["waterlevel_initialization"] = "train_from_scratch"
    if cfg["geometry_mode"] not in {"snapshot", "causal"}:
        raise ValueError("geometry_mode只能为snapshot或causal")
    if not cfg["ratios"] or not cfg["model_seeds"] or not cfg["mask_seeds"]:
        raise ValueError("留出比例与两类种子列表不能为空")
    if len(set(cfg["ratios"])) != len(cfg["ratios"]):
        raise ValueError("留出比例不能重复")
    if any(not 0 <= v < 1 for v in cfg["ratios"]):
        raise ValueError("留出比例须满足0<=ratio<1")
    for key in ("lstm_epochs", "converter_epochs", "sequence_length", "train_stride",
                "valid_stride", "test_stride", "batch_size", "hidden_size",
                "converter_max_pairs_per_basin"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key}必须为正整数")
    if not cfg["converter_widths"] or any(int(w) < 1 for w in cfg["converter_widths"]):
        raise ValueError("转换器候选宽度必须为正整数")
    if cfg.get("waterlevel_initialization", "train_from_scratch") not in (
            "train_from_scratch", "original_strong_checkpoint"):
        raise ValueError("水位模型来源只能是重新训练或直接复用原强配置权重")
    if cfg.get("waterlevel_initialization") == "original_strong_checkpoint" and not cfg.get("reference_run_templates"):
        raise ValueError("直接复用水位模型时必须明确给出原强配置参考目录")
    if cfg.get("target_scaling", "physical") not in ("physical", "observed"):
        raise ValueError("Q归一化只能是physical或observed")
    return cfg


def choose_device(requested: str, preflight: bool, smoke: bool) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if not preflight and not smoke:
            raise RuntimeError("未检测到CUDA；正式CPU训练请显式传入--device cpu")
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"指定了{requested}但CUDA不可用，不自动降级到CPU")
    return requested


def scenario_list(cfg, basins):
    scenarios = []
    for ratio in cfg["ratios"]:
        # 完整标签条件不存在掩膜随机性，不重复跑三份相同实验。
        for mask_seed in (cfg["mask_seeds"][:1] if ratio == 0 else cfg["mask_seeds"]):
            held, retained = heldout_split(basins, ratio, mask_seed,
                                          cfg.get("holdout_selection", "legacy_cross_section"))
            tag = f"q{round(ratio * 100):02d}_mask{mask_seed}"
            scenarios.append({"tag": tag, "ratio": ratio, "mask_seed": mask_seed,
                              "heldout": held, "retained": retained})
    return scenarios


def make_train_config(cfg, architecture, seed, device):
    return TrainConfig(
        architecture=architecture, seq_length=cfg["sequence_length"],
        window_step_train=cfg["train_stride"], window_step_eval=cfg["valid_stride"],
        window_step_test=cfg["test_stride"], batch_size=cfg["batch_size"],
        hidden_size=cfg["hidden_size"], max_epochs=cfg["lstm_epochs"],
        patience=None, target_scaling=cfg.get("target_scaling", "physical"), attr_set=cfg["attribute_set"],
        dropout_rate=cfg.get("dropout_rate", 0.2), learning_rate=cfg.get("learning_rate", 1e-3),
        lr_schedule=cfg.get("lr_schedule", "cosine"), lr_min_factor=cfg.get("lr_min_factor", 0.02),
        loss_norm=cfg.get("loss_norm", "per_valid"),
        model_seed=seed, device=device,
    )


def preflight(prepared, geometry, audit, scenarios, cfg):
    target_scaling = cfg.get("target_scaling", "physical")
    observed_scale = target_scaling == "observed"
    report = {
        "protocol": PROTOCOL, "n_basins": len(prepared.basins),
        "geometry_mode": cfg["geometry_mode"],
        "geometry_interpretation": "retrospective_extra_geometry_information"
        if cfg["geometry_mode"] == "snapshot" else "geometry_before_train_end",
        "n_geometry_available": int((geometry.geom_missing == 0).sum()),
        "n_geometry_missing": int(geometry.geom_missing.sum()),
        "n_post_train_geometry": int(audit.post_train_geometry.sum()),
        "geometry_is_full_absolute_cross_section": False,
        "target_scaling": target_scaling,
        "heldout_q_used_for_training_or_selection": observed_scale,
        "heldout_q_labels_used_in_loss_or_validation_score": False,
        "heldout_q_statistics_used_for_scaling": observed_scale,
        "information_boundary": "known_q_scale_control" if observed_scale else "q_scale_from_retained_basins",
        "original_frozen_split_uses_historical_q_availability": True,
        "n_time_discontinuities": int((~prepared.time_valid).sum()),
        "time_gap_policy": "exclude_windows_crossing_missing_3h_times",
        "numerical_environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "smoke_not_paper_result": cfg["smoke"], "scenarios": [],
    }
    for scenario in scenarios:
        hidden, hidden_valid = q_masks(prepared, scenario["heldout"])
        scaling = prepared.make_scaling(target_scaling, fit_basins=scenario["retained"])
        ds_train = WindowDataset(prepared, "train", cfg["sequence_length"],
                                 cfg["train_stride"], hidden=hidden, scaling=scaling)
        ds_valid = WindowDataset(prepared, "valid", cfg["sequence_length"],
                                 cfg["valid_stride"], hidden=hidden_valid, scaling=scaling)
        for basin in scenario["heldout"]:
            bi = prepared.basin_index[basin]
            for split, mask in (("train", hidden["flow"]), ("valid", hidden_valid["flow"])):
                lo, hi = prepared.split_range(basin, split)
                if not mask[bi, lo:hi + 1].all():
                    raise AssertionError(f"{basin}的{split}Q未完全屏蔽")
            lo, hi = prepared.split_range(basin, "test")
            if hidden["flow"][bi, lo:hi + 1].any() or hidden_valid["flow"][bi, lo:hi + 1].any():
                raise AssertionError("Q掩膜进入了测试段")
        report["scenarios"].append({
            **scenario, "n_train": len(ds_train), "n_valid": len(ds_valid),
            "train_label_counts": ds_train.label_counts(),
            "valid_label_counts": ds_valid.label_counts(),
            "physical_scale_fit_basins": scenario["retained"] if not observed_scale else [],
            "observed_q_scale_basins": prepared.basins if observed_scale else [],
        })
    return report


def training_checkpoint(prepared, cfg, train_cfg, folder, run_id,
                        hidden=None, hidden_valid=None, fit_basins=None):
    """配置/来源指纹相同才允许续跑，完成标记最后写入。"""
    folder.mkdir(parents=True, exist_ok=True)
    finished, checkpoint = folder / "training_done.json", folder / "model.pt"
    if finished.exists():
        previous = json.loads(finished.read_text(encoding="utf-8"))
        if previous["run_fingerprint"] != run_id:
            raise RuntimeError(f"运行目录存在不兼容结果，请改输出目录: {folder}")
        if not checkpoint.is_file():
            raise RuntimeError(f"完成标记存在但模型丢失: {folder}")
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"\n训练 {folder.name}，参数输入属性维数={prepared.attrs.shape[1]}", flush=True)
    result = train_model(prepared, train_cfg, hidden=hidden, hidden_valid=hidden_valid,
                         fit_basins=fit_basins, verbose=True)
    if result["best_epoch"] < 1:
        raise RuntimeError("没有有效验证checkpoint，不能把末轮权重当作成功结果")
    blob = {
        "state_dict": result["state_dict"], "model_config": result["model_config"],
        "training_config": asdict(train_cfg), "run_fingerprint": run_id,
        "attr_columns": list(prepared.attr_columns),
        "target_mean": result["scaling"].mean, "target_std": result["scaling"].std,
        "scaling_info": result["scaling_info"],
    }
    torch.save(blob, checkpoint)
    metadata = {key: result[key] for key in (
        "best_epoch", "best_val_score", "epochs_run", "elapsed_sec",
        "n_train_samples", "n_valid_samples", "n_test_samples",
        "train_label_counts", "valid_label_counts", "model_config", "scaling_info",
    )}
    metadata.update(run_fingerprint=run_id, protocol=PROTOCOL, smoke_not_paper_result=cfg["smoke"])
    write_json(folder / "training_history.json", result["history"])
    write_json(finished, metadata)
    return blob


def flow_metrics(prepared, indices, positions, predicted, scenario, seed, stage,
                 variant, geometry, cfg, input_source="predicted_h"):
    """最终评价才读取测试Q，按所有站输出，缺评价标签的站不静默消失。"""
    rows = []
    for bi, basin in enumerate(prepared.basins):
        selected = indices == bi
        pos, pred = positions[selected], predicted[selected]
        obs = prepared.targets_raw["flow"][bi, pos]
        ok = np.isfinite(obs) & np.isfinite(pred)
        metrics = {"n": int(ok.sum()), "nse": float("nan")}
        if ok.sum() >= 2:
            full = np.arange(pos.min(), pos.max() + 1)
            series_obs, series_pred = np.full(len(full), np.nan), np.full(len(full), np.nan)
            series_obs[pos - full[0]], series_pred[pos - full[0]] = obs, pred
            metrics = basin_metrics(obs[ok], pred[ok], series_obs, series_pred)
        rows.append({
            "basin": basin, "stage": stage, "variant": variant,
            "ratio": scenario["ratio"], "mask_seed": scenario["mask_seed"],
            "model_seed": seed, "held_out": basin in scenario["heldout"],
            "geometry_available": not bool(geometry.loc[basin, "geom_missing"]),
            "geometry_mode": cfg["geometry_mode"], "h_input_source": input_source,
            "target_scaling": cfg.get("target_scaling", "physical"),
            "smoke_not_paper_result": cfg["smoke"],
            "n_negative_predictions": int((pred[np.isfinite(pred)] < 0).sum()),
            **metrics,
        })
    return pd.DataFrame(rows)


def save_prediction(folder, indices, positions, predicted, task="flow"):
    np.savez_compressed(folder / "prediction.npz", basin_idx=indices,
                        target_pos=positions, **{f"pred_{task}": predicted})


def cached_h_prediction(prepared, cfg, seed, device, root, experiment_id):
    folder = root / "waterlevel_models" / f"seed{seed}"
    run_id = fingerprint({"experiment": experiment_id, "stage": "waterlevel_only", "seed": seed})
    # H模型的尺度视图完全不含Q，因此可跨Q掩膜情景复用。
    train_cfg = make_train_config(cfg, "single_waterlevel", seed, device)
    if cfg.get("waterlevel_initialization") == "original_strong_checkpoint":
        source = reference_folder(ROOT, cfg, "single_waterlevel", seed)
        print(f"直接复用原强配置H权重 seed{seed}: {source.name}", flush=True)
        blob = reuse_h_checkpoint(prepared, cfg, train_cfg, source, folder, run_id)
    else:
        view = WaterlevelOnlyPrepared(prepared)
        blob = training_checkpoint(view, cfg, train_cfg, folder, run_id, fit_basins=prepared.basins)
    path = folder / "prediction.npz"
    if not path.exists():
        result = infer(prepared, blob["state_dict"], blob["model_config"], "test",
                       cfg["sequence_length"], cfg["test_stride"], device)
        save_prediction(folder, result["basin_idx"], result["target_pos"],
                        result["pred"]["waterlevel"], "h_normalized")
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def run_stage1(prepared, geometry, cfg, scenario, seed, g, g_shuffled, h_prediction,
               root, device, experiment_id):
    retained = scenario["retained"]
    variants = {
        "h_only": None,
        "h_attrs": prepared.attrs,
        "h_attrs_geometry": np.column_stack([prepared.attrs, g]).astype(np.float32),
    }
    if cfg["sham_controls"]:
        variants["h_attrs_geometry_shuffled"] = np.column_stack(
            [prepared.attrs, g_shuffled]).astype(np.float32)
    chosen_geometry_width = None
    indices, positions = h_prediction["basin_idx"], h_prediction["target_pos"]
    predicted_h = h_prediction["pred_h_normalized"]
    for variant, static in variants.items():
        folder = root / "stage1" / scenario["tag"] / f"seed{seed}" / variant
        folder.mkdir(parents=True, exist_ok=True)
        run_id = fingerprint({"experiment": experiment_id, "scenario": scenario["tag"],
                              "stage": "stage1", "variant": variant, "seed": seed})
        done = folder / "done.json"
        if done.exists():
            metadata = json.loads(done.read_text(encoding="utf-8"))
            if metadata["run_fingerprint"] != run_id:
                raise RuntimeError(f"转换目录指纹不同: {folder}")
            for required in ("model.pt", "metrics.csv", "prediction.npz"):
                if not (folder / required).is_file():
                    raise RuntimeError(f"转换完成标记对应的文件缺失: {folder / required}")
            if variant == "h_attrs_geometry":
                chosen_geometry_width = metadata["hidden"]
            continue
        print(f"拟合独立转换器 {scenario['tag']} seed{seed} {variant}", flush=True)
        forced_width = chosen_geometry_width if variant.endswith("_shuffled") else None
        model, scale, metadata = fit_regional_converter(
            prepared, retained, static, cfg["converter_widths"], cfg["converter_epochs"],
            seed, device, cfg["converter_max_pairs_per_basin"], forced_width,
            target_scaling=cfg.get("target_scaling", "physical"))
        if variant == "h_attrs_geometry":
            chosen_geometry_width = metadata["hidden"]
        x = predicted_h[:, None]
        if static is not None:
            x = np.column_stack([x, static[indices]]).astype(np.float32)
        pred_n = predict_converter(model, x.astype(np.float32), device)
        physical = pred_n * scale.std["flow"][indices] + scale.mean["flow"][indices]
        scores = flow_metrics(prepared, indices, positions, physical, scenario, seed,
                              "stage1", variant, geometry, cfg)
        scores.to_csv(folder / "metrics.csv", index=False, encoding="utf-8-sig")
        save_prediction(folder, indices, positions, physical)
        if cfg["observed_h_diagnostic"]:
            # 实测H只作同一冻结转换器的诊断，不能用诊断分数选宽度或模型。
            observed_h = prepared.targets["waterlevel"][indices, positions]
            observed_x = observed_h[:, None]
            if static is not None:
                observed_x = np.column_stack([observed_x, static[indices]]).astype(np.float32)
            valid = np.isfinite(observed_x).all(axis=1)
            diagnostic = np.full(len(indices), np.nan, np.float32)
            if valid.any():
                normalized = predict_converter(model, observed_x[valid].astype(np.float32), device)
                diagnostic[valid] = (normalized * scale.std["flow"][indices[valid]]
                                     + scale.mean["flow"][indices[valid]])
            flow_metrics(prepared, indices, positions, diagnostic, scenario, seed, "stage1",
                         variant, geometry, cfg, "observed_h_diagnostic").to_csv(
                folder / "diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_dim": metadata["input_dim"], "hidden": metadata["hidden"],
            "flow_mean": scale.mean["flow"], "flow_std": scale.std["flow"],
            "flow_scale_info": scale.info, "variant": variant, "run_fingerprint": run_id,
        }, folder / "model.pt")
        metadata.update(run_fingerprint=run_id, protocol=PROTOCOL, variant=variant,
                        heldout=scenario["heldout"], model_seed=seed,
                        geometry_mode=cfg["geometry_mode"], smoke_not_paper_result=cfg["smoke"])
        write_json(done, metadata)


def run_stage2(prepared, geometry, cfg, scenario, seed, g, g_shuffled,
               root, device, experiment_id):
    variants = {
        "single_flow": ("single_flow", None),
        "dual_head": ("dual_head", None),
        "single_flow_geometry": ("single_flow", g),
        "dual_head_geometry": ("dual_head", g),
    }
    if cfg["sham_controls"]:
        variants.update(single_flow_shuffled=("single_flow", g_shuffled),
                        dual_head_shuffled=("dual_head", g_shuffled))
    hidden, hidden_valid = q_masks(prepared, scenario["heldout"])
    geometry_columns = list(FEATURE_COLUMNS) + ["geom_missing"]
    for variant, (architecture, inputs) in variants.items():
        folder = root / "stage2" / scenario["tag"] / f"seed{seed}" / variant
        view = with_attributes(prepared, inputs, geometry_columns)
        run_id = fingerprint({"experiment": experiment_id, "scenario": scenario["tag"],
                              "stage": "stage2", "variant": variant, "seed": seed})
        blob = training_checkpoint(view, cfg, make_train_config(cfg, architecture, seed, device),
                                   folder, run_id, hidden, hidden_valid, scenario["retained"])
        done = folder / "done.json"
        if done.exists():
            previous = json.loads(done.read_text(encoding="utf-8"))
            if previous["run_fingerprint"] != run_id:
                raise RuntimeError(f"阶段二完成结果指纹不同: {folder}")
            if not (folder / "metrics.csv").exists() or not (folder / "prediction.npz").exists():
                raise RuntimeError(f"阶段二完成文件缺失: {folder}")
            continue
        result = infer(view, blob["state_dict"], blob["model_config"], "test",
                       cfg["sequence_length"], cfg["test_stride"], device)
        indices, positions = result["basin_idx"], result["target_pos"]
        physical = (result["pred"]["flow"] * blob["target_std"]["flow"][indices]
                    + blob["target_mean"]["flow"][indices])
        flow_metrics(prepared, indices, positions, physical, scenario, seed, "stage2",
                     variant, geometry, cfg, "meteorology_to_q").to_csv(
            folder / "metrics.csv", index=False, encoding="utf-8-sig")
        save_prediction(folder, indices, positions, physical)
        write_json(done, {"run_fingerprint": run_id, "protocol": PROTOCOL,
                          "variant": variant, "heldout": scenario["heldout"],
                          "model_seed": seed, "geometry_mode": cfg["geometry_mode"],
                          "smoke_not_paper_result": cfg["smoke"]})


def summarize(root: Path, experiment_id: str):
    frames = []
    for stage in ("stage1", "stage2"):
        for done in (root / stage).glob("*/seed*/*/done.json"):
            metadata = json.loads(done.read_text(encoding="utf-8"))
            folder = done.parent
            for name in ("metrics.csv", "diagnostic_metrics.csv"):
                if (folder / name).exists():
                    frame = pd.read_csv(folder / name, dtype={"basin": str})
                    frame["run_fingerprint"] = metadata["run_fingerprint"]
                    frames.append(frame)
    if not frames:
        return
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(root / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    groups = ["stage", "variant", "ratio", "mask_seed", "model_seed", "h_input_source", "target_scaling"]
    summary = []
    for key, sub in scores.groupby(groups, dropna=False):
        for scope in ("all_86", "q_heldout", "q_retained", "q_heldout_with_geometry"):
            selected = {
                "all_86": sub,
                "q_heldout": sub.loc[sub.held_out],
                "q_retained": sub.loc[~sub.held_out],
                "q_heldout_with_geometry": sub.loc[sub.held_out & sub.geometry_available],
            }[scope]
            valid = selected.nse.dropna()
            record = {**dict(zip(groups, key)), "scope": scope,
                      "n_basins_reported": len(selected), "n_valid_nse": len(valid),
                      "nse_mean": float(valid.mean()) if len(valid) else np.nan,
                      "nse_median": float(valid.median()) if len(valid) else np.nan,
                      "n_nse_negative": int((valid < 0).sum())}
            for metric in ("kge", "pbias", "pearson_r", "peak_time_mae_h"):
                record[f"{metric}_mean"] = selected[metric].mean() if metric in selected else np.nan
            summary.append(record)
    pd.DataFrame(summary).to_csv(root / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    # 先在同一站、同一掩膜与模型种子上配对，禁止跨配置直接减总体均值。
    deployed = scores.loc[scores.h_input_source != "observed_h_diagnostic"]
    contrasts = {
        "stage1_geometry_gain": ("h_attrs_geometry", "h_attrs"),
        "stage1_geometry_vs_shuffle": ("h_attrs_geometry", "h_attrs_geometry_shuffled"),
        "stage2_h_gain_without_geometry": ("dual_head", "single_flow"),
        "stage2_h_gain_with_geometry": ("dual_head_geometry", "single_flow_geometry"),
        "stage2_geometry_gain_single": ("single_flow_geometry", "single_flow"),
        "stage2_geometry_gain_dual": ("dual_head_geometry", "dual_head"),
        "stage2_geometry_vs_shuffle_single": ("single_flow_geometry", "single_flow_shuffled"),
        "stage2_geometry_vs_shuffle_dual": ("dual_head_geometry", "dual_head_shuffled"),
        "geometry_converter_vs_geometry_dual": ("dual_head_geometry", "h_attrs_geometry"),
    }
    paired = []
    keys = ["basin", "ratio", "mask_seed", "model_seed", "target_scaling"]
    for name, (model, control) in contrasts.items():
        left = deployed.loc[deployed.variant == model, keys + ["nse", "held_out", "geometry_available"]]
        right = deployed.loc[deployed.variant == control, keys + ["nse"]]
        joined = left.merge(right, on=keys, suffixes=("_model", "_control"), validate="one_to_one")
        joined["delta_nse"] = joined.nse_model - joined.nse_control
        joined["comparison"] = name
        paired.append(joined)
    if paired:
        pairs = pd.concat(paired, ignore_index=True)
        pairs.to_csv(root / "paired_differences.csv", index=False, encoding="utf-8-sig")
        gains = pairs.loc[pairs.comparison.isin(
            ["stage2_h_gain_without_geometry", "stage2_h_gain_with_geometry"])]
        before = gains.loc[gains.comparison == "stage2_h_gain_without_geometry", keys + ["delta_nse"]]
        after = gains.loc[gains.comparison == "stage2_h_gain_with_geometry", keys + ["delta_nse"]]
        interaction = after.merge(before, on=keys, suffixes=("_with_geometry", "_without_geometry"))
        interaction["h_geometry_interaction"] = (
            interaction.delta_nse_with_geometry - interaction.delta_nse_without_geometry)
        interaction.to_csv(root / "h_geometry_interaction.csv", index=False, encoding="utf-8-sig")
    write_json(root / "summary_manifest.json", {
        "experiment_fingerprint": experiment_id, "n_metric_rows": len(scores),
        "selection_used_test_scores": False, "inferential_statistics_added": False,
    })


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = settings_from_args(args)
    device = choose_device(args.device, args.preflight, args.smoke)
    if device == "cpu":
        torch.set_num_threads(2)
    original = load_basin_ids()
    prepared = PreparedData(attr_set=cfg["attribute_set"])
    validate_population(prepared, original)
    reference_audit = audit_references(ROOT, prepared, cfg, make_train_config)
    package_index = ROOT / "data/cross_sections/profiles/profile_index.csv"
    source_index = ROOT / "research_notes/USGS 原86 历年断面核验/profiles/profile_index.csv"
    index = args.profile_index or (package_index if package_index.exists() else source_index)
    geometry, audit = build_geometry(index, prepared.basins, prepared.splits, cfg["geometry_mode"])
    scenarios = scenario_list(cfg, prepared.basins)
    output = args.out_dir or ROOT / "results" / (
        cfg.get("output_name", f"cross_section86_{cfg['geometry_mode']}") + ("_smoke" if args.smoke else ""))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: file_sha256(ROOT / name) for name in DATA_FILES}
    code_files = [ROOT / "src/training/trainer.py", ROOT / "src/models/lstm_models.py"]
    code_files += sorted((ROOT / "src/cross_section_experiment").glob("*.py"))
    code_files += sorted((ROOT / "src/pipeline").glob("*.py"))
    code_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files}
    spec = {
        "protocol": PROTOCOL, "settings": cfg, "basins": prepared.basins,
        "scenarios": scenarios, "source_hashes": source_hashes, "code_hashes": code_hashes,
        "geometry_sources": audit.to_dict("records"),
        "normalization": "original86_cached_attributes_onehot_identity; " + (
            "observed_q_from_each_basin_training" if cfg.get("target_scaling", "physical") == "observed"
            else "physical_q_from_retained"),
        "target_scaling": cfg.get("target_scaling", "physical"),
        "heldout_q_statistics_used_for_scaling": cfg.get("target_scaling", "physical") == "observed",
        "numerical_environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "geometry_is_absolute_bed_elevation": False,
        "geometry_is_retrospective": cfg["geometry_mode"] == "snapshot",
        "frozen_splits_use_original_q_h_availability": True,
        "time_gap_policy": "exclude_windows_crossing_missing_3h_times",
    }
    if reference_audit:
        spec["reference_model_audit"] = reference_audit
    experiment_id = fingerprint(spec)
    manifest = output / "experiment.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["experiment_fingerprint"] != experiment_id:
            raise RuntimeError(f"输出目录含不同配置/来源/代码的实验，请使用新的--out-dir: {output}")
    else:
        write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    geometry.to_csv(output / "geometry_features.csv", encoding="utf-8-sig")
    audit.to_csv(output / "geometry_audit.csv", index=False, encoding="utf-8-sig")
    report = preflight(prepared, geometry, audit, scenarios, cfg)
    if reference_audit:
        write_json(output / "reference_model_audit.json", reference_audit)
        report["reference_model_audit"] = reference_audit
    write_json(output / "preflight.json", report)
    per_scenario = 6 if cfg["sham_controls"] else 4
    stage2_runs = len(scenarios) * len(cfg["model_seeds"]) * per_scenario
    stage1_h_runs = len(cfg["model_seeds"])
    print(f"原86总体={len(prepared.basins)}；有效断面={report['n_geometry_available']}；"
          f"事后断面={report['n_post_train_geometry']}；device={device}", flush=True)
    h_mode = "直接复用" if cfg.get("waterlevel_initialization") == "original_strong_checkpoint" else "训练"
    print(f"阶段一{h_mode}{stage1_h_runs}个H模型；阶段二LSTM最多{stage2_runs}次；"
          "相同指纹的完成运行自动跳过。", flush=True)
    if cfg["geometry_mode"] == "snapshot":
        print("本实验使用2021—2023固定断面快照，属于给定事后几何信息的回溯研究。", flush=True)
    if args.smoke:
        print("当前为烟雾测试：轮数、窗口、步长均已缩减，不是论文实验。", flush=True)
    if args.preflight:
        print(f"预检完成，尚未训练。输出: {output}", flush=True)
        return 0
    h_predictions = {}
    if args.stage in ("both", "stage1"):
        for seed in cfg["model_seeds"]:
            h_predictions[seed] = cached_h_prediction(prepared, cfg, seed, device, output, experiment_id)
    for scenario in scenarios:
        scaler = fit_geometry_scaler(geometry, scenario["retained"])
        g = transform_geometry(geometry, prepared.basins, scaler)
        g_shuffled = shuffled_geometry(g, scenario["mask_seed"] + 2000)
        write_json(output / "scenario_audits" / f"{scenario['tag']}.json",
                   {**scenario, "geometry_scaler": scaler, "geometry_shuffle_seed":
                    scenario["mask_seed"] + 2000, "flow_training_scope": "retained_only"})
        for seed in cfg["model_seeds"]:
            if args.stage in ("both", "stage1"):
                run_stage1(prepared, geometry, cfg, scenario, seed, g, g_shuffled,
                           h_predictions[seed], output, device, experiment_id)
            if args.stage in ("both", "stage2"):
                run_stage2(prepared, geometry, cfg, scenario, seed, g, g_shuffled,
                           output, device, experiment_id)
        summarize(output, experiment_id)
    write_json(output / f"completed_{args.stage}.json",
               {"experiment_fingerprint": experiment_id, "stage": args.stage,
                "smoke_not_paper_result": cfg["smoke"]})
    print(f"两个阶段的所选任务已完成: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

