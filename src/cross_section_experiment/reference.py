"""核对历史强配置，并可直接复用历史单任务水位权重。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from cross_section_experiment.features import file_sha256
from cross_section_experiment.protocol import PROTOCOL, WaterlevelOnlyPrepared, heldout_split, write_json
from models.lstm_models import build_model


def reference_folder(root: Path, cfg: dict, architecture: str, seed: int) -> Path:
    template = cfg["reference_run_templates"][architecture]
    path = Path(template.format(seed=seed))
    return path if path.is_absolute() else root / path


def validate_reference(prepared, requested, record, blob):
    """字段或张量形状不匹配即停止，不能只凭文件名认定同一模型。"""
    requested = asdict(requested)
    original = record["config"]
    differences = {key: [original.get(key), value] for key, value in requested.items()
                   if key != "device" and original.get(key) != value}
    if differences:
        raise ValueError(f"历史强配置与本次训练配置不同: {differences}")
    architecture = requested["architecture"]
    model = build_model(architecture, prepared.forcing.shape[-1], prepared.attrs.shape[-1],
                        requested["hidden_size"], requested["dropout_rate"], requested["proj_size"])
    if model.config() != record["model_config"] or blob["model_config"] != record["model_config"]:
        raise ValueError("历史模型结构、输入维数或参数量与本次不一致")
    if blob["config"] != original:
        raise ValueError("历史checkpoint与run.json的训练配置不一致")
    model.load_state_dict(blob["state_dict"], strict=True)
    if record["scenario"] != "complete":
        raise ValueError("历史参考必须为完整标签条件，不能混用留出模型")
    return model.config()


def audit_references(root: Path, prepared, cfg: dict, config_factory) -> dict:
    """正式开始前审计全部参考模型，将源文件哈希纳入本次指纹。"""
    templates = cfg.get("reference_run_templates", {})
    if not templates:
        return {}
    required = {"single_waterlevel", "single_flow", "dual_head"}
    if set(templates) != required:
        raise ValueError("参考清单必须包含单H、单Q和双头三种原模型")
    holdout_audit = {}
    if cfg.get("reference_holdouts"):
        path = root / cfg["reference_holdouts"]
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen["basins"] != prepared.basins:
            raise ValueError("冻结留出名单的总体或站号顺序与原86缓存不同")
        matches = []
        for ratio in cfg["ratios"]:
            for seed in cfg["mask_seeds"]:
                held, retained = heldout_split(prepared.basins, ratio, seed,
                                               cfg.get("holdout_selection", "legacy_cross_section"))
                tag = f"q{round(ratio * 100):02d}_mask{seed}"
                if ratio == 0:
                    continue
                if held != frozen["scenarios"][tag]["heldout"]:
                    raise ValueError(f"{tag}的隐藏Q流域不等于原实验名单")
                matches.append({"tag": tag, "n_heldout": len(held), "matches_original": True})
        holdout_audit = {"source": cfg["reference_holdouts"], "sha256": file_sha256(path),
                         "historical_runs_verified": frozen["n_original_runs_verified"], "scenarios": matches}
    rows = []
    for seed in cfg["model_seeds"]:
        for architecture in sorted(required):
            folder = reference_folder(root, cfg, architecture, seed)
            record = json.loads((folder / "run.json").read_text(encoding="utf-8-sig"))
            blob = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
            model_config = validate_reference(prepared, config_factory(cfg, architecture, seed, "cpu"), record, blob)
            metrics = pd.read_csv(folder / "metrics.csv", dtype={"basin": str})
            if set(metrics.basin) != set(prepared.basins):
                raise ValueError(f"历史参考不是原86总体: {folder}")
            rows.append({"architecture": architecture, "model_seed": seed,
                         "folder": str(folder.relative_to(root)), "model_config": model_config,
                         "source_model_sha256": file_sha256(folder / "model.pt"),
                         "source_run_sha256": file_sha256(folder / "run.json"),
                         "source_metrics_sha256": file_sha256(folder / "metrics.csv"),
                         "training_config_matches": True, "state_shapes_match": True})
    return {"models": rows, "waterlevel_initialization": cfg.get("waterlevel_initialization", "train_from_scratch"),
            "holdout_audit": holdout_audit,
            "original_batch_sizes": {"training": cfg["batch_size"], "validation_and_inference": 4096},
            "stage2_initialization": "train_from_scratch_same_architecture_and_hyperparameters",
            "stage2_additional_inputs": "six_geometry_features_and_missing_indicator_for_geometry_variants",
            "protocol_differences_from_original": ["heldout_Q_hidden_in_both_training_and_validation",
                                                    "new_training_and_inference_exclude_time_gap_windows",
                                                    "stage1_inference_not_filtered_by_observed_test_H_availability"],
            "reused_H_training_windows": "original_reference_windows_without_new_time_gap_exclusion"}


def reuse_h_checkpoint(prepared, cfg, train_cfg, source: Path, folder: Path, run_id: str):
    """原H权重逐张量复用；本次仅重新推理，不微调，不构造Q尺度。"""
    folder.mkdir(parents=True, exist_ok=True)
    done, checkpoint = folder / "training_done.json", folder / "model.pt"
    if done.exists():
        previous = json.loads(done.read_text(encoding="utf-8-sig"))
        if previous["run_fingerprint"] != run_id:
            raise ValueError(f"水位模型完成标记的指纹不同: {folder}")
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    record = json.loads((source / "run.json").read_text(encoding="utf-8-sig"))
    original = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    validate_reference(prepared, train_cfg, record, original)
    if original["model_config"]["tasks"] != ["waterlevel"]:
        raise ValueError("只能复用单任务H模型，不能拿双头或单Q权重代替")
    scale = WaterlevelOnlyPrepared(prepared).make_scaling()
    source_info = {"folder": str(source), "model_sha256": file_sha256(source / "model.pt"),
                   "run_sha256": file_sha256(source / "run.json"), "weights_reused_without_finetuning": True}
    blob = {"state_dict": original["state_dict"], "model_config": original["model_config"],
            "training_config": asdict(train_cfg), "run_fingerprint": run_id,
            "attr_columns": list(prepared.attr_columns), "target_mean": scale.mean,
            "target_std": scale.std, "scaling_info": {"mode": "waterlevel_only", **scale.info},
            "source_checkpoint": source_info}
    torch.save(blob, checkpoint)
    metadata = {key: record[key] for key in ("best_epoch", "best_val_score", "epochs_run",
                "n_train_samples", "n_valid_samples", "n_test_samples", "train_label_counts",
                "valid_label_counts", "model_config")}
    metadata.update(elapsed_sec=0.0, source_elapsed_sec=record.get("elapsed_sec"),
                    training_performed_this_run=False, source_checkpoint=source_info,
                    training_time_gap_policy="original_reference_windows", inference_time_gap_policy="exclude_gap_windows",
                    scaling_info=blob["scaling_info"], run_fingerprint=run_id, protocol=PROTOCOL,
                    smoke_not_paper_result=cfg["smoke"])
    write_json(folder / "training_history.json", record.get("history", []))
    write_json(done, metadata)
    return blob
