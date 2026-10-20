"""只读复核测量几何重拟合结果，报告写入独立目录，不训练或改写模型。"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cross_section86_survey_refit_v1"
OUTPUT = ROOT / "reports/cross_section86_survey_refit_review"
RUN_KEYS = ["target_scaling", "ratio", "mask_seed", "model_seed"]
VARIANTS = ["h_only", "h_geometry_scale", "h_survey_stage", "h_survey_geometry",
            "h_survey_geometry_shuffled", "h_only_capacity_control"]
LABELS = {
    "h_only": "仅H",
    "h_geometry_scale": "H＋面积/湿润长度/宽度",
    "h_survey_stage": "H＋测量时水位",
    "h_survey_geometry": "H＋全部测量几何及测量时水位",
    "h_survey_geometry_shuffled": "完整输入联合打乱",
    "h_only_capacity_control": "仅H，匹配完整模型参数量",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def markdown(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def compare_frames(name, actual, expected, keys):
        try:
            pd.testing.assert_frame_equal(
                actual[expected.columns].sort_values(keys).reset_index(drop=True),
                expected.sort_values(keys).reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9,
            )
            check(name, True)
        except AssertionError as error:
            check(name, False, str(error)[:1200])

    spec = read_json(SOURCE / "experiment.json")
    basins = spec["basins"]
    scenes = {scene["tag"]: scene for scene in spec["scenarios"]}
    frozen = read_json(ROOT / "configs/cross_section86_original_holdouts.json")["scenarios"]
    check("原86总体", len(basins) == len(set(basins)) == 86)
    check("全部六个预定变体", set(spec["variants"]) == set(VARIANTS))
    check("水位模型未重训", not spec["waterlevel_training_performed"])
    check("标记为事后静态快照与探索性设计", spec["geometry_mode"] == "snapshot"
          and spec["new_design_after_previous_test_inspection"]
          and spec["survey_stage_is_extra_retrospective_observation"])
    for tag, scene in scenes.items():
        check(f"冻结名单/{tag}", scene["heldout"] == frozen[tag]["heldout"]
              and not set(scene["heldout"]) & set(scene["retained"])
              and set(scene["heldout"] + scene["retained"]) == set(basins))
    for group in ("data_hashes", "code_hashes"):
        for name, expected in spec[group].items():
            check(f"{group}/{name}", sha256(ROOT / name) == expected)
    for scaling, source in spec["sources"].items():
        original = ROOT / "results" / f"cross_section86_strong_{scaling}_snapshot_v2"
        for seed, expected in source["H_predictions"].items():
            folder = original / "waterlevel_models" / f"seed{seed}"
            check(f"冻结H预测/{scaling}/{seed}", sha256(folder / "prediction.npz") == expected["prediction_sha256"])
            check(f"冻结H模型/{scaling}/{seed}", sha256(folder / "model.pt") == expected["model_sha256"])

    geometry = pd.read_csv(SOURCE / "survey_geometry.csv", dtype={"basin": str}).set_index("basin")
    old_geometry = pd.read_csv(ROOT / "results/cross_section86_strong_physical_snapshot_v2/geometry_audit.csv",
                               dtype={"basin": str}).set_index("basin")
    available = set(geometry.index[geometry.geom_missing == 0])
    check("81站合格几何", len(available) == 81 and set(geometry.index) == set(basins))
    for basin in available:
        record = geometry.loc[basin]
        check(f"同一原始剖面/{basin}", record.profile_id == old_geometry.loc[basin, "profile_id"]
              and record.source_sha256 == old_geometry.loc[basin, "source_sha256"])
        roots = [ROOT / "data/cross_sections", ROOT / "research_notes/USGS 原86 历年断面核验"]
        source_path = next(path / record.source_json for path in roots if (path / record.source_json).is_file())
        check(f"剖面哈希/{basin}", sha256(source_path) == record.source_sha256)

    # 直接从原始缓存和冻结切分重建测试时刻，不调用训练程序的评价函数。
    grid = None
    for name in ("mswep_precip_3h_86.parquet", "forcing_temperature_mean_3h_86.parquet",
                 "forcing_solar_radiation_3h_86.parquet"):
        index = pd.read_parquet(ROOT / "data/camelsh_exported" / name, columns=[]).index
        grid = index if grid is None else grid.intersection(index)
    grid = pd.DatetimeIndex(grid).sort_values()
    q = pd.read_parquet(ROOT / "data/camelsh_exported/flow_hourly_86.parquet")
    q = q.reindex(index=grid, columns=basins).to_numpy(dtype="float32").T
    splits = read_json(ROOT / "data/splits/basin_splits.json")["splits"]
    contiguous = np.r_[True, np.diff(grid.to_numpy()) == np.timedelta64(3, "h")]
    prefix = np.r_[0, np.cumsum(~contiguous)]
    groups = []
    for basin in basins:
        lo = grid.searchsorted(pd.Timestamp(splits[basin]["test_start"]), side="left")
        hi = grid.searchsorted(pd.Timestamp(splits[basin]["test_end"]), side="right") - 1
        positions = np.arange(max(lo, 480), hi + 1, dtype=np.int64)
        groups.append(positions[contiguous[positions] & (prefix[positions] == prefix[positions - 480])])
    expected_ids = np.concatenate([np.full(len(pos), bi, dtype=np.int64) for bi, pos in enumerate(groups)])
    expected_pos = np.concatenate(groups)
    edges = np.r_[0, np.cumsum([len(pos) for pos in groups])]
    observations = q[expected_ids, expected_pos]

    files = sorted(SOURCE.glob("*/stage1/*/*/*/done.json"))
    expected_keys = set(itertools.product(spec["scalings"], scenes, spec["model_seeds"], VARIANTS))
    actual_keys, frames, meta_rows, recomputed, old_baseline = [], [], [], [], []
    full_metadata = {}
    for number, done_path in enumerate(files, 1):
        folder = done_path.parent
        scaling, _, tag, seed_dir, variant = folder.relative_to(SOURCE).parts
        seed = int(seed_dir.removeprefix("seed"))
        label = f"{scaling}/{tag}/{seed_dir}/{variant}"
        scene, meta = scenes[tag], read_json(done_path)
        retained, hidden = set(scene["retained"]), set(scene["heldout"])
        actual_keys.append((scaling, tag, seed, variant))
        full_metadata[(scaling, tag, seed, variant)] = meta
        check(f"输出完整/{label}", all((folder / name).is_file() for name in
              ("model.pt", "prediction.npz", "metrics.csv", "diagnostic_metrics.csv")))
        check(f"隐藏Q训练隔离记录/{label}", set(meta["training_q_pair_basins"]) == retained
              and not set(meta["training_q_pair_basins"]) & hidden)
        inner_train, inner_valid = set(meta["inner_train_basins"]), set(meta["inner_validation_basins"])
        check(f"内部整站划分/{label}", not inner_train & inner_valid and inner_train | inner_valid == retained)
        check(f"Q尺度声明/{label}", (scaling == "physical" and not meta["heldout_q_statistics_used_for_scaling"]
              and set(meta["scale_fit_basins"]) == retained and not meta["observed_q_scale_basins"])
              or (scaling == "observed" and meta["heldout_q_statistics_used_for_scaling"]
                  and set(meta["observed_q_scale_basins"]) == set(basins)))
        for stage, expected in (("inner_geometry_scaler", inner_train), ("final_geometry_scaler", retained)):
            scaler = meta[stage]
            check(f"几何预处理来源/{stage}/{label}", scaler is None or set(scaler["fit_basins"]) == expected)
        choice = max(meta["candidates"], key=lambda row: row["best_score"])
        check(f"内部验证选模记录/{label}", meta["selected_by"] == "retained_basins_internal_whole_basin_validation"
              and meta["hidden"] == choice["width"] and meta["epochs_refit"] == choice["best_epoch"]
              and not meta["smoke_not_paper_result"])
        check(f"转换器参数量/{label}", meta["n_parameters"] == (meta["input_dim"] + 2) * meta["hidden"] + 1)
        check(f"训练和推理H来源/{label}", meta["training_h_source"] == "observed_training_h"
              and meta["inference_h_source"] == "frozen_original_h_prediction")

        metrics = pd.read_csv(folder / "metrics.csv", dtype={"basin": str})
        diagnostic = pd.read_csv(folder / "diagnostic_metrics.csv", dtype={"basin": str})
        for frame, h_source in ((metrics, "predicted_h"), (diagnostic, "observed_h_diagnostic")):
            check(f"站点和Q隐藏标记/{h_source}/{label}", len(frame) == 86
                  and set(frame.basin) == set(basins) and frame.basin.is_unique
                  and set(frame.loc[frame.held_out, "basin"]) == hidden
                  and set(frame.loc[frame.geometry_available, "basin"]) == available
                  and set(frame.h_input_source) == {h_source})
            frames.append(frame.assign(run_fingerprint=meta["run_fingerprint"]))
        with np.load(folder / "prediction.npz") as saved:
            ids, pos, pred = saved["basin_idx"], saved["target_pos"], saved["pred_flow"]
        check(f"完整测试时刻/{label}", np.array_equal(ids, expected_ids) and np.array_equal(pos, expected_pos))
        check(f"预测全有限/{label}", np.isfinite(pred).all())
        independent = []
        for bi, basin in enumerate(basins):
            part = slice(int(edges[bi]), int(edges[bi + 1]))
            observed, predicted = observations[part].astype("float64"), pred[part].astype("float64")
            keep = np.isfinite(observed) & np.isfinite(predicted)
            observed, predicted = observed[keep], predicted[keep]
            error = predicted - observed
            denominator = np.sum((observed - observed.mean()) ** 2)
            independent.append({"basin": basin, "n": int(keep.sum()),
                "nse": 1 - np.sum(error ** 2) / denominator if denominator > 0 else np.nan,
                "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))})
        independently = pd.DataFrame(independent).set_index("basin")
        stored = metrics.set_index("basin").loc[basins]
        check(f"原始标签有效数/{label}", np.array_equal(independently.n, stored.n))
        errors = {}
        for measure in ("nse", "rmse", "mae"):
            errors[measure] = float(np.max(np.abs(independently[measure].to_numpy() - stored[measure].to_numpy())))
            check(f"独立重算{measure}/{label}", np.allclose(independently[measure], stored[measure],
                  rtol=2e-6, atol=3e-6, equal_nan=True), errors[measure])
        recomputed.append({"run": label, "n_basins": 86, **errors})
        if variant == "h_only":
            old_folder = ROOT / "results" / f"cross_section86_strong_{scaling}_snapshot_v2" / "stage1" / tag / seed_dir / variant
            with np.load(old_folder / "prediction.npz") as old:
                identical = np.array_equal(pred, old["pred_flow"])
                max_error = float(np.max(np.abs(pred.astype("float64") - old["pred_flow"].astype("float64"))))
                paired_layout = np.array_equal(ids, old["basin_idx"]) and np.array_equal(pos, old["target_pos"])
            old_scores = pd.read_csv(old_folder / "metrics.csv", dtype={"basin": str}).set_index("basin").loc[basins]
            nse_change = stored.nse - old_scores.nse
            heldout_mean_change = float(nse_change.loc[scene["heldout"]].mean())
            old_baseline.append({"target_scaling": scaling, "tag": tag, "model_seed": seed,
                "bitwise_equal": identical, "maximum_prediction_difference": max_error,
                "heldout_mean_nse_change": heldout_mean_change,
                "maximum_absolute_station_nse_change": float(nse_change.abs().max())})
            # 跨数值环境不强求逐位一致；将差异单独报告，收益比较使用本轮H-only。
            check(f"新旧H-only测试位置一致可配对/{label}", paired_layout)
        meta_rows.append({key: meta[key] for key in RUN_KEYS + ["variant", "n_parameters", "hidden", "epochs_refit"]})
        if number % 54 == 0:
            print(f"已核验 {number}/{len(files)} 个转换器预测文件", flush=True)

    check("324组完整且无重复", len(actual_keys) == 324 and set(actual_keys) == expected_keys
          and len(set(actual_keys)) == len(actual_keys))
    for scaling, tag, seed in itertools.product(spec["scalings"], scenes, spec["model_seeds"]):
        full = full_metadata[(scaling, tag, seed, "h_survey_geometry")]
        capacity = full_metadata[(scaling, tag, seed, "h_only_capacity_control")]
        shuffled = full_metadata[(scaling, tag, seed, "h_survey_geometry_shuffled")]
        check(f"容量匹配/{scaling}/{tag}/{seed}", 0 <= capacity["n_parameters"] - full["n_parameters"] <= 2)
        check(f"打乱组同网络容量/{scaling}/{tag}/{seed}", shuffled["hidden"] == full["hidden"]
              and shuffled["n_parameters"] == full["n_parameters"])

    scores = pd.concat(frames, ignore_index=True)
    stored_scores = pd.read_csv(SOURCE / "all_basin_metrics.csv", dtype={"basin": str})
    row_keys = RUN_KEYS + ["variant", "h_input_source", "basin"]
    compare_frames("逐文件指标与总表一致", scores, stored_scores, row_keys)
    group_keys = RUN_KEYS + ["variant", "h_input_source"]
    rows = []
    for values, frame in scores.groupby(group_keys):
        common = dict(zip(group_keys, values))
        for scope, subset in {"all_86": frame, "q_heldout": frame[frame.held_out],
                              "q_heldout_with_geometry": frame[frame.held_out & frame.geometry_available]}.items():
            rows.append({**common, "scope": scope, "n_basins": len(subset),
                "n_valid_nse": int(subset.nse.count()), "nse_mean": float(subset.nse.mean()),
                "nse_median": float(subset.nse.median()), "negative_nse_count": int((subset.nse < 0).sum())})
    by_run = pd.DataFrame(rows)
    compare_frames("独立重算每轮汇总", by_run, pd.read_csv(SOURCE / "summary_by_run.csv"), group_keys + ["scope"])
    predicted = by_run[(by_run.scope == "q_heldout") & (by_run.h_input_source == "predicted_h")]
    summary = predicted.groupby(["target_scaling", "variant", "ratio"])[
        ["nse_mean", "nse_median", "negative_nse_count"]].mean().reset_index()
    compare_frames("独立重算主汇总", summary, pd.read_csv(SOURCE / "heldout_summary.csv"),
                   ["target_scaling", "variant", "ratio"])
    compare_frames("拟合审计汇总一致", pd.DataFrame(meta_rows),
                   pd.read_csv(SOURCE / "fit_audit.csv")[pd.DataFrame(meta_rows).columns], RUN_KEYS + ["variant"])
    check("完成标记324组", read_json(SOURCE / "completed.json")["n_completed"] == 324
          and read_json(SOURCE / "summary_manifest.json")["n_completed"] == 324)

    # 同一隐藏站、同一掩膜、同一模型种子配对，不把重复站点当作独立研究样本。
    hidden = scores[scores.held_out].copy()
    comparisons, pairs = [], []
    pairing = RUN_KEYS + ["h_input_source", "basin"]
    contrasts = [(variant, "h_only") for variant in VARIANTS if variant != "h_only"]
    contrasts += [("h_survey_geometry", reference) for reference in
                  ("h_geometry_scale", "h_survey_stage", "h_only_capacity_control", "h_survey_geometry_shuffled")]
    for variant, reference in contrasts:
        matched = hidden[hidden.variant == variant].merge(
            hidden[hidden.variant == reference][pairing + ["nse"]], on=pairing,
            suffixes=("", "_reference"), validate="one_to_one")
        matched["delta_nse"] = matched.nse - matched.nse_reference
        matched["reference"] = reference
        pairs.append(matched[pairing + ["variant", "reference", "geometry_available", "nse", "nse_reference", "delta_nse"]])
        for keys, frame in matched.groupby(["target_scaling", "ratio", "h_input_source"]):
            for scope, part in {"q_heldout": frame, "q_heldout_with_geometry": frame[frame.geometry_available]}.items():
                run_delta = part.groupby(["mask_seed", "model_seed"]).delta_nse.mean()
                station_delta = part.groupby("basin").delta_nse.mean()
                comparisons.append({**dict(zip(["target_scaling", "ratio", "h_input_source"], keys)),
                    "variant": variant, "reference": reference, "scope": scope,
                    "n_runs": len(run_delta), "n_station_occurrences": len(part), "n_unique_basins": len(station_delta),
                    "mean_run_delta_nse": run_delta.mean(), "median_paired_delta_nse": part.delta_nse.median(),
                    "positive_runs": int((run_delta > 0).sum()), "run_delta_min": run_delta.min(),
                    "run_delta_max": run_delta.max(), "station_occurrence_win_fraction": (part.delta_nse > 0).mean(),
                    "unique_station_win_fraction": (station_delta > 0).mean()})
    paired = pd.concat(pairs, ignore_index=True)
    differences = pd.DataFrame(comparisons)
    station = paired[(paired.variant == "h_survey_geometry") & (paired.reference == "h_only")
                     & (paired.h_input_source == "predicted_h")].groupby(
        ["target_scaling", "ratio", "basin", "geometry_available"])[["nse", "nse_reference", "delta_nse"]].mean().reset_index()
    station = station.sort_values(["target_scaling", "ratio", "delta_nse"])

    by_run.to_csv(OUTPUT / "verified_summary_by_run.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "verified_heldout_summary.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(OUTPUT / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUTPUT / "station_full_vs_h_only.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recomputed).to_csv(OUTPUT / "prediction_metric_recalculation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(old_baseline).to_csv(OUTPUT / "h_only_reproduction.csv", index=False, encoding="utf-8-sig")
    failed = [item for item in checks if not item["passed"]]
    integrity = {"n_checks": len(checks), "n_passed": len(checks) - len(failed), "n_failed": len(failed),
        "n_converter_fits": len(files), "n_predictions_per_model": len(expected_pos),
        "n_recomputed_basin_metrics": 86 * len(files), "n_metric_rows_including_diagnostics": len(scores),
        "n_h_only_models_bitwise_reproduced": sum(item["bitwise_equal"] for item in old_baseline),
        "maximum_absolute_h_only_heldout_mean_nse_change": max(abs(item["heldout_mean_nse_change"]) for item in old_baseline),
        "maximum_absolute_h_only_station_nse_change": max(item["maximum_absolute_station_nse_change"] for item in old_baseline),
        "failures": failed, "checks": checks}
    (OUTPUT / "integrity_summary.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "source_fingerprints.json").write_text(json.dumps({
        name: sha256(SOURCE / name) for name in ("experiment.json", "completed.json", "summary_manifest.json",
            "all_basin_metrics.csv", "summary_by_run.csv", "heldout_summary.csv", "survey_geometry.csv")}, indent=2), encoding="utf-8")

    lines = ["# 原86流域测量几何重拟合结果复核", "",
        "本轮仅重拟合阶段1的H→Q转换器。冻结原强配置水位预测，未重训水位模型或阶段2网络。",
        "使用每站一份事后剖面快照的面积、已测湿润长度、宽度和测量时水位，未构建动态A(H)、P(H)。", "",
        f"完整性：{len(files)}/324组；每组{len(expected_pos):,}个测试预测；独立重算{86 * len(files):,}条逐站NSE/RMSE/MAE。",
        f"核验共{len(checks)}项，通过{len(checks) - len(failed)}项，失败{len(failed)}项。",
        f"本机H-only的54组测试位置与旧实验一致；{sum(item['bitwise_equal'] for item in old_baseline)}/54组预测与旧实验逐值完全相同。",
        f"新旧H-only的隐藏站平均NSE最大绝对差为{integrity['maximum_absolute_h_only_heldout_mean_nse_change']:.8f}，"
        f"单站NSE最大绝对差为{integrity['maximum_absolute_h_only_station_nse_change']:.8f}。",
        "基线重新拟合存在数值差异，新增几何的效果统一相对本轮同机H-only评价，不以新旧实验的逐值一致为前提。", "",
        "## 评价口径", "",
        "主结果仅使用预测H和Q隐藏站。每个比例含3个掩膜种子×3个模型种子；先对每轮隐藏站等权求均值，再对9轮求均值。",
        "physical在Q隐藏站不用其训练Q统计；observed使用隐藏站训练Q均值/标准差，属于已知Q尺度对照。",
        "均值、中位数和负NSE数量均按每轮计算后再取9轮平均。负NSE数量的小数表示每轮平均站数。",
        "同站在不同掩膜/种子重复出现，配对胜率是描述性统计，不把9轮当独立样本做显著性结论。", ""]
    for scaling in ("physical", "observed"):
        lines += [f"## {scaling}：隐藏站平均NSE", ""]
        table = summary[summary.target_scaling == scaling].pivot(index="variant", columns="ratio", values="nse_mean").reindex(VARIANTS)
        table.index = [LABELS[name] for name in table.index]
        table.columns = [f"隐藏{round(ratio * 100)}%" for ratio in table.columns]
        lines += [markdown(table.map(lambda value: f"{value:.4f}").rename_axis("模型").reset_index()), ""]
        robust = summary[(summary.target_scaling == scaling) & summary.variant.isin(["h_only", "h_survey_geometry"])].copy()
        robust["variant"] = robust.variant.map(LABELS)
        robust["ratio"] = robust.ratio.map(lambda value: f"{value:.0%}")
        for column in ("nse_mean", "nse_median", "negative_nse_count"):
            robust[column] = robust[column].map(lambda value: f"{value:.4f}")
        robust.columns = ["归一化", "模型", "Q隐藏比例", "平均NSE", "站点中位NSE均值", "平均负NSE站数"]
        lines += ["### 均值以外的稳定性", "", markdown(robust.drop(columns="归一化")), ""]
    main_diff = differences[(differences.variant == "h_survey_geometry") & (differences.reference == "h_only")
        & (differences.h_input_source == "predicted_h") & (differences.scope == "q_heldout")].copy()
    lines += ["## 完整几何相对H-only的配对结果", ""]
    display = main_diff[["target_scaling", "ratio", "mean_run_delta_nse", "positive_runs", "station_occurrence_win_fraction"]].copy()
    display["ratio"] = display.ratio.map(lambda value: f"{value:.0%}")
    display["mean_run_delta_nse"] = display.mean_run_delta_nse.map(lambda value: f"{value:+.4f}")
    display["positive_runs"] = display.positive_runs.map(lambda value: f"{value}/9")
    display["station_occurrence_win_fraction"] = display.station_occurrence_win_fraction.map(lambda value: f"{value:.1%}")
    display.columns = ["归一化", "Q隐藏比例", "平均NSE差", "提升轮数", "逐站配对提升比例"]
    lines += [markdown(display), "", "## 最明显下降的站点", "",
              "以下先对同站在各次留出评价中的结果求均值；不同站出现次数可能不同，不据此排除站点或重选模型。", ""]
    worst = station[(station.target_scaling == "physical") & (station.ratio == .7)].head(8).copy()
    for column in ("nse", "nse_reference", "delta_nse"):
        worst[column] = worst[column].map(lambda value: f"{value:.4f}")
    worst = worst[["basin", "geometry_available", "nse", "nse_reference", "delta_nse"]]
    worst.columns = ["站点", "有合格断面", "完整几何NSE", "H-only NSE", "差值"]
    lines += [markdown(worst), "", "## 结果解释与边界", "",
        "新增的静态测量几何没有带来跨比例、跨归一化的稳定收益。不能据此宣称断面信息已经改善H→Q转换。",
        "该结论仅针对本次输入表示、转换器和划分，不能据此推断动态过水面积A(H)没有价值。",
        "打乱对照与参数匹配对照用于区分几何对应关系和网络容量；全部结果保留，未按测试结果挑选变体。",
        "测量时水位是额外事后观测；81站可用剖面中79份晚于训练结束，因此本轮属于已明确授权的固定断面回溯实验。",
        "训练转换器使用实测H，部署测试使用预测H，仍存在输入误差分布差异。实测测试H诊断不等同可部署性能。",
        "本轮设计是在查看上一轮测试结果后提出，属于探索性改进，没有新增独立盲测证据。",
        "历史面积与本地水位配对的动态关系尚未加入本轮；后续首先需要核查测量位置、主河道及时间一致性。", "",
        "## 输出与复现", "", "复核命令（不训练）：", "",
        "```powershell", "python -X utf8 tools/review_cross_section86_survey_refit.py", "```", "",
        "`paired_comparisons.csv`含全部预定输入、两类H输入和有无合格几何子集的配对比较。",
        "`station_full_vs_h_only.csv`保存逐站差异。`integrity_summary.json`保存逐项核验。", ""]
    (OUTPUT / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in integrity.items() if key != "checks"}, ensure_ascii=False, indent=2))
    print("\n主配对结果：\n" + main_diff.to_string(index=False))
    print("\n诊断H均值：\n" + by_run[(by_run.scope == "q_heldout") & (by_run.h_input_source == "observed_h_diagnostic")
        & by_run.variant.isin(["h_only", "h_survey_geometry"])].groupby(["target_scaling", "variant", "ratio"]).nse_mean.mean().to_string())
    if failed:
        raise SystemExit("存在未通过的核验，请先阅读integrity_summary.json。")


if __name__ == "__main__":
    main()
