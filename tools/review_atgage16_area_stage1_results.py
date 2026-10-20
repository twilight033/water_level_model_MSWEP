"""审核已完成的旧方案面积阶段1结果；只读模型和预测，不拟合或运行新实验。"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/atgage16_area_stage1_v1"
OUTPUT = ROOT / "reports/atgage16_area_stage1_review"
BASELINE = "h_attrs_area_available"
AREA = "h_attrs_area"
KEYS = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def audit():
    """检查全部324个完成项并从保存预测独立重算27864个逐站指标。"""
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from pipeline.dataset import PreparedData

    spec = load(SOURCE / "experiment.json")
    expected_id = spec["experiment_fingerprint"]
    assert fingerprint({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == expected_id
    for name in ("completed.json", "process.json", "summary_manifest.json"):
        meta = load(SOURCE / name)
        assert meta["experiment_fingerprint"] == expected_id and meta["n_completed"] == 324
    source_checks, line_ending_only = [], []
    for path, expected in spec["code_hashes"].items():
        if digest(ROOT / path) != expected:
            raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
            assert expected in [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b"\n", b"\r\n"))], path
            line_ending_only.append(path)
        source_checks.append(path)
    for scaling, data in spec["sources"].items():
        for seed, item in data["H_predictions"].items():
            folder = ROOT / "results" / f"cross_section86_strong_{scaling}_snapshot_v2/waterlevel_models/seed{seed}"
            assert digest(folder / "prediction.npz") == item["prediction_sha256"]
            assert digest(folder / "model.pt") == item["model_sha256"]
    area_payload = load(SOURCE / "area_curves.json")
    assert area_payload["payload_fingerprint"] == spec["area_payload_fingerprint"]
    assert fingerprint({k: v for k, v in area_payload.items() if k != "payload_fingerprint"}) == area_payload["payload_fingerprint"]
    print("检查来源与冻结水位预测通过；载入原缓存，仅核验已有预测……", flush=True)
    prepared = PreparedData(attr_set="extended")
    assert prepared.basins == spec["basins"] and len(prepared.basins) == 86
    for path, expected in spec["data_hashes"].items():
        actual = digest(ROOT / path)
        if actual != expected:
            raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
            assert Path(path).suffix == ".txt" and expected in [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b"\n", b"\r\n"))]
    metrics = pd.read_csv(SOURCE / "all_basin_metrics.csv", dtype={"basin": str})
    assert len(metrics) == 324 * 86 * 2 and not metrics.duplicated(KEYS + ["variant"]).any()
    scenes = {s["tag"]: s for s in spec["scenarios"]}
    expected_runs = set(itertools.product(("physical", "observed"), scenes, (1, 2, 3), spec["variants"]))
    done_files = sorted(SOURCE.glob("*/stage1/*/*/*/done.json"))
    assert len(done_files) == len(expected_runs) == 324
    checked, maximum_error, seen, audits, common_ids, common_pos = 0, 0., set(), [], None, None
    for i, done in enumerate(done_files):
        scaling, _, tag, seed_folder, variant = done.relative_to(SOURCE).parts[:-1]
        seed = int(seed_folder.removeprefix("seed"))
        run = (scaling, tag, seed, variant)
        assert run in expected_runs and run not in seen
        seen.add(run)
        scene, meta = scenes[tag], load(done)
        expected_run = fingerprint({"experiment": expected_id, "scaling": scaling, "scenario": tag, "seed": seed, "variant": variant})
        assert meta["run_fingerprint"] == expected_run
        assert meta["training_q_pair_basins"] == scene["retained"]
        assert meta["heldout"] == scene["heldout"]
        train, valid = set(meta["inner_train_basins"]), set(meta["inner_validation_basins"])
        assert train.isdisjoint(valid) and train | valid == set(scene["retained"])
        assert set(scene["heldout"]).isdisjoint(train | valid)
        assert set(meta["observed_q_scale_basins"]) == (set(prepared.basins) if scaling == "observed" else set())
        assert meta["scale_fit_basins"] == (scene["retained"] if scaling == "physical" else [])
        for field, permitted in (("inner_area_scaler", train), ("final_area_scaler", set(scene["retained"]))):
            if meta[field] is not None:
                assert set(meta[field]["fit_basins"]) == permitted & set(spec["area_basins"])
        checkpoint = torch.load(done.parent / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["run_fingerprint"] == expected_run
        parameters = sum(v.numel() for v in checkpoint["state_dict"].values())
        assert parameters == meta["n_parameters"] == meta["hidden"] * (meta["input_dim"] + 2) + 1
        with np.load(done.parent / "prediction.npz") as file:
            ids, pos, predicted = file["basin_idx"], file["target_pos"], file["pred_flow"]
        assert np.isfinite(predicted).all()
        if common_ids is None:
            common_ids, common_pos = ids.copy(), pos.copy()
            slices = {bi: np.flatnonzero(ids == bi) for bi in range(86)}
            observed = prepared.targets_raw["flow"][ids, pos].astype(float)
        assert np.array_equal(ids, common_ids) and np.array_equal(pos, common_pos)
        raw_scores = pd.read_csv(done.parent / "metrics.csv", dtype={"basin": str}).set_index("basin")
        assert set(raw_scores.index) == set(prepared.basins)
        for bi, basin in enumerate(prepared.basins):
            selected = slices[bi]
            y, estimate = observed[selected], predicted[selected].astype(float)
            ok = np.isfinite(y) & np.isfinite(estimate)
            y, estimate = y[ok], estimate[ok]
            row = raw_scores.loc[basin]
            assert len(y) == row.n
            variance = np.square(y - y.mean()).sum()
            if len(y) >= 2 and variance > 0:
                nse = 1 - np.square(estimate - y).sum() / variance
                rmse = np.sqrt(np.square(estimate - y).mean())
                mae = np.abs(estimate - y).mean()
                for actual, stored in ((nse, row.nse), (rmse, row.rmse), (mae, row.mae)):
                    assert np.isclose(actual, stored, rtol=2e-6, atol=2e-6), (run, basin, actual, stored)
                    maximum_error = max(maximum_error, abs(actual - stored))
            checked += 1
        for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
            original = pd.read_csv(done.parent / filename, dtype={"basin": str}).set_index("basin")
            combined = metrics[metrics.target_scaling.eq(scaling) & metrics.ratio.eq(scene["ratio"])
                               & metrics.mask_seed.eq(scene["mask_seed"]) & metrics.model_seed.eq(seed)
                               & metrics.variant.eq(variant) & metrics.h_input_source.eq(source)].set_index("basin")
            assert set(combined.index) == set(prepared.basins)
            assert np.allclose(original.loc[prepared.basins, "nse"], combined.loc[prepared.basins, "nse"], equal_nan=True)
        audits.append({"scaling": scaling, "scenario": tag, "seed": seed, "variant": variant,
                       "n_parameters": parameters, "n_area_train": meta["n_area_train_basins"],
                       "n_area_inner_valid": meta["n_area_inner_validation_basins"]})
        if (i + 1) % 36 == 0:
            print(f"已核验{i + 1}/324个转换器及对应测试预测", flush=True)
    assert seen == expected_runs
    pd.DataFrame(audits).to_csv(OUTPUT / "model_audit.csv", index=False, encoding="utf-8-sig")
    return spec, metrics, prepared, {"status": "PASS", "n_completed_models": 324,
        "independently_recomputed_station_metrics": checked, "maximum_absolute_metric_error": maximum_error,
        "n_original_h_predictions_verified": 6, "n_source_code_files_verified": len(source_checks),
        "code_hash_differences_only_lf_crlf": line_ending_only,
        "old_protocol_all_retained_q_basins_fit": True, "hidden_q_pair_and_internal_split_checks_pass": True,
        "observed_h_diagnostic_csv_matches_summary": True, "no_training_or_new_inference_performed": True}


def analyse(spec, data):
    predicted = data[data.h_input_source.eq("predicted_h") & data.held_out & data.geometry_available]
    wide = predicted.pivot(index=KEYS, columns="variant", values="nse")
    wide["delta_area"] = wide[AREA] - wide[BASELINE]
    wide.reset_index().to_csv(OUTPUT / "paired_hidden_area_station_runs.csv", index=False, encoding="utf-8-sig")
    run_keys = KEYS[:-1]
    runs = wide.groupby(level=run_keys).mean()
    rows = []
    for keys, frame in runs.groupby(level=["target_scaling", "ratio"]):
        source = predicted[predicted.target_scaling.eq(keys[0]) & predicted.ratio.eq(keys[1])]
        medians = source.groupby(["mask_seed", "model_seed", "variant"]).nse.median().groupby("variant").mean()
        negative = source.groupby(["mask_seed", "model_seed", "variant"]).nse.agg(lambda x: int((x < 0).sum())).groupby("variant").mean()
        row = {"target_scaling": keys[0], "ratio": keys[1], "n_runs": len(frame),
               "n_unique_hidden_area_stations": source.basin.nunique(),
               "mean_delta": frame.delta_area.mean(), "n_runs_improved": int(frame.delta_area.gt(0).sum()),
               "delta_min": frame.delta_area.min(), "delta_max": frame.delta_area.max()}
        for variant in spec["variants"]:
            row[variant + "_mean"] = frame[variant].mean()
            row[variant + "_median"] = medians[variant]
            row[variant + "_negative_count"] = negative[variant]
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUTPUT / "main_comparison.csv", index=False, encoding="utf-8-sig")
    station = wide.groupby(level=["target_scaling", "ratio", "basin"]).mean().reset_index()
    station["analysis_group"] = station.basin.map(lambda b: "focus5" if b in spec["focus_basins"] else "qualified11")
    station.to_csv(OUTPUT / "station_comparison.csv", index=False, encoding="utf-8-sig")
    station[station.analysis_group.eq("focus5")].to_csv(OUTPUT / "focus5_comparison.csv", index=False, encoding="utf-8-sig")
    balanced = station.groupby(["target_scaling", "ratio"]).agg(
        n_stations=("basin", "size"), baseline_mean=(BASELINE, "mean"), area_mean=(AREA, "mean"),
        delta_mean=("delta_area", "mean"), delta_median=("delta_area", "median"),
        n_stations_improved=("delta_area", lambda x: int((x > 0).sum())))
    balanced.to_csv(OUTPUT / "station_balanced_sensitivity.csv", encoding="utf-8-sig")
    combined = data[data.held_out & data.geometry_available]
    diagnostic = combined.groupby(["target_scaling", "ratio", "h_input_source", "mask_seed", "model_seed", "variant"]).nse.mean().groupby(
        ["target_scaling", "ratio", "h_input_source", "variant"]).mean().reset_index()
    diagnostic.to_csv(OUTPUT / "observed_h_diagnostic.csv", index=False, encoding="utf-8-sig")
    support = predicted[predicted.variant.eq(AREA)].groupby("basin")[["area_h_outside_fraction", "area_negative_clipped_fraction",
        "geometry_lovo_r2", "geometry_time_r2", "geometry_lovo_relative_mae"]].mean()
    support.to_csv(OUTPUT / "area_support_and_geometry.csv", encoding="utf-8-sig")
    return aggregate, station, runs, diagnostic, support


def feature_support(prepared):
    """检查最差运行的面积数值是否超出已训练面积站范围，不重新拟合转换器。"""
    spec, payload = load(SOURCE / "experiment.json"), load(SOURCE / "area_curves.json")
    folder = SOURCE / "physical/stage1/q70_mask42/seed2/h_attrs_area"
    metadata = load(folder / "done.json")
    scaler = metadata["final_area_scaler"]

    def area(basin, normalized):
        bi, curve = prepared.basin_index[basin], payload["curves"][basin]
        h = normalized * prepared.target_std["waterlevel"][bi] + prepared.target_mean["waterlevel"][bi]
        return np.maximum(curve["intercept_m2"] + curve["slope_m"] * h, 0)

    training_area = []
    for basin in scaler["fit_basins"]:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, "train")
        h, q = prepared.targets["waterlevel"][bi, lo:hi + 1], prepared.targets_raw["flow"][bi, lo:hi + 1]
        h = h[np.isfinite(h) & np.isfinite(q)]
        if len(h) > 4096:
            h = h[np.unique(np.linspace(0, len(h) - 1, 4096).astype(int))]
        training_area.extend(area(basin, h).tolist())
    lo, hi = min(training_area), max(training_area)
    source = ROOT / "results/cross_section86_strong_physical_snapshot_v2/waterlevel_models/seed2/prediction.npz"
    rows = []
    with np.load(source) as prediction:
        ids, h = prediction["basin_idx"], prediction["pred_h_normalized"]
        for basin in spec["area_basins"]:
            value = area(basin, h[ids == prepared.basin_index[basin]])
            rows.append({"basin": basin, "training_area_min_m2": lo, "training_area_max_m2": hi,
                         "test_area_median_m2": np.median(value), "test_area_p95_m2": np.quantile(value, .95),
                         "fraction_outside_training_area_value_range": ((value < lo) | (value > hi)).mean()})
    pd.DataFrame(rows).to_csv(OUTPUT / "worst_run_area_value_extrapolation.csv", index=False, encoding="utf-8-sig")


def write_report(spec, aggregate, station, runs, diagnostic, support, integrity):
    lines = ["# 已完成的旧方案：16站面积阶段1结果分析", "",
        "已完成324/324个转换器。该结果属于旧方案：原86中全部Q保留站共同拟合，只有16站提供面积；不是新确认的16站专用训练方案。原H模型未重训，阶段2未运行。", "",
        "用户已确认后续新方案：转换器只在16站中的Q保留站训练，在同16站中的Q隐藏站测试；有无面积对照同训练站，原86站H模型保持冻结。本报告只分析已有结果，不修改方案或重跑。", "",
        "## 主比较", "",
        "比较对象为同样使用H、扩展属性和面积可用标记的两组，区别是是否提供面积数值。统计仅限16站中当次Q隐藏站。每个比例9次运行：先在每次运行内按站等权平均，再对9次运行等权平均。30/50/70%比例的隐藏面积站并集分别为13/15/16站。", "",
        "| 归一化 | Q隐藏比例 | 不加面积平均NSE | 加面积平均NSE | 差值 | 9次中改善次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {getattr(row, BASELINE + '_mean'):.4f} | {getattr(row, AREA + '_mean'):.4f} | {row.mean_delta:+.4f} | {row.n_runs_improved}/9 |")
    lines.extend(["", "physical是隐藏站Q尺度由Q保留站推断的主口径；observed额外使用隐藏站训练Q均值/标准差，是已知Q尺度对照。", "",
        "面积没有表现出稳定收益：physical只在30%条件下有约0.01的运行平均改善，50%和70%下降；observed三个比例均下降。", "",
        "## 其他对照", "",
        "下表与主表使用同一隐藏面积站范围和9次运行平均口径。真实面积没有稳定优于打乱面积或容量上界对照。", "",
        "| 归一化 | 隐藏比例 | 只用H | H+属性 | 加面积 | 打乱面积 | 容量对照 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in aggregate.itertuples(index=False):
        values = [getattr(row, v + "_mean") for v in ("h_only", "h_attrs", AREA, "h_attrs_area_shuffled", "h_attrs_capacity_control")]
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend(["",
        "## 中位数与负NSE站数", "",
        "下表先在每次运行的隐藏面积站内统计，再对9次运行平均，因此负NSE站数可能为小数。observed70%的中位数上升、负NSE站数下降，但均值和运行间表现不稳定，不能将结果概括为所有站一致变差。", "",
        "| 归一化 | 隐藏比例 | 不加/加面积的中位NSE | 不加/加面积的负NSE站数 |",
        "| --- | ---: | --- | --- |"])
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {getattr(row, BASELINE + '_median'):.4f} / {getattr(row, AREA + '_median'):.4f} | {getattr(row, BASELINE + '_negative_count'):.2f} / {getattr(row, AREA + '_negative_count'):.2f} |")
    lines.extend(["",
        "## 稳健性与极端失败", "",
        "每次运行的隐藏站数不同，故另提供先按流域汇总再等权的敏感性结果。不能把这两种平均口径混用。", "",
        "| 归一化 | 隐藏比例 | 流域等权差值均值 | 差值中位数 | 改善站数 |",
        "| --- | ---: | ---: | ---: | ---: |"])
    for keys, group in station.groupby(["target_scaling", "ratio"]):
        lines.append(f"| {keys[0]} | {keys[1]:.0%} | {group.delta_area.mean():+.4f} | {group.delta_area.median():+.4f} | {int(group.delta_area.gt(0).sum())}/{len(group)} |")
    worst = runs.loc[("physical", .7)]
    excluded = ((worst.index.get_level_values("mask_seed") == 42)
                & (worst.index.get_level_values("model_seed") == 2))
    remaining = worst[~excluded]
    lines.extend(["", "physical的70%均值受mask42、seed2这一次严重失败影响很大：当次不加面积0.5143，加面积-2.6496。原结果必须保留。",
        f"仅作敏感性诊断，排除这一次后，其余8次不加面积{remaining[BASELINE].mean():.4f}，加面积{remaining[AREA].mean():.4f}，差值{remaining.delta_area.mean():+.4f}，仍没有稳定改善。",
        "该次01059000的NSE从0.7104降到-13.1439，01092000从0.4382降到-8.5338，01030500从0.6309降到-3.9370。包含面积曲线验证很好的站，不能把全部失败归于5个重点关注站。", "",
        "## 11站与5站分析", "",
        "下表为70%隐藏Q时，同一流域适用运行先平均后的NSE；不是9次运行的组均值。5站全部保留。", "",
        "| 归一化 | 重点关注站 | 不加面积NSE | 加面积NSE | 差值 |",
        "| --- | --- | ---: | ---: | ---: |"])
    focus = station[station.ratio.eq(.7) & station.analysis_group.eq("focus5")]
    for row in focus.itertuples(index=False):
        lines.append(f"| {row.target_scaling} | {row.basin} | {getattr(row, BASELINE):.4f} | {getattr(row, AREA):.4f} | {row.delta_area:+.4f} |")
    value_support = pd.read_csv(OUTPUT / "worst_run_area_value_extrapolation.csv", dtype={"basin": str})
    example = value_support[value_support.basin.eq("01059000")].iloc[0]
    lines.extend(["", "physical的70%条件下，11站达标组的运行平均NSE也由0.3290降为-0.0645；重点关注5站由0.4013降为0.0652。问题并非只发生在面积曲线较弱的5站。", "",
        "## 失败原因的证据与边界", "",
        "- 面积训练站少是已确认事实：70%时仅3/5/6站；内部选模时可能只有1个面积验证站。其他无面积站只能帮助H/属性基础关系，不能增加面积监督的独立流域数。",
        "- 部分运行有严重跨流域泛化失败和负流量输出；无非负约束的MLP对新面积尺度的外推可能不稳定。最差运行的训练面积数值范围及各站超出比例保存在worst_run_area_value_extrapolation.csv。这是诊断证据，不是已识别的唯一因果机制。",
        f"- 具体例子：最差运行的转换器训练输入面积最大约{example.training_area_max_m2:.1f}m²；01059000测试时传入网络的估算面积中位数约{example.test_area_median_m2:.1f}m²，{example.fraction_outside_training_area_value_range:.0%}超出训练站面积数值范围。这是跨流域面积尺度外推，与该站水位是否超出自己A(H)曲线范围是两件事。",
        "- 用实测测试H诊断时，physical70%的不加面积/加面积运行平均NSE为0.6901/0.2730；最差运行中01059000即使换实测H仍为-12.7219。水位预测误差不能解释全部失败。",
        "- observed70%下，H-only的预测H平均NSE为0.6171，而H+属性的版本为0.0621，加入面积后为0.0180。问题也存在于高维属性共享转换器本身，不能把全部下降归因于面积曲线。",
        "- 已测H支持范围外的面积是外推估计。01072800约31.8%预测时刻超界，01085500约33.2%；010735562约9.7%、01100505约7.1%时刻出现负线性面积后被截成0。缺少连续面积真值，不能把这些诊断当面积预测精度。", "",
        "结论仅限旧的混合训练方案：当前没有稳定证据表明面积改善H→Q转换。不能据此断言面积本身无价值，也不能预先保证16站专用方案一定改善。后续新方案仍需相同训练站的公平对照，并保留所有16站及5站分组。", "",
        "## 核验", "",
        f"全部324项完成文件、原始模型参数量、Q配对拟合名单及内部隔离均核验。独立从保存预测重算{integrity['independently_recomputed_station_metrics']:,}个站次的NSE/RMSE/MAE，最大数值差{integrity['maximum_absolute_metric_error']:.3g}。诊断H的CSV与汇总逐项一致，但未重新运行神经网络前向。没有训练或新增推理。", "",
        "主要派生表：main_comparison.csv、station_comparison.csv、focus5_comparison.csv、observed_h_diagnostic.csv、area_support_and_geometry.csv；原结果未修改。"])
    (OUTPUT / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = {name: digest(SOURCE / name) for name in ("all_basin_metrics.csv", "experiment.json", "group_summary.csv")}
    spec, data, prepared, integrity = audit()
    aggregate, station, runs, diagnostic, support = analyse(spec, data)
    feature_support(prepared)
    for name, expected in before.items():
        assert digest(SOURCE / name) == expected
    integrity["source_result_files_unchanged"] = True
    (OUTPUT / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(spec, aggregate, station, runs, diagnostic, support, integrity)
    print(json.dumps(integrity, ensure_ascii=False), flush=True)
    print(f"分析完成：{OUTPUT / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
