"""比较Q保留站加入面积前后的已保存模型效果；仅评价，不重新拟合。"""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_stage1 import AREA_BASINS, load_json
from cross_section_experiment.area_stage1_only16 import DEFAULT_OUTPUT as SOURCE
from cross_section_experiment.converters import collect_pairs
from cross_section_experiment.protocol import write_json
from pipeline.dataset import PreparedData


OUTPUT = ROOT / "reports/atgage16_retained_area_comparison"
BASE, AREA = "h_attrs_area_available", "h_attrs_area"


def evaluate_retained_pairs():
    """从保存权重评价每个训练站，配对来源与原拟合/前次诊断保持相同。"""
    helper = runpy.run_path(str(ROOT / "tools/review_atgage16_only16_neural.py"))
    prepared = PreparedData(attr_set="extended")
    spec = load_json(SOURCE / "experiment.json")
    curves = load_json(SOURCE / "area_curves.json")["curves"]
    rows = []
    paths = sorted(p for p in SOURCE.glob("*/stage1/*/*/*/done.json") if p.parent.name in (BASE, AREA))
    assert len(paths) == 108
    for count, path in enumerate(paths, 1):
        meta = load_json(path)
        checkpoint = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["run_fingerprint"] == meta["run_fingerprint"]
        names = checkpoint["flow_scale_basins"]
        assert names == list(AREA_BASINS)
        ids = [prepared.basin_index[b] for b in names]
        scale = SimpleNamespace(mean={"flow": np.full(86, np.nan)}, std={"flow": np.full(86, np.nan)})
        scale.mean["flow"][ids], scale.std["flow"][ids] = checkpoint["flow_mean"], checkpoint["flow_std"]
        retained = meta["training_q_pair_basins"]
        assert set(retained).issubset(AREA_BASINS) and not set(retained) & set(meta["heldout"])
        for split in ("train", "valid"):
            pairs = collect_pairs(prepared, retained, split, scale, None,
                spec["source_settings"][meta["target_scaling"]]["converter_max_pairs_per_basin"])
            x = helper["features"](prepared, curves, pairs.basin, pairs.x[:, 0], meta["variant"], meta["final_area_scaler"])
            prediction_z = helper["neural_prediction"](checkpoint, x)
            assert np.isfinite(prediction_z).all()
            predicted = prediction_z * scale.std["flow"][pairs.basin] + scale.mean["flow"][pairs.basin]
            for basin in retained:
                bi = prepared.basin_index[basin]
                take = pairs.basin == bi
                y = prepared.targets_raw["flow"][bi, pairs.target_pos[take]].astype(float)
                pred = predicted[take].astype(float)
                assert len(y) >= 2 and np.isfinite(y).all()
                denominator = np.square(y - y.mean()).sum()
                rows.append({"target_scaling": meta["target_scaling"], "ratio": meta["ratio"],
                    "mask_seed": meta["mask_seed"], "model_seed": meta["model_seed"], "basin": basin,
                    "variant": meta["variant"], "period": split, "h_input_source": "observed_h",
                    "n": len(y), "nse": 1 - np.square(pred - y).sum() / denominator,
                    "rmse": np.sqrt(np.square(pred - y).mean()), "mae": np.abs(pred - y).mean(),
                    "n_negative_predictions": int((pred < 0).sum()), "hidden": meta["hidden"],
                    "epochs_refit": meta["epochs_refit"]})
        if count % 27 == 0:
            print(f"已评价{count}/108个保存模型的Q保留站，未重新训练", flush=True)
    frame = pd.DataFrame(rows)
    # 用既有逐运行诊断交叉核对，区别仅来自float32归一化与物理量纲重算误差。
    old = pd.read_csv(ROOT / "reports/atgage16_only16_neural_review/retained_fit_diagnostics.csv")
    keys = ["target_scaling", "ratio", "mask_seed", "model_seed", "variant", "period"]
    mean = frame.groupby(keys).nse.mean().reset_index()
    check = mean.merge(old.rename(columns={"split": "period"}), on=keys, validate="one_to_one")
    assert len(check) == 216
    np.testing.assert_allclose(check.nse, check.nn_final_model_nse, atol=2e-6, rtol=2e-6)
    return frame, float((check.nse - check.nn_final_model_nse).abs().max())


def summarize(frame):
    """先同站配对，再在每次运行内等权平均，最后汇总27次或每比例9次运行。"""
    keys = ["target_scaling", "ratio", "mask_seed", "model_seed", "period", "h_input_source", "basin"]
    baseline = frame[frame.variant.eq(BASE)]
    with_area = frame[frame.variant.eq(AREA)]
    paired = with_area.merge(baseline, on=keys, suffixes=("_area", "_no_area"), validate="one_to_one")
    assert (paired.n_area == paired.n_no_area).all()
    paired["delta_nse"] = paired.nse_area - paired.nse_no_area
    paired["delta_rmse"] = paired.rmse_area - paired.rmse_no_area
    paired.to_csv(OUTPUT / "paired_station_differences.csv", index=False, encoding="utf-8-sig")
    run_keys = keys[:-1]
    runs = paired.groupby(run_keys).agg(n_basins=("basin", "size"), no_area_nse=("nse_no_area", "mean"),
        area_nse=("nse_area", "mean"), delta_nse=("delta_nse", "mean"), median_station_delta=("delta_nse", "median"),
        improved_station_count=("delta_nse", lambda s: int((s > 0).sum())),
        no_area_negative_nse_count=("nse_no_area", lambda s: int((s < 0).sum())),
        area_negative_nse_count=("nse_area", lambda s: int((s < 0).sum()))).reset_index()
    runs.to_csv(OUTPUT / "paired_run_comparison.csv", index=False, encoding="utf-8-sig")
    for include_ratio, name in ((True, "comparison_by_ratio.csv"), (False, "overall_comparison.csv")):
        group_keys = ["target_scaling", "period", "h_input_source"] + (["ratio"] if include_ratio else [])
        rows = []
        for values, group in runs.groupby(group_keys):
            key = dict(zip(group_keys, values))
            selected = np.ones(len(paired), dtype=bool)
            for column, value in key.items():
                selected &= paired[column].eq(value)
            stations = paired[selected]
            rows.append({**key, "no_area_nse": group.no_area_nse.mean(), "area_nse": group.area_nse.mean(),
                "delta_nse": group.delta_nse.mean(), "n_paired_runs": len(group),
                "area_better_runs": int((group.delta_nse > 0).sum()), "median_run_delta": group.delta_nse.median(),
                "median_station_delta": stations.delta_nse.median(), "n_station_runs": len(stations),
                "area_better_station_runs": int((stations.delta_nse > 0).sum()),
                "no_area_negative_station_runs": int((stations.nse_no_area < 0).sum()),
                "area_negative_station_runs": int((stations.nse_area < 0).sum())})
        result = pd.DataFrame(rows)
        result.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")
    return paired, runs, pd.read_csv(OUTPUT / "overall_comparison.csv"), pd.read_csv(OUTPUT / "comparison_by_ratio.csv")


def write_report(overall, by_ratio, maximum_error):
    lines = ["# Q保留站：加入面积前后的神经网络对比", "",
        "只评价每个冻结情景中提供过训练Q的站，不混入Q隐藏站。比较已保存的两个最终网络：无面积为H+27个属性+恒定可用标记，有面积再增加估算面积。"
        "每个方案沿用各自在内部验证中选定的宽度和轮数，因此不是固定同一组权重后开关面积。未重新训练模型。", "",
        "训练期和valid期输入实测H；每站使用与原诊断相同的最多4096个有效配对。测试期另分实测H诊断和冻结预测H，使用原测试预测时刻上的全部可评价样本。", "",
        "下面先逐站计算NSE，每次运行在Q保留站内等权平均，再平均27次运行（3比例×3掩膜×3模型种子）。", "",
        "| 归一化 | 时段/水位来源 | 无面积NSE | 有面积NSE | 差值 | 有面积更好的运行数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    labels = {("train", "observed_h"): "训练期/实测H", ("valid", "observed_h"): "后续valid期/实测H",
              ("test", "observed_h_diagnostic"): "测试期/实测H诊断", ("test", "predicted_h"): "测试期/预测H"}
    for scaling in ("physical", "observed"):
        for period, source in labels:
            row = overall[overall.target_scaling.eq(scaling) & overall.period.eq(period) & overall.h_input_source.eq(source)].iloc[0]
            lines.append(f"| {scaling} | {labels[(period, source)]} | {row.no_area_nse:.4f} | {row.area_nse:.4f} | {row.delta_nse:+.4f} | {int(row.area_better_runs)}/{int(row.n_paired_runs)} |")
    lines += ["", "平均而言，加入面积在这些熟悉流域的训练期和后续时段也变差。此前仅强调跨流域迁移不稳并不完整："
        "面积方案在已知站的拟合也弱于无面积方案。现有结果尚不能区分优化、选模差异、正则或输入共线性的具体影响，更不能直接认定面积没有物理价值。", "",
        "同站测试期从实测H换成预测H后成绩明显下降，说明实际水位预测误差也影响转换；不能把实测H下的高NSE当作部署成绩。", "",
        "## 分Q隐藏比例查看", "",
        "比例是原86的名义Q隐藏比例。比例越高，本轮可用于训练的站总体越少，实际为3—12站。下表仍只评价其中的Q保留站。", "",
        "| 归一化 | 比例 | 时段 | 无面积NSE | 有面积NSE | 差值 | 有面积更好次数 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    table = by_ratio[by_ratio.period.isin(("train", "valid"))].sort_values(["target_scaling", "ratio", "period"], ascending=[False, True, True])
    for row in table.itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.period} | {row.no_area_nse:.4f} | {row.area_nse:.4f} | {row.delta_nse:+.4f} | {row.area_better_runs}/{row.n_paired_runs} |")
    lines += ["", "observed、30%这一组虽然平均下降，但训练和valid期均有6/9次运行改善，平均值受到少数较差运行影响；不能概括成每组、每次、每个站都下降。", "",
        "## 解释与复核", "",
        "这里的valid是最终网络已用过该站train期Q之后的后续时段，不是内部整站未见Q验证；训练期本身也不能作为泛化证据。", "",
        "observed允许隐藏站训练Q尺度，属于已知尺度对照；physical只使用Q保留站拟合尺度。当前比较对象本来就提供过训练Q。", "",
        f"从保存权重重新评价的216个训练/valid逐运行平均，与既有诊断最大差{maximum_error:.3g}，复核通过。"
        "逐站配对、运行配对、中位差值和负NSE站次数均另存CSV，重复站次不作为互相独立样本。", "",
        "文件：paired_station_differences.csv、paired_run_comparison.csv、comparison_by_ratio.csv、overall_comparison.csv。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, maximum_error = evaluate_retained_pairs()
    test = pd.read_csv(SOURCE / "all_basin_metrics.csv", dtype={"basin": str})
    test = test[~test.held_out & test.variant.isin((BASE, AREA))].copy()
    test["period"] = "test"
    test = test[[c for c in frame.columns if c in test.columns]]
    all_rows = pd.concat([frame, test], ignore_index=True)
    all_rows.to_csv(OUTPUT / "retained_station_metrics.csv", index=False, encoding="utf-8-sig")
    paired, runs, overall, by_ratio = summarize(all_rows)
    assert len(paired) == 1680 and len(runs) == 216
    write_report(overall, by_ratio, maximum_error)
    write_json(OUTPUT / "validation.json", {"status": "PASS", "models_retrained": 0,
        "n_evaluated_saved_models": 108, "n_train_valid_run_metrics_crosschecked": 216,
        "max_difference_vs_previous_diagnostics": maximum_error, "n_paired_station_runs": len(paired),
        "training_and_valid_h_source": "observed_h", "test_h_sources_separate": True,
        "only_q_retained_basins_evaluated": True, "max_pairs_per_station_train_valid": 4096})
    print(overall.to_string(index=False))
    print(f"报告：{OUTPUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
