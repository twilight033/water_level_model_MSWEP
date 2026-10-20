"""从保存的独立留出预测重算湿周拟合指标，检查测量分组及输入来源。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/cross_section86_perimeter_h_curves"


def main():
    raw = pd.read_csv(OUTPUT / "profile_inventory.csv", dtype={"basin": str})
    table = pd.read_csv(OUTPUT / "validation_metrics.csv", dtype={"basin": str})
    predicted = pd.read_csv(OUTPUT / "heldout_predictions.csv.gz", dtype={"basin": str})
    raw = raw.set_index("record_id", verify_integrity=True)
    raw["measurement_time"] = pd.to_datetime(raw.measurement_time)
    keys = ["basin", "h_source", "scope", "scheme"]
    assert not predicted.duplicated(keys + ["record_id"]).any()
    checked = 0
    groups = dict(tuple(predicted.groupby(keys)))
    for row in table.itertuples(index=False):
        if row.scheme == "fit":
            continue
        group = groups[(row.basin, row.h_source, row.scope, row.scheme)]
        subset = group[np.isfinite(group[row.model])]
        if row.range == "in_range":
            subset = subset[subset.in_range]
        assert len(subset) == row.n_predictions
        # 独立使用每个测量日总权重1的定义，避免调用原拟合工具的指标函数。
        w = 1 / subset.groupby("visit_key").record_id.transform("count").to_numpy(float)
        w /= w.sum()
        y = subset.sampled_length_m.to_numpy(float)
        e = subset[row.model].to_numpy(float) - y
        mu = np.dot(w, y)
        mae = np.dot(w, abs(e))
        rmse = np.sqrt(np.dot(w, e ** 2))
        variance = np.dot(w, (y - mu) ** 2)
        r2 = 1 - np.dot(w, e ** 2) / variance if variance > 1e-15 else np.nan
        assert np.isclose(row.mae_m, mae, rtol=1e-9, atol=1e-9)
        assert np.isclose(row.rmse_m, rmse, rtol=1e-9, atol=1e-9)
        assert np.isclose(row.relative_mae, mae / mu, rtol=1e-9, atol=1e-9)
        if np.isfinite(r2):
            assert np.isclose(row.r2, r2, rtol=1e-8, atol=1e-8)
        checked += 1
    reconstructed = 0
    for (basin, source, scope, scheme), group in groups.items():
        inventory = raw.loc[group.record_id]
        assert inventory.accepted.all() and inventory.basin.eq(basin).all()
        assert np.allclose(inventory.sampled_length_m, group.sampled_length_m)
        assert np.allclose(inventory["h_local_m" if source == "local" else "h_official_m"], group.h_m)
        assert np.array_equal(inventory.visit_key.to_numpy(), group.visit_key.to_numpy())
        if scope.startswith("endpoint_zero"):
            assert inventory.endpoint_zero.all()
        if "dominant_position" in scope or scope == "endpoint_zero_known_positions":
            assert inventory.position_known.all() and inventory.channel_number.eq(1).all()
        if "dominant_position" in scope:
            assert inventory.position_key.nunique() == 1
        if scope.endswith("_method"):
            assert inventory.method_key.nunique() == 1
        if scheme == "leave_one_visit_out":
            assert np.array_equal(group.fold.to_numpy(), group.visit_key.to_numpy())
            for visit, test in group.groupby("visit_key"):
                # 用独立的加权协方差公式重建留出直线，确认留出日没有进入参数估计。
                train = group[~group.visit_key.eq(visit)]
                mass = 1 / train.groupby("visit_key").record_id.transform("size").to_numpy(float)
                mass /= mass.sum()
                x, y = train.h_m.to_numpy(float), train.sampled_length_m.to_numpy(float)
                mx, my = np.dot(mass, x), np.dot(mass, y)
                assert np.allclose(test.constant, my, rtol=1e-9, atol=1e-9)
                variance = np.dot(mass, (x - mx) ** 2)
                if train.h_m.nunique() >= 2 and variance > 0:
                    slope = np.dot(mass, (x - mx) * (y - my)) / variance
                    estimate = my + slope * (test.h_m.to_numpy(float) - mx)
                    assert np.allclose(test.linear, estimate, rtol=1e-8, atol=1e-8)
                    reconstructed += 1
        if scheme == "leave_one_position_out":
            assert np.array_equal(group.fold.to_numpy(), group.position_key.to_numpy())
    result = {"status": "PASS", "independently_recomputed_metric_rows": checked,
              "validated_prediction_records": len(predicted), "validated_scope_scheme_groups": len(groups),
              "independently_reconstructed_linear_folds": reconstructed,
              "source_h_and_length_match_inventory": True, "endpoint_and_position_filters_pass": True,
              "all86_station_rows": len(pd.read_csv(OUTPUT / "all86_station_summary.csv")),
              "note": "拟合时另逐折断言测量日隔离、时间顺序和整位置隔离；本工具从保存的预测独立重算指标。"}
    (OUTPUT / "integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
