"""逐流域性能与流域属性的关联诊断。

回应审稿意见 Major 8「应报告失败流域而不只给总体平均」。除了列出表现最差的
流域，更有解释力的是说明**哪一类流域**难以预测。

实测结论（86 个流域，双头模型完整标签）：与逐流域 NSE 相关最强的是人口密度
（−0.535）、城镇化率（−0.516）、平均高程（+0.462），其中前两项此前完全不在
模型输入里，高程则在 config.py 中被注释掉。该流域集 74/86 有水坝、58/86 为
GAGES-II 非参照流域。

**相关不等于因果**：城镇化流域预测差的根因是雨水管网与快速汇流在 3 小时气象
输入下难以刻画，加一个静态属性只能让模型按流域类型分别校准，修不好根因；
高程、地下水位埋深、坝密度则是模型真正缺失的信息，有明确物理理由。论文中
必须区分这两类。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import COVERAGE_DIR, RESULTS_ROOT, load_basin_ids  # noqa: E402

SUMMARY_DIR = RESULTS_ROOT / "summary"

# 诊断用的候选属性：(展示名, 文件, 列)。此处只做相关分析，不进入模型输入，
# 因此可以包含 GAGES-II 的扰动分类等描述性变量。
DIAGNOSTIC_COLUMNS = [
    ("流域面积", "attributes_gageii_BasinID.csv", "DRAIN_SQKM"),
    ("平均高程", "attributes_gageii_Topo.csv", "ELEV_MEAN_M_BASIN"),
    ("高程标准差", "attributes_gageii_Topo.csv", "ELEV_STD_M_BASIN"),
    ("平均坡度", "attributes_gageii_Topo.csv", "SLOPE_PCT"),
    ("森林比例", "attributes_gageii_LC06_Basin.csv", "FORESTNLCD06"),
    ("城镇化比例", "attributes_gageii_LC06_Basin.csv", "DEVNLCD06"),
    ("水体比例", "attributes_gageii_LC06_Basin.csv", "WATERNLCD06"),
    ("人口密度", "attributes_gageii_Pop_Infrastr.csv", "PDEN_2000_BLOCK"),
    ("路网密度", "attributes_gageii_Pop_Infrastr.csv", "ROADS_KM_SQ_KM"),
    ("坝密度", "attributes_gageii_HydroMod_Dams.csv", "DDENS_2009"),
    ("库容径流比", "attributes_gageii_HydroMod_Dams.csv", "STOR_NOR_2009"),
    ("水文扰动指数", "attributes_gageii_Bas_Classif.csv", "HYDRO_DISTURB_INDX"),
    ("河网密度", "attributes_gageii_Hydro.csv", "STREAMS_KM_SQ_KM"),
    ("最大河流级别", "attributes_gageii_Hydro.csv", "STRAHLER_MAX"),
    ("地下水位埋深", "attributes_hydroATLAS.csv", "gwt_cm_sav"),
    ("年均降水", "attributes_nldas2_climate.csv", "p_mean"),
    ("潜在蒸散", "attributes_nldas2_climate.csv", "pet_mean"),
    ("融雪比例", "attributes_nldas2_climate.csv", "frac_snow"),
]


def _read(file_name: str, basins: list) -> pd.DataFrame:
    from pipeline.attribute_sources import _attr_dir, TAB_SEPARATED

    sep = "\t" if file_name in TAB_SEPARATED else ","
    df = pd.read_csv(_attr_dir() / file_name, sep=sep, dtype={"STAID": str},
                     low_memory=False)
    df["STAID"] = df["STAID"].astype(str).str.strip().str.zfill(8)
    return df.set_index("STAID").reindex(basins)


def load_diagnostic_attributes(basins=None) -> pd.DataFrame:
    basins = [str(b) for b in (basins if basins is not None else load_basin_ids())]
    cache, cols = {}, {}
    for label, file_name, column in DIAGNOSTIC_COLUMNS:
        if file_name not in cache:
            cache[file_name] = _read(file_name, basins)
        cols[label] = pd.to_numeric(cache[file_name][column], errors="coerce")
    table = pd.DataFrame(cols, index=basins)
    table.index.name = "basin"
    return table


def correlate_with_performance(metrics: pd.DataFrame, attributes: pd.DataFrame,
                               group: str = "4A_main", architecture: str = "dual_head",
                               task: str = "flow", metric: str = "nse") -> pd.DataFrame:
    from scipy import stats as sps

    sub = metrics[(metrics["group"] == group) & (metrics["architecture"] == architecture)
                  & (metrics["task"] == task)]
    score = sub.groupby("basin")[metric].mean()
    rows = []
    for label in attributes.columns:
        values = attributes[label].reindex(score.index)
        ok = values.notna() & score.notna()
        if ok.sum() < 10:
            continue
        rho, p = sps.spearmanr(values[ok], score[ok])
        rows.append({"属性": label, "spearman_rho": float(rho), "p": float(p),
                     "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def worst_basins(metrics: pd.DataFrame, attributes: pd.DataFrame, n: int = 10,
                 group: str = "4A_main", architecture: str = "dual_head",
                 task: str = "flow") -> pd.DataFrame:
    sub = metrics[(metrics["group"] == group) & (metrics["architecture"] == architecture)
                  & (metrics["task"] == task)]
    score = sub.groupby("basin")["nse"].mean().sort_values()
    return attributes.reindex(score.index[:n]).assign(nse=score.iloc[:n]).round(3)


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SUMMARY_DIR / "all_metrics.csv", dtype={"basin": str},
                          low_memory=False)
    attrs = load_diagnostic_attributes()
    attrs.to_csv(COVERAGE_DIR / "basin_diagnostic_attributes.csv",
                 encoding="utf-8-sig")

    for task in ("flow", "waterlevel"):
        corr = correlate_with_performance(metrics, attrs, task=task)
        path = SUMMARY_DIR / f"attribute_performance_{task}.csv"
        corr.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n=== {task} 任务：逐流域 NSE 与属性的 Spearman 相关 ===")
        print(corr.head(10).round(4).to_string(index=False))
        print(f"  已写入 {path}")

    worst = worst_basins(metrics, attrs)
    worst.to_csv(SUMMARY_DIR / "worst_basins_flow.csv", encoding="utf-8-sig")
    print(f"\n=== 径流表现最差的 10 个流域 ===")
    print(worst[["nse", "城镇化比例", "人口密度", "平均高程", "流域面积", "坝密度"]]
          .to_string())
    print(f"  已写入 {SUMMARY_DIR / 'worst_basins_flow.csv'}")


if __name__ == "__main__":
    main()
