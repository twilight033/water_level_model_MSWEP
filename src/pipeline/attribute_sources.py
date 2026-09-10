"""直接从 CAMELSH 原始 CSV 读取流域静态属性，不经 hydrodataset。

为什么不用 hydrodataset
----------------------
`hydrodataset.Camelsh.read_attr_xrdataset` 内部维护一张变量名映射表，只暴露映射
过的子集，名字不在表里就报"不是标准变量名"。项目早期据此认为"部分属性不可用"，
实际是误判：**全部 86 个流域在 28 个属性文件中均有完整记录，697 个数值属性可用
（缺失<5% 且非常数）**。直接用 pandas 读原始 CSV 即可绕开该限制。

现有 12 个属性的来源已逐列比对确认与缓存完全一致（见 BASE_SPECS）。

径流导出属性的黑名单
------------------
`BFI_AVE`（基流指数）以及 `attributes_gageii_FlowRec.csv` 的全部列都是**从实测
径流统计出来的**。按流域留出情景假设该流域没有径流数据，用这些列等于绕个弯把
径流信息还给模型，会让整个替代实验失效。本模块在构表时强制拒绝这些列。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import EXPORT_DIR, load_basin_ids, verify_camelsh_path  # noqa: E402

# hydroATLAS 是制表符分隔的，其余为逗号
TAB_SEPARATED = ("attributes_hydroATLAS.csv",)

# 由实测径流导出，任何情况下都不得作为静态属性使用
FLOW_DERIVED_FILES = ("attributes_gageii_FlowRec.csv",)
FLOW_DERIVED_COLUMNS = ("BFI_AVE",)

# 在这 86 个流域上为常数，无信息
CONSTANT_COLUMNS = ("SNOWICENLCD06", "prm_pc_use")


class AttrSpec:
    """一个静态属性的来源声明。

    kind="categorical" 时输出 one-hot 列，列名为 ``{name}__{code}``：
    数值类别码直接用码本身，字符串类别按字典序映射为 0,1,2…（与既有缓存一致）。
    """

    __slots__ = ("name", "file", "column", "kind")

    def __init__(self, name, file, column, kind="continuous"):
        if kind not in ("continuous", "categorical"):
            raise ValueError(f"未知属性类型 {kind!r}")
        if column in FLOW_DERIVED_COLUMNS or file in FLOW_DERIVED_FILES:
            raise ValueError(
                f"{file}:{column} 由实测径流导出，不得用作静态属性——"
                f"按流域留出情景假设该流域无径流数据")
        if column in CONSTANT_COLUMNS:
            raise ValueError(f"{file}:{column} 在本流域集上为常数，无信息")
        self.name, self.file, self.column, self.kind = name, file, column, kind


# ── 基础集：复现项目既有的 12 个属性（逐列比对确认一致）────────────────────────
BASE_SPECS = (
    AttrSpec("area", "attributes_gageii_BasinID.csv", "DRAIN_SQKM"),
    AttrSpec("p_mean", "attributes_nldas2_climate.csv", "p_mean"),
    AttrSpec("p_seasonality", "attributes_nldas2_climate.csv", "p_seasonality"),
    AttrSpec("frac_snow", "attributes_nldas2_climate.csv", "frac_snow"),
    AttrSpec("aridity", "attributes_nldas2_climate.csv", "aridity_index"),
    AttrSpec("slope_mean", "attributes_gageii_Topo.csv", "SLOPE_PCT"),
    AttrSpec("frac_forest", "attributes_gageii_LC06_Basin.csv", "FORESTNLCD06"),
    AttrSpec("soil_depth_statgso", "attributes_gageii_Soils.csv", "ROCKDEPAVE"),
    AttrSpec("swc_pc_syr", "attributes_hydroATLAS.csv", "swc_pc_syr"),
    AttrSpec("geol_permeability", "attributes_gageii_Soils.csv", "PERMAVE"),
    AttrSpec("dom_land_cover", "attributes_hydroATLAS.csv", "glc_cl_smj", "categorical"),
    AttrSpec("geol_class_1st", "attributes_gageii_Geology.csv",
             "GEOL_REEDBUSH_DOM", "categorical"),
)

# ── 扩展集新增：逐流域 NSE 相关分析与水文机理共同筛出 ─────────────────────────
EXTRA_SPECS = (
    # 地形。高程与逐流域 NSE 相关 +0.462，此前在 config.py 中被注释掉
    AttrSpec("elev_mean", "attributes_gageii_Topo.csv", "ELEV_MEAN_M_BASIN"),
    AttrSpec("elev_std", "attributes_gageii_Topo.csv", "ELEV_STD_M_BASIN"),
    # 气候。水量平衡的支出项，此前完全缺失
    AttrSpec("pet_mean", "attributes_nldas2_climate.csv", "pet_mean"),
    # 土壤。水文土壤组四组和为 100，只取两端避免完全共线
    AttrSpec("soil_group_a", "attributes_gageii_Soils.csv", "HGA"),
    AttrSpec("soil_group_d", "attributes_gageii_Soils.csv", "HGD"),
    AttrSpec("soil_awc", "attributes_gageii_Soils.csv", "AWCAVE"),
    # 地下水。21 天窗口看不到的那部分记忆；两者均非径流导出
    AttrSpec("water_table_depth", "attributes_hydroATLAS.csv", "gwt_cm_sav"),
    AttrSpec("subsurface_contact", "attributes_gageii_Hydro.csv", "CONTACT"),
    # 人类活动。与逐流域 NSE 相关最强的三项都在这里，且此前全部缺失
    AttrSpec("pop_density", "attributes_gageii_Pop_Infrastr.csv", "PDEN_2000_BLOCK"),
    AttrSpec("urban_frac", "attributes_gageii_LC06_Basin.csv", "DEVNLCD06"),
    AttrSpec("dam_density", "attributes_gageii_HydroMod_Dams.csv", "DDENS_2009"),
    AttrSpec("reservoir_storage", "attributes_gageii_HydroMod_Dams.csv", "STOR_NOR_2009"),
)

EXTENDED_SPECS = BASE_SPECS + EXTRA_SPECS
ATTR_SETS = {"base": BASE_SPECS, "extended": EXTENDED_SPECS}


def _attr_dir() -> Path:
    from config import CAMELSH_DATA_PATH
    return verify_camelsh_path(CAMELSH_DATA_PATH) / "CAMELSH" / "attributes"


def _read_source(file_name: str, basins: list) -> pd.DataFrame:
    """读取一个属性文件并对齐到给定流域顺序。STAID 需补零到 8 位。"""
    sep = "\t" if file_name in TAB_SEPARATED else ","
    df = pd.read_csv(_attr_dir() / file_name, sep=sep, dtype={"STAID": str},
                     low_memory=False)
    if "STAID" not in df.columns:
        raise ValueError(f"{file_name} 缺少 STAID 列")
    df["STAID"] = df["STAID"].astype(str).str.strip().str.zfill(8)
    return df.set_index("STAID").reindex(basins)


def _encode_categorical(values: pd.Series, name: str) -> pd.DataFrame:
    """类别 → one-hot。数值码直接用码本身作后缀；字符串按字典序映射为 0,1,2…

    字典序映射是为了与既有缓存一致：GEOL_REEDBUSH_DOM 的
    gneiss/granitic/sedimentary 恰好映射为 0/1/2（计数 4/22/60）。
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        codes = numeric.astype("int64")
    else:
        levels = sorted(values.dropna().unique())
        codes = values.map({lv: i for i, lv in enumerate(levels)}).astype("int64")
    dummies = pd.get_dummies(codes, prefix=f"{name}_").astype("float32")
    return dummies


def build_attribute_table(attr_set: str = "base", basins: list = None) -> tuple:
    """构建属性矩阵，返回 (DataFrame, one-hot 列名列表)。行索引为流域 ID。"""
    if attr_set not in ATTR_SETS:
        raise ValueError(f"attr_set 必须是 {tuple(ATTR_SETS)} 之一，收到 {attr_set!r}")
    basins = [str(b) for b in (basins if basins is not None else load_basin_ids())]
    specs = ATTR_SETS[attr_set]

    # 按文件分组，每个文件只读一次
    cache = {}
    for spec in specs:
        if spec.file not in cache:
            cache[spec.file] = _read_source(spec.file, basins)

    parts, onehot_cols = [], []
    for spec in specs:
        raw = cache[spec.file][spec.column]
        if spec.kind == "continuous":
            col = pd.to_numeric(raw, errors="coerce").astype("float32")
            col.name = spec.name
            parts.append(col.to_frame())
        else:
            dummies = _encode_categorical(raw, spec.name)
            parts.append(dummies)
            onehot_cols.extend(dummies.columns.tolist())

    table = pd.concat(parts, axis=1)
    table.index.name = "gauge_id"

    missing = table.isna().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        raise ValueError(f"属性存在缺失值，需先决定填补策略:\n{bad.to_string()}")
    return table, onehot_cols


def cache_path(attr_set: str) -> Path:
    suffix = "" if attr_set == "base" else f"_{attr_set}"
    return EXPORT_DIR / f"attributes_86{suffix}.parquet"


def get_attribute_table(attr_set: str = "base", overwrite: bool = False,
                        basins: list = None) -> tuple:
    """带 parquet 缓存的属性矩阵。缓存后不再依赖原始数据集是否可访问。"""
    path = cache_path(attr_set)
    if not overwrite and path.exists():
        df = pd.read_parquet(path)
        onehot = [c for c in df.columns if "__" in c]
        return (df.reindex([str(b) for b in basins]) if basins is not None else df), onehot

    df, onehot = build_attribute_table(attr_set, basins)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd")
    return df, onehot


if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    for name in ("base", "extended"):
        df, onehot = get_attribute_table(name, overwrite=overwrite)
        cont = [c for c in df.columns if c not in onehot]
        print(f"{name:9s}: {df.shape[0]} 流域 × {df.shape[1]} 列"
              f"（连续 {len(cont)} + one-hot {len(onehot)}）→ {cache_path(name).name}")
        print(f"           连续列: {cont}")
        print(f"           one-hot: {onehot}")
