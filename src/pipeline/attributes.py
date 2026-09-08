"""流域静态属性的读取与编码。

回应审稿意见 Major 6：dom_land_cover 与 geol_class_1st 在 CAMELSH 中是
整数类别编码（实测取值分别为 {2, 6} 与 {0, 1, 2}），原实现把它们与连续
属性一起做 Z-score 标准化，等于人为规定"地质类别 1 位于类别 0 与 2 之间
且等距"。这里改为 one-hot，且 one-hot 列不参与标准化。

实测 86 个流域的 12 个属性无缺失值，因此不需要填补策略；若将来换属性
集合出现缺失，_assert_no_missing 会直接报错而不是静默填均值。
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "src" / "others")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import EXPORT_DIR, load_basin_ids, verify_camelsh_path  # noqa: E402

# CAMELSH 中以整数类别编码存储的属性，必须 one-hot 而非 Z-score
CATEGORICAL_ATTRS = ("dom_land_cover", "geol_class_1st")
ATTRS_CACHE = EXPORT_DIR / "attributes_86.parquet"


def _assert_no_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        raise ValueError(
            f"静态属性存在缺失值，需先决定填补策略再继续:\n{bad.to_string()}"
        )


def build_attributes(overwrite: bool = False) -> tuple:
    """返回 (属性 DataFrame, one-hot 列名列表)。行索引为流域 ID。"""
    if not overwrite and ATTRS_CACHE.exists():
        df = pd.read_parquet(ATTRS_CACHE)
        onehot = [c for c in df.columns if c.startswith(tuple(f"{a}__" for a in CATEGORICAL_ATTRS))]
        return df, onehot

    from improved_camelsh_reader import ImprovedCAMELSHReader
    from config import CAMELSH_DATA_PATH, ATTRIBUTE_VARIABLES

    data_root = verify_camelsh_path(CAMELSH_DATA_PATH)
    basins = load_basin_ids()
    reader = ImprovedCAMELSHReader(str(data_root), download=False, use_batch=True)
    raw = reader.read_attr_xrdataset(gage_id_lst=basins, var_lst=ATTRIBUTE_VARIABLES).to_pandas()
    raw.index = [str(i) for i in raw.index]
    raw = raw.reindex(basins)
    _assert_no_missing(raw)

    present = [a for a in CATEGORICAL_ATTRS if a in raw.columns]
    continuous = [c for c in raw.columns if c not in present]

    parts = [raw[continuous].astype("float32")]
    onehot_cols = []
    for attr in present:
        codes = raw[attr].astype("int64")
        dummies = pd.get_dummies(codes, prefix=f"{attr}_").astype("float32")
        # 列名形如 dom_land_cover__2，双下划线便于识别 one-hot 列
        parts.append(dummies)
        onehot_cols.extend(dummies.columns.tolist())
        print(f"  {attr}: {codes.nunique()} 类 → one-hot {list(dummies.columns)}")

    df = pd.concat(parts, axis=1)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ATTRS_CACHE, compression="zstd")
    print(f"属性矩阵: {df.shape}（连续 {len(continuous)} 列 + one-hot {len(onehot_cols)} 列）")
    print(f"已写入: {ATTRS_CACHE}")
    return df, onehot_cols


if __name__ == "__main__":
    df, onehot = build_attributes(overwrite="--overwrite" in sys.argv)
    print(f"\n列: {list(df.columns)}")
    print(f"one-hot 列（不参与标准化）: {onehot}")
