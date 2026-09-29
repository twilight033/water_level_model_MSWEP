"""把 USGS 官方率定曲线套到 CAMELSH 实测水位上，与项目的分箱曲线比 NSE。

动机
----
项目的率定基线是归一化空间的 50 级阶梯（分箱均值），而 USGS 官方的是物理单位下的
分段幂律（对数空间折点内插）。两者形式差别很大，需要知道：**官方那条曲线在同一套
数据上能到多少 NSE？** 这既是对自拟合曲线质量的外部校验，也直接量化了
"CAMELSH 的径流由水位经率定曲线换算而来"这一事实的强度。

方法
----
官方 `base` 文件给出折点表与偏移量，USGS 用对数内插（`RATING EXPANSION="logarithmic"`）：
在 log(h - offset) 与 log(Q) 之间对折点做线性内插，超出折点范围时按端段斜率外推
（USGS 自身亦如此延伸）。单位需换算——CAMELSH 为 m 与 m³/s，官方为 ft 与 ft³/s。

已知局限（结果解读时不可回避）
----
1. **官方只发布当前在用的那一版**。本研究区率定曲线中位改到第 17 版、最多第 96 版，
   历史时段实际使用的版本无从获取，故只在测试段评估——越靠近当前，当前版越适用。
2. **逐时段 shift 拿不到**。USGS 产出径流时会逐时段叠加 shift，公开接口只给当前 shift，
   因此本脚本复现的是"未加 shift 的基础曲线"，必然劣于官方实际产出的径流。
3. 因此本对比回答的是"官方曲线能解释多少"，**不是**"谁的曲线更好"。
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.rating_baseline import _pairs, apply_rating, fit_rating, nse  # noqa: E402
from evaluation.usgs_rating_metadata import CACHE_DIR  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402

FT_PER_M = 1 / 0.3048
CMS_PER_CFS = 0.0283168466

_RE_OFFSET = re.compile(r"# //RATING OFFSET(\d)=([-\d.E+]+)")
_RE_BREAKPOINT = re.compile(r"# //RATING BREAKPOINT(\d)=([-\d.E+]+)")
_RE_EXPANSION_VAL = re.compile(r'# //RATING EXPANSION="([^"]*)"')


def parse_base_rating(text: str) -> dict:
    """从 base 文件解析折点表、偏移量与分段点。"""
    offsets = {int(i): float(v) for i, v in _RE_OFFSET.findall(text)}
    breakpoints = {int(i): float(v) for i, v in _RE_BREAKPOINT.findall(text)}

    indep, dep = [], []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            indep.append(float(parts[0]))
            dep.append(float(parts[1]))
        except ValueError:
            continue  # 表头 INDEP/DEP 与格式行 16N/16N

    return {
        "indep": np.array(indep), "dep": np.array(dep),
        "offsets": offsets, "breakpoints": breakpoints,
        "expansion": (_RE_EXPANSION_VAL.search(text).group(1)
                      if _RE_EXPANSION_VAL.search(text) else None),
    }


def _offset_for(h_ft: np.ndarray, rating: dict) -> np.ndarray:
    """逐点选用的偏移量。多段时按 BREAKPOINT 切换，单段时全用 OFFSET1。"""
    offsets, breaks = rating["offsets"], rating["breakpoints"]
    out = np.full(h_ft.shape, offsets.get(1, 0.0))
    for i in sorted(breaks):
        nxt = offsets.get(i + 1)
        if nxt is not None:
            out[h_ft >= breaks[i]] = nxt
    return out


def apply_official_rating(h_m: np.ndarray, rating: dict) -> np.ndarray:
    """官方曲线：对数空间折点内插，端外按端段斜率外推。返回 m³/s。"""
    h_ft = h_m * FT_PER_M
    off_pts = _offset_for(rating["indep"], rating)
    off_h = _offset_for(h_ft, rating)

    # 折点与待求点都要减去各自的偏移，且必须落在正域才能取对数
    xp = rating["indep"] - off_pts
    yp = rating["dep"]
    keep = (xp > 0) & (yp > 0)
    if keep.sum() < 2:
        return np.full(h_m.shape, np.nan)
    lxp, lyp = np.log(xp[keep]), np.log(yp[keep])
    order = np.argsort(lxp)
    lxp, lyp = lxp[order], lyp[order]

    x = h_ft - off_h
    out = np.full(h_m.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    lx = np.log(x[ok])

    ly = np.interp(lx, lxp, lyp)
    # np.interp 端外取常数；USGS 按端段斜率延伸，这里照做
    if lxp.size >= 2:
        lo_slope = (lyp[1] - lyp[0]) / (lxp[1] - lxp[0])
        hi_slope = (lyp[-1] - lyp[-2]) / (lxp[-1] - lxp[-2])
        below, above = lx < lxp[0], lx > lxp[-1]
        ly[below] = lyp[0] + lo_slope * (lx[below] - lxp[0])
        ly[above] = lyp[-1] + hi_slope * (lx[above] - lxp[-1])

    out[ok] = np.exp(ly) * CMS_PER_CFS
    return out


def compare_basin(basin: str, prepared, split: str = "test") -> dict:
    """在同一段配对观测上比较官方曲线与项目分箱曲线。"""
    cache = CACHE_DIR / f"{basin}.base.rdb"
    row = {"basin": basin, "n_pairs": 0, "nse_official": np.nan,
           "nse_binned_self": np.nan, "nse_binned_train_fit": np.nan,
           "n_breakpoints": np.nan, "note": ""}
    if not cache.is_file():
        row["note"] = "无官方 base 文件缓存"
        return row

    rating = parse_base_rating(cache.read_text(encoding="utf-8", errors="replace"))
    if rating["indep"].size < 2:
        row["note"] = "官方文件无折点（该站无当前率定曲线）"
        return row
    row["n_breakpoints"] = len(rating["indep"])

    # 项目侧用的是归一化值，官方曲线要物理量，故两边分别取数后按同一掩膜对齐
    h_norm, q_norm = _pairs(prepared, basin, split)
    bi = prepared.basin_index[basin]
    lo, hi = prepared.split_range(basin, split)
    q_raw = prepared.targets_raw["flow"][bi, lo:hi + 1].astype(float)
    h_raw = prepared.targets_raw["waterlevel"][bi, lo:hi + 1].astype(float)
    ok = np.isfinite(q_raw) & np.isfinite(h_raw)
    q_raw, h_raw = q_raw[ok], h_raw[ok]
    if q_raw.size < 500:
        row["note"] = f"{split} 段配对观测不足 500"
        return row
    row["n_pairs"] = int(q_raw.size)

    row["nse_official"] = nse(q_raw, apply_official_rating(h_raw, rating))

    # 项目分箱曲线：同期自拟合（上界）与训练期拟合→本段应用
    fit_self = fit_rating(h_norm, q_norm)
    if fit_self is not None:
        row["nse_binned_self"] = nse(q_norm, apply_rating(fit_self, h_norm))
    h_tr, q_tr = _pairs(prepared, basin, "train")
    if h_tr.size:
        fit_tr = fit_rating(h_tr, q_tr)
        if fit_tr is not None:
            row["nse_binned_train_fit"] = nse(q_norm, apply_rating(fit_tr, h_norm))
    return row


def load_official_ratings(basins) -> dict:
    """站号 → 解析好的官方曲线；无当前曲线的站不入表。"""
    out = {}
    for basin in basins:
        cache = CACHE_DIR / f"{basin}.base.rdb"
        if not cache.is_file():
            continue
        rating = parse_base_rating(cache.read_text(encoding="utf-8", errors="replace"))
        if rating["indep"].size >= 2:
            out[basin] = rating
    return out


def two_stage_with_official(prepared, ratio: float = 0.70,
                            mask_seeds=(42, 123, 456), config: str = "base",
                            max_wl_seeds: int = 3) -> pd.DataFrame:
    """把两阶段基线的第二步换成官方率定曲线，与自拟合分箱曲线逐流域对照。

    两条路径共用同一个 ``single_waterlevel`` 模型的预测水位，只有换算方式不同：

    - 自拟合：预测水位（归一化）→ 保留流域拟合的区域化分箱曲线 → 归一化径流；
    - 官方：预测水位（归一化）→ 用**实测水位统计量**反归一化成米 → 官方曲线 → m³/s。

    官方路径的关键性质：它直接产出物理流量，**不需要留出流域的任何径流统计量**。
    因此它在 physical 配置下同样合法，可用来检验"去掉已知流量量级后率定路径会塌"
    这一结论是否源于换算方式本身。

    NSE 对逐流域仿射变换不变，故归一化空间与物理空间的取值一致，两条路径可直接比。
    """
    from evaluation.two_stage_baseline import (WL_SOURCES, _global_rating,
                                               available_waterlevel_runs,
                                               predict_waterlevel)
    from pipeline.dataset import TargetScaling
    from pipeline.masking import get_hidden

    _, _, _, seq_length, scaling = WL_SOURCES[config]
    wl_runs = available_waterlevel_runs(config, max_wl_seeds)
    if not wl_runs:
        raise FileNotFoundError(f"未找到 {config} 配置的单任务水位模型权重")
    predictions = {seed: predict_waterlevel(prepared, ckpt, seq_length)
                   for seed, ckpt in wl_runs}

    # 水位统计量在两种口径下都用实测——留出情景的前提就是该流域有水位记录
    wl_scale = TargetScaling(prepared, "observed")
    officials = load_official_ratings(prepared.splits["splits"])
    q_raw = prepared.targets_raw["flow"]

    collected = {}
    for mask_seed in mask_seeds:
        _, stats = get_hidden(prepared, {"flow": ratio},
                              mechanism="basin_holdout", mask_seed=mask_seed)
        held = set(stats["per_task"]["flow"]["held_out_basins"])
        kept = [b for b in prepared.splits["splits"] if b not in held]
        q_norm = (TargetScaling(prepared, scaling, fit_basins=kept).normalized["flow"]
                  if scaling == "physical" else prepared.targets["flow"])
        binned = _global_rating(prepared, kept, q_norm)
        if binned is None:
            continue

        for pred in predictions.values():
            for basin in held:
                bi = prepared.basin_index[basin]
                sel = pred["basin_idx"] == bi
                if sel.sum() < 500:
                    continue
                pos = pred["target_pos"][sel]
                h_hat = pred["pred_h"][sel].astype(float)
                slot = collected.setdefault(basin, {"binned": [], "official": []})

                slot["binned"].append(nse(q_norm[bi, pos].astype(float),
                                          apply_rating(binned, h_hat)))
                if basin in officials:
                    h_m = wl_scale.denormalize("waterlevel", bi, h_hat)
                    slot["official"].append(
                        nse(q_raw[bi, pos].astype(float),
                            apply_official_rating(h_m, officials[basin])))

    rows = []
    for basin, slot in collected.items():
        vals = {k: [x for x in v if np.isfinite(x)] for k, v in slot.items()}
        rows.append({
            "basin": basin, "config": config,
            "nse_binned": float(np.mean(vals["binned"])) if vals["binned"] else np.nan,
            "nse_official": float(np.mean(vals["official"])) if vals["official"] else np.nan,
            "has_official": basin in officials,
        })
    return pd.DataFrame(rows).sort_values("basin").reset_index(drop=True)


def run_two_stage_comparison():
    """四个配置各跑一遍，与双头、单任务径流并列报告。"""
    from evaluation.run_config import load_summary, select
    from evaluation.two_stage_baseline import WL_SOURCES
    from pipeline.dataset import PreparedData

    metrics, _ = load_summary()
    configs = ["base", "physical", "L480+extended", "L480+physical+extended"]

    # base 与 extended 的输入维度不同（18 vs 30），须按属性集分别构造
    prepared_cache = {}

    def get_prepared(attr_set):
        if attr_set not in prepared_cache:
            prepared_cache[attr_set] = PreparedData(attr_set=attr_set)
        return prepared_cache[attr_set]

    frames, summary = [], []
    for config in configs:
        _, _, attr_set, _, _ = WL_SOURCES[config]
        table = two_stage_with_official(get_prepared(attr_set), config=config)
        frames.append(table)

        sub = select(metrics, config=config, scenario="q_holdout70",
                     task="flow", held_out=True)
        ok = table.dropna(subset=["nse_official", "nse_binned"])
        idx = ok["basin"]
        row = {"config": config, "n": len(ok),
               "两阶段_自拟合曲线": ok["nse_binned"].mean(),
               "两阶段_官方曲线": ok["nse_official"].mean()}
        for arch, label in (("single_flow", "单任务径流"), ("dual_head", "双头")):
            per_basin = sub[sub["architecture"] == arch].groupby("basin")["nse"].mean()
            row[label] = float(per_basin.reindex(idx).mean())
        summary.append(row)

    out_dir = RESULTS_ROOT / "summary"
    path = out_dir / "two_stage_official_rating.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")

    table = pd.DataFrame(summary)
    print("70% 留出、被留出流域的平均 NSE（均只用气象输入）\n")
    cols = ["config", "n", "单任务径流", "两阶段_自拟合曲线", "两阶段_官方曲线", "双头"]
    print(table[cols].round(4).to_string(index=False))

    print()
    print("官方曲线 − 自拟合曲线:")
    for row in summary:
        print(f"  {row['config']:24s} {row['两阶段_官方曲线'] - row['两阶段_自拟合曲线']:+.4f}")
    print()
    print("双头 − 两阶段（官方曲线）:")
    for row in summary:
        print(f"  {row['config']:24s} {row['双头'] - row['两阶段_官方曲线']:+.4f}")
    print()
    print(f"  已写入 {path}")


def main():
    from pipeline.dataset import PreparedData

    prepared = PreparedData()
    basins = list(prepared.splits["splits"])
    rows = [compare_basin(b, prepared) for b in basins]
    table = pd.DataFrame(rows)

    out_dir = RESULTS_ROOT / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "official_vs_binned_rating.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    usable = table.dropna(subset=["nse_official", "nse_binned_train_fit"])
    print(f"测试段逐流域 NSE（{len(usable)}/{len(table)} 个流域可比）\n")
    print(f"{'口径':34s} {'均值':>8s} {'中位':>8s}")
    for col, label in (("nse_official", "USGS 官方曲线（物理量、当前版）"),
                       ("nse_binned_self", "项目分箱曲线（同期自拟合，上界）"),
                       ("nse_binned_train_fit", "项目分箱曲线（训练期拟合→测试期）")):
        v = usable[col].astype(float)
        print(f"{label:34s} {v.mean():8.4f} {v.median():8.4f}")

    diff = usable["nse_binned_train_fit"] - usable["nse_official"]
    print()
    print(f"分箱（训练期拟合）− 官方: 均值 {diff.mean():+.4f}  中位 {diff.median():+.4f}  "
          f"分箱更高的流域 {int((diff > 0).sum())}/{len(diff)}")
    print()
    print("官方曲线 NSE 最低的 5 个流域（多为率定改版或超出折点范围）:")
    print(usable.nsmallest(5, "nse_official")[
        ["basin", "n_pairs", "n_breakpoints", "nse_official", "nse_binned_train_fit"]
    ].to_string(index=False))
    if table["note"].astype(bool).any():
        print()
        print("未参与对比的流域:")
        print(table.loc[table["note"].astype(bool), ["basin", "note"]].to_string(index=False))
    print()
    print(f"  已写入 {path}")


if __name__ == "__main__":
    # --two-stage：接进两阶段流程与双头对照；不带参数：只做静态曲线对比
    if "--two-stage" in sys.argv:
        run_two_stage_comparison()
    else:
        main()
