"""两阶段率定基线：气象 → 预测水位 → 区域化率定曲线 → 径流。

这是"你不就是在隐式学率定曲线吗"这一质疑的**可执行版本**：先用充足的水位
标签训一个单任务水位模型，再用其他流域拟合的区域化率定关系把预测水位换算
成流量。推理时同样只用气象，因此与双头模型可以公平比较。

如果双头真的只是在隐式做这件事，显式版本应该打平甚至更好。事后估算显示
双头 0.571、两阶段 0.479——显式版本明显更差，说明双头迁移的是"气象→流域
蓄水状态"的表示，而不是"水位→流量"的映射；两阶段还会把水位预测误差经率定
变换再放大一次。本模块把该估算换成正式基线。

不需要新训练
------------
按流域留出情景只删径流标签、**水位标签未动**，因此 4A 已训好的
``single_waterlevel`` 模型看到的训练信号与留出情景下完全一致，直接复用其
权重重新推理即可。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.rating_baseline import apply_rating, fit_rating, nse  # noqa: E402
from evaluation.run_config import SUMMARY_DIR, load_summary, select  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402

RUNS_DIR = RESULTS_ROOT / "runs"
WL_RUN_PREFIX = "4A_main_complete_single_waterlevel_ms"


def available_waterlevel_runs(max_seeds: int = 3) -> list:
    """找出可复用的单任务水位模型权重，按种子号排序。"""
    runs = []
    for path in sorted(RUNS_DIR.glob(f"{WL_RUN_PREFIX}*")):
        ckpt = path / "model.pt"
        if ckpt.exists():
            seed = int(path.name.replace(WL_RUN_PREFIX, ""))
            runs.append((seed, ckpt))
    runs.sort()
    return runs[:max_seeds]


@torch.no_grad()
def predict_waterlevel(prepared, ckpt_path: Path, seq_length: int = 168,
                       device: str = None) -> dict:
    """加载权重重新推理，返回 {basin_idx: (target_pos, 归一化预测水位)}。"""
    from models.lstm_models import build_model
    from pipeline.dataset import WindowDataset, make_loader

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["model_config"]
    model = build_model("single_waterlevel", forcing_size=cfg["forcing_size"],
                        attr_size=cfg["attr_size"], hidden_size=cfg["hidden_size"],
                        dropout_rate=cfg["dropout_rate"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    ds = WindowDataset(prepared, "test", seq_length, 1, tasks=("waterlevel",))
    loader = make_loader(ds, 4096, shuffle=False)
    pred_chunks, bi_chunks, pos_chunks = [], [], []
    for x, c, _, _, bi, pos in loader:
        out = model(x.to(device), c.to(device))["waterlevel"].squeeze(-1)
        pred_chunks.append(out.cpu().numpy())
        bi_chunks.append(bi.numpy())
        pos_chunks.append(pos.numpy())
    return {"pred_h": np.concatenate(pred_chunks),
            "basin_idx": np.concatenate(bi_chunks),
            "target_pos": np.concatenate(pos_chunks)}


def _global_rating(prepared, fit_basins, q_norm: np.ndarray) -> tuple:
    """在 fit_basins 的训练段上拟合 归一化水位 → 归一化径流 的区域化率定关系。

    q_norm 是径流的归一化矩阵，由调用方决定口径（实测统计量或物理尺度）。
    """
    hs, qs = [], []
    for basin in fit_basins:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, "train")
        q = q_norm[bi, lo:hi + 1].astype(float)
        h = prepared.targets["waterlevel"][bi, lo:hi + 1].astype(float)
        ok = np.isfinite(q) & np.isfinite(h)
        if ok.any():
            hs.append(h[ok])
            qs.append(q[ok])
    return fit_rating(np.concatenate(hs), np.concatenate(qs))


def two_stage_scores(prepared, ratio: float = 0.70, mask_seeds=(42, 123, 456),
                     max_wl_seeds: int = 3, seq_length: int = 168,
                     scaling: str = "observed") -> pd.DataFrame:
    """逐流域的两阶段基线径流 NSE，在掩膜种子与模型种子上取平均。

    scaling 决定率定关系落在哪个径流空间：

    - ``observed``：逐流域实测 z-score。率定表把预测水位映射成"该流域实测
      均值方差下的 z 值"，换回流量要用留出流域自己的实测统计量——这与 base
      配置的模型拿到的信息一致，只能拿来对比 base 配置。
    - ``physical``：径流尺度由 面积×降水 回归给出，回归只在保留流域上拟合，
      留出流域零径流观测。这才是与 4H_physical 配置公平的基线；用实测口径的
      基线去比 physical 的双头，等于让基线多拿了一份模型没有的信息。

    水位在两种口径下都用实测统计量：留出情景的前提就是该流域有水位记录。
    """
    from pipeline.dataset import TargetScaling
    from pipeline.masking import get_hidden

    wl_runs = available_waterlevel_runs(max_wl_seeds)
    if not wl_runs:
        raise FileNotFoundError(
            f"未找到可复用的单任务水位模型权重（{RUNS_DIR}/{WL_RUN_PREFIX}*/model.pt）")

    predictions = {seed: predict_waterlevel(prepared, ckpt, seq_length)
                   for seed, ckpt in wl_runs}

    collected = {}
    for mask_seed in mask_seeds:
        _, stats = get_hidden(prepared, {"flow": ratio},
                             mechanism="basin_holdout", mask_seed=mask_seed)
        held = set(stats["per_task"]["flow"]["held_out_basins"])
        kept = [b for b in prepared.splits["splits"] if b not in held]
        # 物理尺度必须按掩膜种子重新拟合：哪些流域算"有资料"随留出集变化
        q_norm = (TargetScaling(prepared, scaling, fit_basins=kept).normalized["flow"]
                  if scaling == "physical" else prepared.targets["flow"])
        rating = _global_rating(prepared, kept, q_norm)
        if rating is None:
            continue

        for seed, pred in predictions.items():
            for basin in held:
                bi = prepared.basin_index[basin]
                sel = pred["basin_idx"] == bi
                if sel.sum() < 500:
                    continue
                pos = pred["target_pos"][sel]
                # NSE 对逐流域仿射变换不变，在归一化空间算与原始流量空间一致
                q_obs = q_norm[bi, pos].astype(float)
                q_hat = apply_rating(rating, pred["pred_h"][sel].astype(float))
                score = nse(q_obs, q_hat)
                if np.isfinite(score):
                    collected.setdefault(basin, []).append(score)

    return pd.DataFrame([{"basin": b, "nse": float(np.mean(v)), "n_estimates": len(v)}
                         for b, v in collected.items()])


def compare_with_models(two_stage: pd.DataFrame, ratio: float,
                        config: str = "base") -> pd.DataFrame:
    """与双头、单任务 Q 在同一批被留出流域上做逐流域配对检验。

    必须限定可比配置：4H_physical 与 4J_best 也包含 q_holdout30/50/70，只按
    scenario 过滤会把三种配置的 dual_head 一起对上率定曲线基线，逐流域配对
    立刻出现重复索引。
    """
    from scipy import stats as sps

    metrics, _ = load_summary()
    scenario = f"q_holdout{int(ratio * 100)}"
    sub = select(metrics, config=config, scenario=scenario, task="flow", held_out=True)

    ts = two_stage.set_index("basin")["nse"]
    rows = []
    for arch in ("single_flow", "dual_head"):
        model = sub[sub["architecture"] == arch].groupby("basin")["nse"].mean()
        idx = ts.index.intersection(model.index)
        if len(idx) < 5:
            continue
        diff = (model.loc[idx] - ts.loc[idx]).dropna()
        nz = diff[diff != 0]
        p = float(sps.wilcoxon(nz).pvalue) if len(nz) >= 3 else float("nan")
        rows.append({
            "config": config, "scenario": scenario, "model": arch, "n_basins": len(idx),
            "two_stage_nse": float(ts.loc[idx].mean()),
            "model_nse": float(model.loc[idx].mean()),
            "diff_model_minus_two_stage": float(diff.mean()),
            "wilcoxon_p": p,
            "n_model_better": int((diff > 0).sum()),
            "n_two_stage_better": int((diff < 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    from pipeline.dataset import PreparedData

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    prep = PreparedData()
    print(f"复用的单任务水位模型: {[s for s, _ in available_waterlevel_runs()]}")

    metrics, _ = load_summary()
    configs = [c for c in ("base", "physical") if c in set(metrics["config"])]

    all_cmp = []
    for config in configs:
        # base 配置对 observed 口径的基线；physical 配置对 physical 口径的基线
        scaling = "physical" if config == "physical" else "observed"
        for ratio in (0.30, 0.50, 0.70):
            ts = two_stage_scores(prep, ratio, scaling=scaling)
            suffix = "" if scaling == "observed" else "_physical"
            path = SUMMARY_DIR / f"two_stage_holdout{int(ratio * 100)}{suffix}.csv"
            ts.to_csv(path, index=False, encoding="utf-8-sig")
            cmp = compare_with_models(ts, ratio, config=config)
            cmp["rating_scaling"] = scaling
            all_cmp.append(cmp)
            print()
            print(f"[{config} 配置 / {scaling} 口径] 留出 {ratio:.0%}："
                  f"两阶段基线 n={len(ts)}  均值 NSE {ts.nse.mean():.4f}")
            if not cmp.empty:
                print(cmp[["model", "two_stage_nse", "model_nse",
                           "diff_model_minus_two_stage", "wilcoxon_p",
                           "n_model_better", "n_two_stage_better"]].round(4).to_string(index=False))

    if all_cmp:
        merged = pd.concat(all_cmp, ignore_index=True)
        out = SUMMARY_DIR / "two_stage_comparison.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
