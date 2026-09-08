"""人工标签缺失掩膜：MCAR 与基于真实故障段长 ECDF 的连续段注入。

相对原 ablation 脚本的修正（回应审稿意见 Major 3 与 Major 9）：

1. **只作用于训练段**。原实现把掩膜施加到整条序列，训练集和验证集一起
   被删，早停可用样本随场景变化，性能下降混了"训练监督减少"和"模型
   选择变差"两个原因。这里按冻结划分裁剪，验证段与测试段始终保持原始
   完整标签。
2. **比例基于原始有效标签**。分母是该流域该任务在训练段内原本有效的
   标签数，不含原本就缺失的位置。
3. **记录实际达成比例**，而不是只报告目标比例。连续段注入时最多只对
   每个流域每个任务的最后一段做尾部截断以对齐预算，截断步数如实记录。
4. **记录段起点的月份分布**，用于回答"均匀抽样是否保持了真实故障季节性"。
5. **掩膜种子独立于模型种子**，本模块只接受 mask_seed。
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.loaders import TASKS  # noqa: E402

STEP_HOURS = 3
FAULT_STATS_DIR = _ROOT / "src" / "missing_period" / "output"
FAULT_CSV = {"flow": FAULT_STATS_DIR / "fault_recovery_Q.csv",
             "waterlevel": FAULT_STATS_DIR / "fault_recovery_H.csv"}
MECHANISMS = ("mcar", "segment", "basin_holdout")


def _subseed(mask_seed: int, task: str, basin: str) -> int:
    """跨进程稳定的子种子。zlib.crc32 无随机盐，内置 hash() 有。"""
    key = f"{mask_seed}|{task}|{basin}".encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


def load_fault_lengths_steps(task: str) -> np.ndarray:
    """读取真实故障段长度（小时）并换算为 3 小时步数，用于 bootstrap 采样。"""
    path = FAULT_CSV[task]
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "fault_lengths_hours" not in df.columns:
        raise ValueError(f"{path} 缺少 fault_lengths_hours 列")
    pieces = []
    for row in df["fault_lengths_hours"].dropna():
        parts = [int(x) for x in str(row).split(";") if x.strip()]
        if parts:
            pieces.append(np.asarray(parts, dtype=np.int64))
    if not pieces:
        raise ValueError(f"{path} 中 fault_lengths_hours 全为空")
    hours = np.concatenate(pieces)
    steps = np.maximum(1, np.rint(hours / STEP_HOURS).astype(np.int64))
    return steps


def _train_slice(prepared, basin: str) -> tuple:
    """该流域训练段目标时刻的整数位置闭区间。"""
    return prepared.split_range(basin, "train")


def _hide_mcar(rng, valid_pos: np.ndarray, budget: int) -> tuple:
    """在有效位置中均匀随机选取 budget 个隐藏。"""
    chosen = rng.choice(valid_pos, size=budget, replace=False)
    # MCAR 逐点删除，没有"段"的概念，段数记 0 以免与连续段机制混淆
    return np.sort(chosen), {"n_segments": 0, "n_trimmed_steps": 0,
                             "start_months": []}


def real_gap_start_months(prepared, task: str) -> np.ndarray:
    """统计训练段内**真实**缺口的起始月份分布。

    故障统计 CSV 只保留了段长度、没有保留起始时刻，因此段长 ECDF 无法
    携带季节信息。这里直接从原始序列里数真实缺口的起始月份，用作
    "均匀抽样是否保持季节性"（Major 9）的实测对照。
    """
    counts = np.zeros(12, dtype=np.int64)
    for basin in prepared.splits["splits"]:
        bi = prepared.basin_index[basin]
        lo, hi = _train_slice(prepared, basin)
        if hi < lo:
            continue
        missing = np.isnan(prepared.targets[task][bi, lo:hi + 1])
        if not missing.any():
            continue
        # 缺口起点 = 由"有效"翻转为"缺失"的位置
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        for s in starts:
            counts[prepared.grid[lo + int(s)].month - 1] += 1
    return counts


def _hide_segments(rng, valid_pos: np.ndarray, budget: int,
                   lo: int, hi: int, lengths_steps: np.ndarray,
                   grid: pd.DatetimeIndex, max_attempts_factor: int = 50,
                   month_probs: np.ndarray = None) -> tuple:
    """从真实段长 ECDF 采样，在训练段内放置连续缺失段。

    段可能覆盖原本就缺失的位置，也可能与已放置的段重叠；只有"本次新隐藏
    的原始有效标签"才计入预算，避免名义比例与实际比例不符。

    month_probs 给定时，段起点先按该月份分布抽月、再在该月内均匀取位置。
    实测真实径流缺口有 68.6% 起始于 12–2 月（结冰期），均匀抽样与之总变差
    距离达 0.547，因此默认按真实月份分布抽样（见 real_gap_start_months）。
    """
    span = hi - lo + 1
    valid_set = np.zeros(span, dtype=bool)
    valid_set[valid_pos - lo] = True
    hidden_local = np.zeros(span, dtype=bool)

    local_months = grid[lo:hi + 1].month.to_numpy()
    month_pos = [np.flatnonzero(local_months == m + 1) for m in range(12)]
    if month_probs is not None:
        usable = np.array([p.size > 0 for p in month_pos])
        probs = np.where(usable, month_probs, 0.0)
        probs = probs / probs.sum() if probs.sum() > 0 else None
    else:
        probs = None

    n_hidden = 0
    n_segments = 0
    n_trimmed = 0
    start_months = []
    max_attempts = max(1000, max_attempts_factor * max(1, budget // max(1, int(lengths_steps.mean()))))
    attempts = 0

    while n_hidden < budget and attempts < max_attempts:
        attempts += 1
        seg_len = int(rng.choice(lengths_steps))
        seg_len = min(seg_len, span)
        if probs is None:
            start = int(rng.integers(0, span - seg_len + 1))
        else:
            month = int(rng.choice(12, p=probs))
            start = int(rng.choice(month_pos[month]))
            start = min(start, span - seg_len)
        end = start + seg_len

        newly = valid_set[start:end] & ~hidden_local[start:end]
        gain = int(newly.sum())
        if gain == 0:
            continue

        if n_hidden + gain > budget:
            # 只对最后一段做尾部截断以对齐预算，截断量如实记录
            need = budget - n_hidden
            cum = np.cumsum(newly)
            cut = int(np.searchsorted(cum, need, side="left")) + 1
            n_trimmed += (end - start) - cut
            end = start + cut
            newly = valid_set[start:end] & ~hidden_local[start:end]
            gain = int(newly.sum())

        hidden_local[start:end] |= valid_set[start:end]
        n_hidden += gain
        n_segments += 1
        start_months.append(int(grid[lo + start].month))

    positions = np.flatnonzero(hidden_local) + lo
    positions = positions[np.isin(positions, valid_pos)]
    return positions, {"n_segments": n_segments, "n_trimmed_steps": int(n_trimmed),
                       "start_months": start_months, "attempts": attempts}


def _build_basin_holdout(prepared, active: dict, hidden: dict,
                         mask_seed: int, ratios: dict) -> tuple:
    """按流域留出：抽取一部分流域，把它们该任务的训练标签**整段**删除。

    这是"某些流域根本没有径流观测、只有廉价的水位观测"这一现实情形的
    直接模拟，与在所有流域上均匀稀疏化是两种不同的缺失结构。ratio 在此
    表示**被留出的流域比例**，而不是标签比例。

    关于归一化的一个明确假设
    ------------------------
    目标仍按逐流域的观测均值与标准差归一化，而这两个统计量对一个"完全
    没有径流观测"的流域本来是拿不到的。因此本设定实际对应的是"知道量级、
    缺少连续时序"——例如只有少数几次实测流量、足以定出均值与变幅，但没有
    连续记录。论文中必须如实这样表述，不能说成完全无资料流域。
    """
    basins = list(prepared.splits["splits"])
    rows, held = [], {}
    for task, ratio in active.items():
        rng = np.random.default_rng(_subseed(mask_seed, f"holdout_{task}", "all"))
        n_hold = int(round(len(basins) * ratio))
        chosen = sorted(rng.choice(len(basins), size=n_hold, replace=False).tolist())
        held[task] = [basins[i] for i in chosen]

        for basin in basins:
            bi = prepared.basin_index[basin]
            lo, hi = _train_slice(prepared, basin)
            if hi < lo:
                continue
            window = prepared.targets[task][bi, lo:hi + 1]
            valid_pos = np.flatnonzero(~np.isnan(window)) + lo
            is_held = basin in held[task]
            if is_held:
                hidden[task][bi, valid_pos] = True
            rows.append({
                "task": task, "basin": basin, "mechanism": "basin_holdout",
                "target_ratio": ratio, "n_valid_train": int(valid_pos.size),
                "n_hidden": int(valid_pos.size) if is_held else 0,
                "realized_ratio": 1.0 if is_held else 0.0,
                "held_out": bool(is_held),
                "n_segments": 0, "n_trimmed_steps": 0,
            })

    per_task = {}
    for task, ratio in active.items():
        sub = [r for r in rows if r["task"] == task]
        per_task[task] = {
            "target_ratio": ratio,
            "n_basins_held_out": len(held[task]),
            "n_basins_total": len(basins),
            "realized_basin_ratio": len(held[task]) / len(basins),
            "n_hidden_total": int(sum(r["n_hidden"] for r in sub)),
            "n_valid_train_total": int(sum(r["n_valid_train"] for r in sub)),
            "held_out_basins": held[task],
        }
    stats = {
        "meta": {"mechanism": "basin_holdout", "mask_seed": mask_seed,
                 "ratios": dict(ratios),
                 "scope": "仅训练段；验证与测试保持原始完整",
                 "note": "ratio 表示被留出的流域比例；归一化仍用观测统计量，"
                         "对应'知道量级、缺少连续时序'的设定"},
        "per_task": per_task,
        "per_basin": rows,
    }
    return hidden, stats


def build_hidden(prepared, ratios: dict, mechanism: str = "segment",
                 mask_seed: int = 42, seasonal: bool = True) -> tuple:
    """构造训练段人工缺失掩膜。

    Parameters
    ----------
    prepared : PreparedData
    ratios : dict
        {"flow": 0.5, "waterlevel": 0.0} 形式的目标缺失比例。
    mechanism : {"mcar", "segment"}
    mask_seed : int
        仅控制掩膜，独立于模型种子。
    seasonal : bool
        仅对 segment 机制有效。True 时段起点按该任务真实缺口的月份分布抽样，
        使人工缺失同时匹配真实的段长分布与季节时机；False 时起点均匀抽样，
        作为"季节性是否重要"的对照。

    Returns
    -------
    (hidden, stats)
        hidden: {task: bool 数组 [n_basin, n_time]}，True 表示该标签被删除；
                比例为 0 的任务不出现在字典中。
        stats:  逐流域逐任务的实际缺失比例、段数、截断步数、起点月份分布。
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism 必须是 {MECHANISMS} 之一，收到 {mechanism!r}")

    active = {t: r for t, r in ratios.items() if r and r > 0}
    if not active:
        return {}, {"meta": {"mechanism": mechanism, "mask_seed": mask_seed,
                             "ratios": dict(ratios), "note": "无人工缺失"},
                    "per_basin": []}

    n_basin, n_time = prepared.targets[TASKS[0]].shape
    hidden = {t: np.zeros((n_basin, n_time), dtype=bool) for t in active}

    if mechanism == "basin_holdout":
        return _build_basin_holdout(prepared, active, hidden, mask_seed, ratios)

    lengths, month_probs = {}, {}
    if mechanism == "segment":
        for t in active:
            lengths[t] = load_fault_lengths_steps(t)
            if seasonal:
                real = real_gap_start_months(prepared, t).astype("float64")
                month_probs[t] = real / real.sum() if real.sum() > 0 else None
            else:
                month_probs[t] = None

    rows = []
    for task, ratio in active.items():
        for basin in prepared.splits["splits"]:
            bi = prepared.basin_index[basin]
            lo, hi = _train_slice(prepared, basin)
            if hi < lo:
                continue
            # 每个流域每个任务使用独立子种子。必须用 crc32 而不是内置
            # hash()：CPython 对 str 的 hash 带每进程随机盐，同一 mask_seed
            # 在不同进程会生成不同掩膜，而单任务与双头模型是分别在各自进程
            # 里训练的，配对比较要求它们看到完全相同的掩膜。
            rng = np.random.default_rng(_subseed(mask_seed, task, basin))

            window = prepared.targets[task][bi, lo:hi + 1]
            valid_pos = np.flatnonzero(~np.isnan(window)) + lo
            n_valid = int(valid_pos.size)
            if n_valid == 0:
                continue

            budget = int(round(n_valid * ratio))
            if budget <= 0:
                continue

            if mechanism == "mcar":
                pos, info = _hide_mcar(rng, valid_pos, budget)
            else:
                pos, info = _hide_segments(rng, valid_pos, budget, lo, hi,
                                           lengths[task], prepared.grid,
                                           month_probs=month_probs[task])

            hidden[task][bi, pos] = True
            n_hidden = int(pos.size)
            rows.append({
                "task": task, "basin": basin, "mechanism": mechanism,
                "target_ratio": ratio, "n_valid_train": n_valid,
                "n_hidden": n_hidden,
                "realized_ratio": n_hidden / n_valid,
                "n_segments": info["n_segments"],
                "n_trimmed_steps": info["n_trimmed_steps"],
                # 逐流域的完整起点列表只用于汇总直方图，不写进缓存
                "_start_months": info.get("start_months", []),
            })

    per_task = {}
    for task in active:
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        realized = np.array([r["realized_ratio"] for r in sub])
        months = np.concatenate([np.asarray(r["_start_months"], dtype=int) for r in sub]) \
            if any(r["_start_months"] for r in sub) else np.array([], dtype=int)
        month_hist = (np.bincount(months, minlength=13)[1:].tolist()
                      if months.size else [])
        per_task[task] = {
            "target_ratio": active[task],
            "realized_ratio_mean": float(realized.mean()),
            "realized_ratio_min": float(realized.min()),
            "realized_ratio_max": float(realized.max()),
            "n_hidden_total": int(sum(r["n_hidden"] for r in sub)),
            "n_valid_train_total": int(sum(r["n_valid_train"] for r in sub)),
            "n_segments_total": int(sum(r["n_segments"] for r in sub)),
            "n_trimmed_steps_total": int(sum(r["n_trimmed_steps"] for r in sub)),
            "start_month_hist": month_hist,
        }

    # 汇总完成后丢弃逐流域的起点明细，缓存 JSON 只保留可读的统计量
    for row in rows:
        row.pop("_start_months", None)

    stats = {
        "meta": {"mechanism": mechanism, "mask_seed": mask_seed,
                 "ratios": dict(ratios), "seasonal": bool(seasonal),
                 "scope": "仅训练段；验证与测试保持原始完整"},
        "per_task": per_task,
        "per_basin": rows,
    }
    return hidden, stats


MASK_CACHE_DIR = _ROOT / "data" / "splits" / "masks"


def _mask_key(ratios: dict, mechanism: str, mask_seed: int, seasonal: bool) -> str:
    parts = "_".join(f"{t}{ratios.get(t, 0.0):g}" for t in TASKS)
    tag = "seasonal" if (mechanism == "segment" and seasonal) else "uniform"
    return f"{mechanism}_{tag}_{parts}_seed{mask_seed}"


def get_hidden(prepared, ratios: dict, mechanism: str = "segment",
               mask_seed: int = 42, seasonal: bool = True,
               use_cache: bool = True) -> tuple:
    """带磁盘缓存的掩膜获取。

    同一场景下的单任务模型与双头模型必须使用**完全相同**的掩膜，否则
    配对比较不成立。缓存让这一点从"依赖随机数可复现"变成"读同一个文件"，
    同时省掉连续段注入的 30–50 秒生成开销。
    """
    import json

    key = _mask_key(ratios, mechanism, mask_seed, seasonal)
    npz_path = MASK_CACHE_DIR / f"{key}.npz"
    json_path = MASK_CACHE_DIR / f"{key}.stats.json"

    if use_cache and npz_path.exists() and json_path.exists():
        with np.load(npz_path) as data:
            hidden = {t: data[t] for t in data.files}
        stats = json.loads(json_path.read_text(encoding="utf-8"))
        return hidden, stats

    hidden, stats = build_hidden(prepared, ratios, mechanism=mechanism,
                                 mask_seed=mask_seed, seasonal=seasonal)
    assert_train_only(prepared, hidden)
    if use_cache:
        MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **hidden)
        json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return hidden, stats


def assert_train_only(prepared, hidden: dict) -> None:
    """断言掩膜没有触碰验证段与测试段，供测试与运行前自检使用。"""
    for task, arr in hidden.items():
        for basin in prepared.splits["splits"]:
            bi = prepared.basin_index[basin]
            lo, hi = _train_slice(prepared, basin)
            outside = arr[bi].copy()
            outside[lo:hi + 1] = False
            if outside.any():
                raise AssertionError(
                    f"掩膜越界: 任务 {task} 流域 {basin} 在训练段之外隐藏了 "
                    f"{int(outside.sum())} 个标签")
