"""独立的无 STGQ 候选流域外部训练实验。

原 86 流域不改动；额外加入 13 个新流域（01100561 已在原 86 内）。14 个候选
流域的 Q 在训练和验证期均被屏蔽，H 保留；测试期 Q 仅用于最终评估。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT, ROOT / "src" / "others"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

CANDIDATES = ["01100561","01189000","02322800","02365769","02366996","02407000","03198000","05422600","06843500","06890900","06893620","06893890","07154500","07230500"]
EMBARGO = 480

def args():
    p = argparse.ArgumentParser(description="99 流域：候选流域 Q 屏蔽、H 保留的外部训练")
    p.add_argument("--mswep-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "nonrating14_external_seed1")
    p.add_argument("--model-seeds", nargs="+", type=int, default=[1], help="模型种子；试跑默认 1，正式建议 1 2 3")
    p.add_argument("--mask-seeds", nargs="+", type=int, default=[42], help="随机留出掩膜种子")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preflight", action="store_true", help="只校验99流域数据、掩膜和滑窗，不训练")
    return p.parse_args()

def _frame(ds, var, basins):
    x = ds[var].transpose("time", "basin").to_pandas()
    x.columns = [str(c) for c in x.columns]
    return x.reindex(columns=basins).astype("float32")

def extra_data(basins, mswep_path):
    from config import CAMELSH_DATA_PATH
    from hydrodataset import StandardVariable
    from improved_camelsh_reader import ImprovedCAMELSHReader
    from pipeline.paths import verify_camelsh_path
    header = pd.read_csv(mswep_path, nrows=0).columns.astype(str).tolist()
    miss = [b for b in basins if b not in header]
    if miss: raise ValueError(f"MSWEP 缺少列: {miss}")
    rain = pd.read_csv(mswep_path, usecols=["time", *basins], parse_dates=["time"]).set_index("time").sort_index()
    if rain.index.duplicated().any():
        print(f"MSWEP 重复 {int(rain.index.duplicated().sum())} 行，按原 86 流域规则保留第一条")
        rain = rain[~rain.index.duplicated(keep="first")]
    reader = ImprovedCAMELSHReader(str(verify_camelsh_path(CAMELSH_DATA_PATH)), download=False, use_batch=True)
    tr = reader.camelsh.default_t_range
    forc = reader.read_ts_xrdataset(gage_id_lst=basins, t_range=tr, var_lst=[StandardVariable.TEMPERATURE_MEAN, StandardVariable.SOLAR_RADIATION]).resample(time="3h").mean()
    targ = reader.read_ts_xrdataset(gage_id_lst=basins, t_range=tr, var_lst=[StandardVariable.STREAMFLOW, StandardVariable.WATER_LEVEL])
    return rain.astype("float32"), _frame(forc, StandardVariable.TEMPERATURE_MEAN, basins), _frame(forc, StandardVariable.SOLAR_RADIATION, basins), _frame(targ, StandardVariable.STREAMFLOW, basins), _frame(targ, StandardVariable.WATER_LEVEL, basins)

def candidate_split(grid, q, h):
    union, both = q.notna().to_numpy() | h.notna().to_numpy(), q.notna().to_numpy() & h.notna().to_numpy()
    pos = np.flatnonzero(both)
    if len(pos) < 3 * EMBARGO: raise ValueError("Q/H 共同有效期不足以划分")
    a, z = int(pos[0]), int(pos[-1]); span = z-a+1; m1=a+int(.6*span); m2=a+int(.8*span)
    start = int(np.flatnonzero(union)[0])
    return {"train_start":str(grid[start]),"train_end":str(grid[m1-1]),"valid_start":str(grid[m1+EMBARGO]),"valid_end":str(grid[m2-1]),"test_start":str(grid[m2+EMBARGO]),"test_end":str(grid[z]),"ref_start":str(grid[a]),"ref_end":str(grid[z]),"ref_span_steps":span,"short_test":bool(z-(m2+EMBARGO)+1 < 2920)}

class ExternalPrepared:
    def __init__(self, grid, forcing_raw, attrs_raw, targets_raw, basins, splits):
        from pipeline.dataset import TargetScaling
        self.grid, self.basins, self.splits = grid, basins, {"splits":splits}
        self.basin_index = {b:i for i,b in enumerate(basins)}; self.n_time=len(grid)
        expected_targets = (len(basins), self.n_time)
        expected_forcing = (len(basins), self.n_time)
        if forcing_raw.ndim != 3 or forcing_raw.shape[:2] != expected_forcing:
            raise ValueError(
                "99流域强迫拼接失败：期望前两维为 "
                f"{expected_forcing}，实际为 {forcing_raw.shape}")
        if list(attrs_raw.index.astype(str)) != [str(b) for b in basins]:
            raise ValueError("99流域属性表的行顺序与流域列表不一致")
        for task in ("flow", "waterlevel"):
            if task not in targets_raw or targets_raw[task].shape != expected_targets:
                actual = None if task not in targets_raw else targets_raw[task].shape
                raise ValueError(
                    f"99流域{task}目标拼接失败：期望 {expected_targets}，实际 {actual}")
        if set(splits) != set(basins):
            missing = sorted(set(basins) - set(splits))
            extra = sorted(set(splits) - set(basins))
            raise ValueError(f"划分流域与数据流域不一致：缺少={missing[:5]}，多出={extra[:5]}")
        # 标记共同时间轴中的断点。WindowDataset 会丢弃跨越断点的样本，避免
        # 例如 MSWEP 缺一个 3 小时时刻时把前后数据错误拼成连续 480 步输入。
        self.time_valid = np.r_[True, np.diff(grid.asi8) == pd.Timedelta("3h").value]
        self.targets_raw, self.attrs_raw = targets_raw, attrs_raw
        self.stats = {"flow":{}, "waterlevel":{}, "forcing":{}, "attr":{}}
        for bi,b in enumerate(basins):
            lo,hi=self.split_range(b,"train")
            for task in ("flow","waterlevel"):
                v=targets_raw[task][bi,lo:hi+1]; v=v[np.isfinite(v)]
                self.stats[task][b]={"mean":float(v.mean()) if len(v) else 0.,"std":float(max(v.std(),1e-6)) if len(v) else 1.,"n":len(v)}
        chunks=[]
        for bi,b in enumerate(basins):
            lo,hi=self.split_range(b,"train"); chunks.append(forcing_raw[bi,lo:hi+1])
        allf=np.concatenate(chunks,axis=0); fm=allf.mean(0); fs=np.maximum(allf.std(0),1e-6)
        self.forcing=((forcing_raw-fm)/fs).astype("float32")
        am=attrs_raw.to_numpy("float32").mean(0); ast=np.maximum(attrs_raw.to_numpy("float32").std(0),1e-6)
        self.attrs=((attrs_raw.to_numpy("float32")-am)/ast).astype("float32")
        self.stats["forcing"]={str(i):{"mean":float(fm[i]),"std":float(fs[i])} for i in range(len(fm))}
        self.default_scaling=TargetScaling(self,"observed"); self.targets=self.default_scaling.normalized
    def split_range(self, basin, split):
        x=self.splits["splits"][basin]; a=int(self.grid.searchsorted(pd.Timestamp(x[f"{split}_start"]),"left")); b=int(self.grid.searchsorted(pd.Timestamp(x[f"{split}_end"]),"right"))-1; return a,b
    def make_scaling(self, scaling="observed", fit_basins=None):
        from pipeline.dataset import TargetScaling
        return self.default_scaling if scaling=="observed" else TargetScaling(self,scaling,fit_basins)
    def denormalize(self, task, basin_idx, values, scaling=None): return (scaling or self.default_scaling).denormalize(task,basin_idx,values)

def fixed_mask(prep, basins, include_valid):
    shape=(len(prep.basins),len(prep.grid)); tr={"flow":np.zeros(shape,dtype=bool)}; va={"flow":np.zeros(shape,dtype=bool)}
    for b in basins:
        bi=prep.basin_index[b]
        for split,mask in (("train",tr),("valid",va)):
            if split=="valid" and not include_valid: continue
            lo,hi=prep.split_range(b,split); mask["flow"][bi,lo:hi+1]=np.isfinite(prep.targets_raw["flow"][bi,lo:hi+1])
    return tr,va

def random_basin_holdout(prep, ratio, mask_seed):
    """在99流域本地构造随机整流域 Q 留出掩膜。

    不能复用原86流域实验的掩膜数组；该数组若仍是 86 行，会在第87个
    流域索引时报错。随机数规则与 pipeline.masking.basin_holdout 保持一致。
    """
    from pipeline.masking import _subseed
    rng = np.random.default_rng(_subseed(mask_seed, "holdout_flow", "all"))
    n_hold = int(round(len(prep.basins) * ratio))
    chosen = sorted(rng.choice(len(prep.basins), size=n_hold, replace=False).tolist())
    held = [prep.basins[i] for i in chosen]
    mask = np.zeros((len(prep.basins), len(prep.grid)), dtype=bool)
    rows = []
    for basin in prep.basins:
        bi = prep.basin_index[basin]
        lo, hi = prep.split_range(basin, "train")
        valid = np.isfinite(prep.targets_raw["flow"][bi, lo:hi + 1])
        if basin in held:
            mask[bi, lo:hi + 1] = valid
        rows.append({"task":"flow", "basin":basin, "mechanism":"basin_holdout",
                     "target_ratio":ratio, "n_valid_train":int(valid.sum()),
                     "n_hidden":int(valid.sum()) if basin in held else 0,
                     "realized_ratio":1.0 if basin in held else 0.0,
                     "held_out":basin in held, "n_segments":0, "n_trimmed_steps":0})
    stats = {"meta":{"mechanism":"basin_holdout", "mask_seed":mask_seed,
                     "ratios":{"flow":ratio}, "scope":"仅训练段；验证与测试保持原始完整"},
             "per_task":{"flow":{"target_ratio":ratio, "n_basins_held_out":n_hold,
                                  "n_basins_total":len(prep.basins),
                                  "realized_basin_ratio":n_hold / len(prep.basins),
                                  "held_out_basins":held}},
             "per_basin":rows}
    return {"flow":mask}, stats, held

def assert_mask_shape(mask, prep, label):
    if mask is None:
        return
    expected = (len(prep.basins), len(prep.grid))
    for task, values in mask.items():
        if values.shape != expected:
            raise ValueError(f"{label} 的 {task} 掩膜形状错误：期望 {expected}，实际 {values.shape}")

def select_in_model_controls(attrs, base_basins):
    """从原 86 流域中为每个候选选一个唯一的属性/气候相似对照。

    外部 CAMELSH 全库中的最近邻未必在当前 99 流域数据集中，不能用于固定
    留出。此处只在原 86 流域中匹配，保证对照既有相同气象强迫缓存，也真实
    参与本次训练。候选 01100561 本身在原 86 中，因此从候选池中排除。
    """
    cols = ["area", "p_mean", "p_seasonality", "frac_snow", "aridity", "slope_mean"]
    missing = [c for c in cols if c not in attrs.columns]
    if missing:
        raise ValueError(f"无法匹配原 86 流域对照，属性缺列: {missing}")
    pool = [b for b in base_basins if b not in CANDIDATES]
    data = attrs.loc[pool + CANDIDATES, cols].astype(float)
    z = (data - data.mean()) / data.std(ddof=0).replace(0, 1.0)
    chosen, used = [], set()
    for candidate in CANDIDATES:
        distance = ((z.loc[pool] - z.loc[candidate]) ** 2).sum(axis=1)
        for basin in distance.sort_values().index:
            if basin not in used:
                chosen.append(basin); used.add(basin); break
    if len(chosen) != len(CANDIDATES):
        raise ValueError("原 86 流域中无法选出 14 个唯一匹配对照")
    return chosen

def preflight(prep, controls):
    """不训练地验证最易出错的数据维度、尺度拟合与固定留出滑窗。"""
    from pipeline.dataset import WindowDataset
    from training.trainer import gauged_basins
    random_hidden, _, _ = random_basin_holdout(prep, .30, 42)
    cases = [
        ("完整监督", None, None),
        ("候选流域固定留出", *fixed_mask(prep, CANDIDATES, True)),
        ("原86匹配对照固定留出", *fixed_mask(prep, controls, True)),
        ("随机30%流域留出", random_hidden, None),
    ]
    for label, hidden, hidden_valid in cases:
        assert_mask_shape(hidden, prep, "预检训练")
        assert_mask_shape(hidden_valid, prep, "预检验证")
        fit_basins = gauged_basins(prep, hidden)
        scaling = prep.make_scaling("physical", fit_basins)
        counts = []
        for split, mask in (("train", hidden), ("valid", hidden_valid), ("test", None)):
            ds = WindowDataset(prep, split, 480, 8 if split != "test" else 1,
                               tasks=("flow", "waterlevel"), hidden=mask,
                               scaling=scaling)
            counts.append(f"{split}={len(ds)}")
        print(f"预检通过：{label}；物理尺度拟合流域={len(fit_basins)}；" + "，".join(counts))

def main():
    a=args()
    from pipeline.loaders import load_forcing, load_targets
    from pipeline.splits import load_splits
    from pipeline.attribute_sources import get_attribute_table, build_attribute_table
    from training.trainer import TrainConfig, train_model
    from evaluation.metrics import evaluate_run, aggregate
    base_grid, base_forcing, base_basins=load_forcing(); base_targets=load_targets(base_grid,base_basins); base_splits=load_splits()["splits"]
    extra=[b for b in CANDIDATES if b not in base_basins]
    rain,temp,solar,q,h=extra_data(extra,a.mswep_csv)
    grid=base_grid.intersection(rain.index).intersection(temp.index).intersection(solar.index).intersection(q.index).intersection(h.index).sort_values()
    ix=base_grid.get_indexer(grid); base_f=base_forcing[:,ix,:]
    ext_f=np.stack([rain.reindex(grid,columns=extra).to_numpy("float32").T,temp.reindex(grid,columns=extra).to_numpy("float32").T,solar.reindex(grid,columns=extra).to_numpy("float32").T],axis=-1)
    basins=list(base_basins)+extra
    forcing=np.ascontiguousarray(np.vstack([base_f,ext_f]), dtype="float32")
    targets={
        "flow": np.ascontiguousarray(np.vstack([
            base_targets["flow"][:,ix], q.reindex(grid,columns=extra).to_numpy("float32").T]), dtype="float32"),
        "waterlevel": np.ascontiguousarray(np.vstack([
            base_targets["waterlevel"][:,ix], h.reindex(grid,columns=extra).to_numpy("float32").T]), dtype="float32"),
    }
    if not np.isfinite(forcing).all():
        raise ValueError(f"合并后的气象强迫含 {int((~np.isfinite(forcing)).sum())} 个缺失值，训练输入不允许缺失")
    # 类别属性（地质/土地覆盖）必须在 99 个流域上统一编码；若把 13 个新增
    # 流域单独 one-hot，会让同一类别的列号与原 86 流域不一致。
    base_attr,_=get_attribute_table("extended",basins=base_basins)
    all_attr,_=build_attribute_table("extended",basins)
    all_attr=all_attr.reindex(columns=base_attr.columns,fill_value=0.)
    cached_base=base_attr.reindex(columns=all_attr.columns)
    rebuilt_base=all_attr.reindex(base_basins)
    if not np.allclose(cached_base.to_numpy("float32"),rebuilt_base.to_numpy("float32"),equal_nan=True):
        raise ValueError("99 流域属性重建后与原 86 流域缓存不一致，拒绝混用不同类别编码")
    attrs=all_attr.reindex(basins)
    splits=dict(base_splits)
    # Q/H 原始表是小时级，必须先对齐到模型使用的共同 3 小时时间轴；
    # 否则小时级的位置会被错误地当作 3 小时时间轴索引。
    q_on_grid=q.reindex(grid,columns=extra); h_on_grid=h.reindex(grid,columns=extra)
    for b in extra: splits[b]=candidate_split(grid,q_on_grid[b],h_on_grid[b])
    # 每次训练均重建该对象，禁止 75 次连续运行共享可变的 targets / scaling
    # 状态。targets 的 copy 使某一运行即使意外原地修改数组，也不会污染下一次。
    def make_prepared():
        return ExternalPrepared(
            grid, forcing, attrs,
            {task: values.copy() for task, values in targets.items()}, basins, splits)

    prep=make_prepared()
    controls=select_in_model_controls(attrs,base_basins)
    print(f"99流域数据已组装：forcing={prep.forcing.shape}，Q={prep.targets_raw['flow'].shape}，H={prep.targets_raw['waterlevel'].shape}")
    if a.preflight:
        preflight(prep, controls)
        print("预检完成：未启动训练。")
        return
    a.out_dir.mkdir(parents=True,exist_ok=True); allm=[]; records=[]
    scenarios=[("complete",None,None,False), ("candidate_fixed",CANDIDATES,None,True), ("matched_86_fixed",controls,None,True)]
    for ratio in (.30,.50,.70):
        for xs in a.mask_seeds: scenarios.append((f"random_q_holdout{int(ratio*100)}",None,(ratio,xs),False))
    for name,fixed,random_spec,hide_valid in scenarios:
        arches=("single_flow","dual_head") if name!="complete" else ("single_flow","single_waterlevel","dual_head")
        for arch in arches:
            for seed in a.model_seeds:
                cfg=TrainConfig(architecture=arch,seq_length=480,attr_set="extended",target_scaling="physical",model_seed=seed,device=a.device)
                key=f"{name}_{arch}_ms{seed}"+(f"_xs{random_spec[1]}" if random_spec else ""); out=a.out_dir/key; out.mkdir(exist_ok=True)
                if (out/"run.json").exists() and (out/"metrics.csv").exists():
                    print(f"跳过已完成运行: {key}"); allm.append(pd.read_csv(out/"metrics.csv",dtype={"basin":str})); records.append({"run_key":key,"scenario":name,"architecture":arch,"seed":seed,"status":"skipped"}); continue
                run_prep=make_prepared()
                if random_spec:
                    hidden,mask_stats,held=random_basin_holdout(run_prep,random_spec[0],random_spec[1]); hidden_valid=None
                    fit_basins=[b for b in run_prep.basins if b not in held]
                elif fixed:
                    hidden,hidden_valid=fixed_mask(run_prep,fixed,hide_valid); mask_stats={"fixed_basins":fixed,"scope":"训练与验证 Q 屏蔽，H 保留"}
                    fit_basins=[b for b in run_prep.basins if b not in fixed]
                else:
                    hidden=hidden_valid=None; mask_stats=None; fit_basins=list(run_prep.basins)
                assert_mask_shape(hidden, run_prep, "训练")
                assert_mask_shape(hidden_valid, run_prep, "验证")
                print(f"开始运行: {key}；Q={run_prep.targets_raw['flow'].shape}；物理尺度拟合流域={len(fit_basins)}")
                result=train_model(run_prep,cfg,hidden=hidden,hidden_valid=hidden_valid,fit_basins=fit_basins,verbose=True)
                m,_=evaluate_run(run_prep,result["test_prediction"],result["tasks"],scaling=result["scaling"]); m.insert(0,"scenario",name); m.insert(1,"architecture",arch); m["model_seed"]=seed; m["candidate"]=m.basin.isin(CANDIDATES); m["matched_control"]=m.basin.isin(controls)
                m.to_csv(out/"metrics.csv",index=False,encoding="utf-8-sig"); aggregate(m).to_csv(out/"aggregate.csv",index=False,encoding="utf-8-sig"); torch.save({"state_dict":result["state_dict"],"model_config":result["model_config"],"config":result["config"]},out/"model.pt")
                (out/"run.json").write_text(json.dumps({"scenario":name,"architecture":arch,"seed":seed,"mask":mask_stats,"best_epoch":result["best_epoch"],"best_val_score":result["best_val_score"],"scaling_info":result["scaling_info"]},ensure_ascii=False,indent=2),encoding="utf-8")
                allm.append(m); records.append({"run_key":key,"scenario":name,"architecture":arch,"seed":seed,"best_val_score":result["best_val_score"]})
    pd.concat(allm).to_csv(a.out_dir/"metrics_all.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(records).to_csv(a.out_dir/"runs.csv",index=False,encoding="utf-8-sig")
    (a.out_dir/"experiment.json").write_text(json.dumps({"n_basins":len(basins),"new_basins":extra,"candidate_basins":CANDIDATES,"matched_controls":controls,"mswep_csv":str(a.mswep_csv),"model_seeds":a.model_seeds,"mask_seeds":a.mask_seeds,"target_scaling":"physical"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"完成：{a.out_dir}")
if __name__=="__main__": main()
