"""独立区域化率定函数：仅用保留流域训练H/Q，预测水位来自单任务H模型。"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def training_pairs(prepared, scaling, fit_basins):
    """只读取显式允许名单的训练段；Q除以尺度std，避免幂律拟合负的z-score。"""
    if not fit_basins or len(set(fit_basins)) != len(fit_basins):
        raise ValueError("率定拟合名单必须非空且唯一")
    if not set(fit_basins).issubset(prepared.basins):
        raise ValueError("率定拟合名单含未知流域")
    hs, qs, counts = [], [], {}
    for basin in fit_basins:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, "train")
        h = scaling.normalized["waterlevel"][bi, lo:hi+1].astype(float)
        q = prepared.targets_raw["flow"][bi, lo:hi+1].astype(float) / scaling.std["flow"][bi]
        ok = np.isfinite(h) & np.isfinite(q)
        counts[basin] = int(ok.sum())
        hs.append(h[ok]); qs.append(q[ok])
    h, q = np.concatenate(hs), np.concatenate(qs)
    if len(h) < 20 or np.ptp(h) <= 1e-8:
        raise ValueError("率定函数的保留流域训练配对不足或水位无变化")
    return h, q, counts


def fit_functions(h, q, n_bins=50):
    """固定50分位箱+加权PAVA；幂律有界最小二乘，不用验证或测试调参。"""
    from scipy.optimize import least_squares
    h, q = np.asarray(h, float), np.asarray(q, float)
    ok = np.isfinite(h) & np.isfinite(q); h, q = h[ok], q[ok]
    if len(h) < 20 or np.ptp(h) <= 1e-8:
        raise ValueError("率定训练配对不足或水位无变化")
    edges = np.unique(np.quantile(h, np.linspace(0, 1, n_bins+1)))
    ix = np.clip(np.searchsorted(edges, h, "right")-1, 0, len(edges)-2)
    knots, levels, weights = [], [], []
    for k in range(len(edges)-1):
        sel = ix == k
        if sel.any():
            knots.append(float(h[sel].mean())); levels.append(float(q[sel].mean())); weights.append(int(sel.sum()))
    # 相邻违背单调的块合并；再在原节点上做连续线性插值。
    blocks = []
    for i, (y, w) in enumerate(zip(levels, weights)):
        blocks.append([i, i, y*w, w])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right=blocks.pop(); left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    fitted=np.empty(len(knots))
    for start,end,total,weight in blocks: fitted[start:end+1]=total/weight
    if len(knots)<2: raise ValueError("分段率定至少需要两个不同水位节点")
    piece={"kind":"piecewise","h":knots,"q":fitted.tolist(),"n_train_pairs":len(h),"extrapolation":"linear_endpoint_slopes"}
    # 非负幂律不能表示真实倒流：明确记录排除的负Q训练配对，而非裁剪测试真值。
    positive=q>=0
    if positive.sum()<20: raise ValueError("幂律非负Q训练配对不足")
    hp,qp=h[positive],q[positive]
    # 确定性抽样控制CPU开销；采样位置只来自训练数组。
    positions=np.linspace(0,len(hp)-1,min(len(hp),100000),dtype=int)
    hp,qp=hp[positions],qp[positions]
    span=max(float(np.ptp(hp)),1e-3); low=float(hp.min())
    h0=low-.1*span; b=1.5; a=max(float(qp.mean()/np.mean((hp-h0)**b)),1e-8)
    upper_h0=low-1e-6*span
    def residual(p): return p[0]*np.maximum(hp-p[1],0.)**p[2]-qp
    fit=least_squares(residual,[min(a,1e6),h0,b],bounds=([0.,low-10*span,.1],[1e8,upper_h0,5.]),max_nfev=1000)
    if not fit.success: raise RuntimeError(f"幂律率定拟合未收敛: {fit.message}")
    power={"kind":"power","a":float(fit.x[0]),"h0":float(fit.x[1]),"b":float(fit.x[2]),
           "n_train_pairs":len(h),"n_nonnegative_pairs":int(positive.sum()),"n_optimization_pairs":len(hp),
           "n_negative_pairs_excluded":int((~positive).sum()),"bounds":{"b":[.1,5.]},"converged":True}
    return {"piecewise":piece,"power":power}


def apply_function(model, h):
    h=np.asarray(h,float)
    original_shape=h.shape
    h=h.reshape(-1)
    if model["kind"]=="power":
        return (model["a"]*np.maximum(h-model["h0"],0.)**model["b"]).reshape(original_shape)
    x,y=np.asarray(model["h"]),np.asarray(model["q"])
    out=np.interp(h,x,y)
    for end,next_end,sel in ((0,1,h<x[0]),(-1,-2,h>x[-1])):
        slope=(y[end]-y[next_end])/(x[end]-x[next_end])
        out[sel]=y[end]+slope*(h[sel]-x[end])
    return out.reshape(original_shape)


def water_predictions(prepared,cfg,root,cache):
    """复用同预处理的单任务H检查点；在Q/H并集测试窗口预测，不借用测试H。"""
    import torch
    from models.lstm_models import build_model
    from pipeline.dataset import WindowDataset, make_loader
    from training.trainer import predict
    from evaluation.train_nonrating14_external import PREPROCESSING_PROTOCOL
    key=cfg.model_seed
    if key in cache: return cache[key]
    folder=Path(root)/f"complete_single_waterlevel_ms{key}"
    checkpoint=folder/"model.pt"
    record=folder/"run.json"
    if not checkpoint.exists():
        raise FileNotFoundError("率定对照需要本次恢复预处理的完整单任务H检查点，请完整运行矩阵，不使用--random-only")
    if json.loads(record.read_text(encoding="utf-8"))["preprocessing"]["protocol"] != PREPROCESSING_PROTOCOL:
        raise ValueError("单任务H检查点预处理版本不一致")
    payload=torch.load(checkpoint,map_location=cfg.device,weights_only=False)
    hcfg=replace(cfg,architecture="single_waterlevel")
    model=build_model(hcfg.architecture,forcing_size=prepared.forcing.shape[-1],attr_size=prepared.attrs.shape[-1],
                      hidden_size=hcfg.hidden_size,dropout_rate=hcfg.dropout_rate,proj_size=hcfg.proj_size).to(cfg.device)
    model.load_state_dict(payload["state_dict"])
    scaling=prepared.make_scaling("observed")
    ds=WindowDataset(prepared,"test",cfg.seq_length,cfg.window_step_test,tasks=("waterlevel","flow"),scaling=scaling)
    result=predict(model,make_loader(ds,4096,shuffle=False),("waterlevel",),cfg.device)
    cache[key]=result
    return result


def run_baselines(prepared,cfg,root,run_dir,fit_basins,cache):
    """每个双头运行旁保存两种独立基线、训练名单和可审计的函数参数。"""
    from evaluation.metrics import evaluate_run
    scaling=prepared.make_scaling(cfg.target_scaling,fit_basins)
    h,q,counts=training_pairs(prepared,scaling,fit_basins)
    models=fit_functions(h,q)
    water=water_predictions(prepared,cfg,root,cache)
    bi=water["basin_idx"].astype(int); pos=water["target_pos"].astype(int)
    obs=prepared.targets_raw["flow"][bi,pos]
    qstd=scaling.std["flow"][bi]; qmean=scaling.mean["flow"][bi]
    for name,model in models.items():
        folder=Path(run_dir)/"rating_baselines"/name; folder.mkdir(parents=True,exist_ok=True)
        pred=apply_function(model,water["pred"]["waterlevel"])
        result={"basin_idx":bi,"target_pos":pos,"pred":{"flow":(pred-qmean/qstd).astype("float32")},
                "obs":{"flow":((obs-qmean)/qstd).astype("float32")},"mask":{"flow":np.isfinite(obs).astype("float32")}}
        metrics,series=evaluate_run(prepared,result,["flow"],scaling=scaling)
        reference=Path(run_dir)/"metrics.csv"
        if reference.exists():
            import pandas as pd
            target=pd.read_csv(reference,dtype={"basin":str}).query("task == 'flow'").set_index("basin")["n"]
            actual=metrics.set_index("basin")["n"].reindex(target.index)
            if not np.array_equal(actual.to_numpy(),target.to_numpy()):
                raise ValueError("率定与双头的有效Q测试样本不一致，拒绝不公平比较")
        metrics["architecture"]=f"rating_{name}"; metrics["model_seed"]=cfg.model_seed
        metrics["held_out"]=~metrics.basin.isin(fit_basins)
        record=json.loads((Path(run_dir)/"run.json").read_text(encoding="utf-8"))
        metrics["scenario"]=record["scenario"]
        metrics["run_key"]=Path(run_dir).name
        metrics["target_scaling"]=cfg.target_scaling
        metrics.to_csv(folder/"metrics.csv",index=False,encoding="utf-8-sig")
        np.savez_compressed(folder/"predictions.npz",basin_idx=bi,target_pos=pos,
                            predicted_h_z=water["pred"]["waterlevel"],predicted_q=pred*qstd,observed_q=obs)
        held=set(prepared.basins)-set(fit_basins)
        if held.intersection(counts): raise AssertionError("率定拟合名单泄漏")
        manifest={"fit_split":"train","fit_basins":list(fit_basins),"held_out_basins":sorted(held),
                  "pair_counts":counts,"h_coordinate":"training_H_zscore","q_coordinate":"Q/divisor_std",
                  "scaling":cfg.target_scaling,"function":model,"water_model":"single_waterlevel",
                  "validation_used_for_curve":False,"test_used_for_curve":False,
                  "preprocessing":record["preprocessing"],"scenario":record["scenario"],
                  "run_key":Path(run_dir).name,
                  "interpretation":"区域化迁移函数，不是逐站官方率定曲线；observed使用留出Q尺度，仅作诊断"}
        (folder/"fit.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
