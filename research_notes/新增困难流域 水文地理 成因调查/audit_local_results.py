"""只读现成结果；在调查目录输出派生审计表，不触发训练和数据准备。"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DIRS = {
    "nonrating14_full_matrix": "physical",
    "nonrating14_observed_matrix": "observed",
    "nonrating99_random_physical_v2": "physical",
    "nonrating99_random_observed_v2": "observed",
}
frames, records = [], []
for dirname, scaling in DIRS.items():
    folder = ROOT / "results" / dirname
    for runpath in sorted(folder.glob("*/run.json")):
        run = json.loads(runpath.read_text(encoding="utf-8-sig"))
        scenario = run["scenario"]
        if dirname.startswith("nonrating14") and scenario not in ("candidate_fixed", "complete"):
            continue
        mask = run.get("mask") or {}
        ids = mask.get("fixed_basins", mask.get("per_task", {}).get("flow", {}).get("held_out_basins", []))
        ms = mask.get("meta", {}).get("mask_seed", -1)
        if scenario.startswith("random"):
            assert mask["meta"]["protocol"] == "train_valid_q_hidden_v2"
            assert ms in (42, 123, 456) and ids
        df = pd.read_csv(runpath.parent / "metrics.csv", dtype={"basin": str})
        df["scaling"] = scaling
        df["source_dir"] = dirname
        df["run_key"] = runpath.parent.name
        df["mask_seed"] = ms
        df["held_out"] = df.basin.isin(ids)
        df["source_metrics"] = str(runpath.parent / "metrics.csv")
        df["source_run"] = str(runpath)
        df["true_new"] = df.basin.isin(json.loads((folder / "experiment.json").read_text())["new_basins"])
        df["original86"] = ~df["true_new"]
        assert not df.duplicated(["basin", "task"]).any()
        for key in ("mean", "std"):
            inf = run.get("scaling_info", {}).get("flow", {}).get(key, {})
            for field in ("slope", "intercept", "r2", "n_fit"):
                df[f"scale_{key}_{field}"] = inf.get(field, np.nan)
        frames.append(df)
        records.append({"source_dir": dirname, "run_key": runpath.parent.name,
                        "scenario": scenario, "architecture": run["architecture"],
                        "model_seed": run["seed"], "mask_seed": ms,
                        "n_held_out": len(ids), "held_out_ids": " ".join(ids),
                        "scope": mask.get("scope", mask.get("meta", {}).get("scope", "")),
                        "source_run": str(runpath)})

data = pd.concat(frames, ignore_index=True)
keys = ["source_dir", "scaling", "scenario", "architecture", "model_seed", "mask_seed", "basin", "task"]
assert not data.duplicated(keys).any()
# 用已保存的均值、标准差和相关系数分解归一化 MSE，检查恒等式闭合。
data["cv_test"] = data.obs_std / data.obs_mean
data["bias_mse_norm"] = ((data.kge_beta - 1) / data.cv_test) ** 2
data["amplitude_mse_norm"] = (data.kge_alpha - 1) ** 2
data["decorrelation_mse_norm"] = 2 * data.kge_alpha * (1 - data.pearson_r)
data["mse_norm"] = 1 - data.nse
data["closure_error"] = (data.bias_mse_norm + data.amplitude_mse_norm + data.decorrelation_mse_norm - data.mse_norm)
data["bias_share_mse"] = data.bias_mse_norm / data.mse_norm
data["amplitude_share_mse"] = data.amplitude_mse_norm / data.mse_norm
data["decorrelation_share_mse"] = data.decorrelation_mse_norm / data.mse_norm
data["nse_after_mean_alignment_diagnostic"] = data.nse + data.bias_mse_norm
data["affine_fit_nse_upper_diagnostic"] = data.pearson_r ** 2
keep = data[(data.scenario == "complete") | data.held_out].copy()
candidate = keep[keep.candidate].copy()
candidate.to_csv(OUT / "local_candidate_run_audit.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(records).to_csv(OUT / "local_run_manifest.csv", index=False, encoding="utf-8-sig")

gkeys = ["source_dir", "scaling", "scenario", "architecture", "basin", "task"]
agg = candidate.groupby(gkeys, dropna=False).agg(
    n_runs=("nse", "size"), n_masks=("mask_seed", "nunique"), n_model_seeds=("model_seed", "nunique"),
    n_test_min=("n", "min"), n_test_max=("n", "max"),
    nse_mean=("nse", "mean"), nse_sd=("nse", "std"), nse_min=("nse", "min"), nse_max=("nse", "max"),
    r_mean=("pearson_r", "mean"), alpha_mean=("kge_alpha", "mean"), beta_mean=("kge_beta", "mean"),
    q_test_mean=("obs_mean", "mean"), q_test_std=("obs_std", "mean"), cv_test=("cv_test", "mean"),
    bias_mean=("bias_mse_norm", "mean"), amplitude_mean=("amplitude_mse_norm", "mean"),
    decorrelation_mean=("decorrelation_mse_norm", "mean"), mse_norm_mean=("mse_norm", "mean"),
    nse_mean_aligned=("nse_after_mean_alignment_diagnostic", "mean"),
    affine_upper=("affine_fit_nse_upper_diagnostic", "mean"),
).reset_index()
agg["bias_share_pooled_mse"] = agg.bias_mean / agg.mse_norm_mean
agg["amplitude_share_pooled_mse"] = agg.amplitude_mean / agg.mse_norm_mean
agg["decorrelation_share_pooled_mse"] = agg.decorrelation_mean / agg.mse_norm_mean
agg.to_csv(OUT / "local_candidate_summary.csv", index=False, encoding="utf-8-sig")

pairkeys = ["source_dir", "scaling", "scenario", "model_seed", "mask_seed", "basin", "task"]
sing = candidate[candidate.architecture == "single_flow"].set_index(pairkeys)
dual = candidate[candidate.architecture == "dual_head"].set_index(pairkeys)
pairs = sing[["nse", "n"]].join(dual[["nse", "n"]], lsuffix="_single", rsuffix="_dual", how="inner")
assert (pairs.n_single == pairs.n_dual).all()
pairs["delta_dual_minus_single"] = pairs.nse_dual - pairs.nse_single
pairs.reset_index().to_csv(OUT / "local_paired_delta.csv", index=False, encoding="utf-8-sig")

headline = []
for group, frame in keep[keep.task == "flow"].groupby(["source_dir", "scaling", "scenario", "architecture"]):
    for pop, sub in [("candidate14", frame[frame.candidate]), ("true_new13", frame[frame.true_new]), ("original86", frame[frame.original86]), ("all99", frame)]:
        if sub.empty:
            continue
        bm = sub.groupby("basin").nse.mean()
        headline.append(dict(zip(["source_dir", "scaling", "scenario", "architecture"], group),
                             population=pop, n_unique_basins=len(bm), n_run_rows=len(sub),
                             basin_equal_nse_mean=bm.mean(), basin_equal_nse_median=bm.median(),
                             n_basins_negative=int((bm < 0).sum())))
pd.DataFrame(headline).to_csv(OUT / "local_population_summary.csv", index=False, encoding="utf-8-sig")
fixed = agg[(agg.scenario == "candidate_fixed") & (agg.task == "flow")]
cols = ["basin", "scaling", "architecture", "n_runs", "nse_mean", "nse_sd", "r_mean", "alpha_mean", "beta_mean", "bias_share_pooled_mse", "nse_mean_aligned", "affine_upper"]
print(fixed[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nCLOSURE", data.closure_error.abs().max(), "RELATIVE", (data.closure_error.abs()/data.mse_norm).max())
print("RUNS", len(records), "SELECTED_ROWS", len(candidate), "PAIRS", len(pairs))
print("\nRANDOM_HELD_OUT_N")
print(agg[(agg.scenario.str.startswith("random")) & (agg.task == "flow") & (agg.architecture == "dual_head") & (agg.scaling == "physical")][["scenario", "basin", "n_runs", "n_masks", "nse_mean", "r_mean"]].to_string(index=False, float_format=lambda x:f"{x:.4f}"))
