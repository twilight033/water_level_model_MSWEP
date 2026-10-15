"""用历史权重重算原测试H，核对旧强配置的输入与归一化是否兼容。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.inference import infer
from cross_section_experiment.protocol import write_json
from cross_section_experiment.reference import audit_references, reference_folder
from cross_section_experiment.runner import make_train_config
from pipeline.dataset import PreparedData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_section86_strong_physical.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    p = PreparedData(attr_set="extended")
    audit = audit_references(ROOT, p, cfg, make_train_config)
    results = []
    # 原训练没有过滤时间缺口，此处先按原时刻复现；正式新实验的推理仍过滤缺口。
    for seed in cfg["model_seeds"]:
        folder = reference_folder(ROOT, cfg, "single_waterlevel", seed)
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
        prediction = infer(p, checkpoint["state_dict"], checkpoint["model_config"], "test", 480, 1,
                           args.device)
        old = pd.read_csv(folder / "metrics.csv", dtype={"basin": str}).set_index("basin")
        indices, positions = prediction["basin_idx"], prediction["target_pos"]
        normalized = prediction["pred"]["waterlevel"]
        physical = normalized * p.target_std["waterlevel"][indices] + p.target_mean["waterlevel"][indices]
        bounds = np.r_[0, np.cumsum(np.bincount(indices, minlength=len(p.basins)))]
        rows = []
        for bi, basin in enumerate(p.basins):
            selected = slice(bounds[bi], bounds[bi + 1])
            obs, pred = p.targets_raw["waterlevel"][bi, positions[selected]], physical[selected]
            ok = np.isfinite(obs) & np.isfinite(pred)
            obs, pred = obs[ok], pred[ok]
            nse = float(1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())
            rmse = float(np.sqrt(((pred - obs) ** 2).mean()))
            rows.append({"basin": basin, "n_match": len(obs) == old.loc[basin, "n"],
                         "delta_nse": nse - old.loc[basin, "nse"],
                         "delta_rmse": rmse - old.loc[basin, "rmse"]})
        differences = pd.DataFrame(rows)
        passed = bool(differences.n_match.all() and differences.delta_nse.abs().max() < 5e-5
                      and differences.delta_rmse.abs().max() < 5e-5)
        result = {"model_seed": seed, "passed": passed, "n_basins": len(rows),
                  "max_absolute_nse_difference": float(differences.delta_nse.abs().max()),
                  "max_absolute_rmse_difference": float(differences.delta_rmse.abs().max())}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"reference_model_audit": audit, "H_reproduction": results,
              "validation_and_inference_batch_size": 4096,
              "test_H_used_for_reproduction_only_not_model_selection": True,
              "passed": all(r["passed"] for r in results)}
    write_json(ROOT / f"reports/cross_section86_strong_{cfg.get('target_scaling', 'physical')}_model_check_4096.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
