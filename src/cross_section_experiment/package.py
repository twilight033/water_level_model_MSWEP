"""白名单打包原86缓存与剖面，剔除原始测流Q、流速和率定信息。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.features import file_sha256
from cross_section_experiment.protocol import json_ready, write_json
from cross_section_experiment.runner import DATA_FILES

LAUNCH_FILES = (
    "RUN_CROSS_SECTION86.cmd", "SMOKE_CROSS_SECTION86.cmd",
    "run_cross_section86.ps1", "run_cross_section86.sh",
    "requirements_cross_section86.txt", "configs/cross_section86.json",
    "tests/test_cross_section86.py", "config.py",
)
RECORD_FIELDS = (
    "mmt_ID", "station", "CrossSectionType", "mmt_Timestamp",
    "MeasurementQualityCode", "ChannelNumber", "SectionLocationCode",
    "SectionDistanceMeasure_ft",
)


def sanitized_profile(payload: dict, original_sha256: str) -> dict:
    """仅白名单几何和出处字段进入可移植包，不递归复制record。"""
    record = payload.get("record", {})
    return {
        "station": payload["station"], "water_year": payload.get("water_year"),
        "mmt_ID": payload["mmt_ID"],
        "measurement_date_local": payload.get("measurement_date_local"),
        "measurement_timestamp_raw": payload.get("measurement_timestamp_raw"),
        "record": {key: record.get(key) for key in RECORD_FIELDS},
        "units": {"Distance_ft": "ft", "Depth_ft": "ft"},
        "points": [{key: point.get(key) for key in ("fid", "Distance_ft", "Depth_ft")}
                   for point in payload["points"]],
        "source": {
            "original_json_sha256": original_sha256,
            "original_point_values_sha256": payload.get("source", {}).get("point_values_sha256"),
            "doi": payload.get("source", {}).get("doi"),
        },
    }


def build_package(destination: Path, code_only: bool = False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = "cross_section86_oneclick"
    files = list(LAUNCH_FILES)
    for relative in ("src/cross_section_experiment", "src/pipeline", "src/models"):
        files += [str(p.relative_to(ROOT)).replace("\\", "/")
                  for p in sorted((ROOT / relative).glob("*.py"))]
    files += ["src/training/trainer.py", "src/evaluation/metrics.py"]
    if not code_only:
        files += list(DATA_FILES)
    source_index = ROOT / "research_notes/USGS 原86 历年断面核验/profiles/profile_index.csv"
    index_root = source_index.parent.parent
    manifest = {"format": "original86_cross_section_oneclick_v1", "contains_cached_data": not code_only,
                "contains_formal_trained_weights": False, "profiles_exclude_q_and_velocity": True,
                "files": {}, "original_profile_hashes": {}}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for relative in sorted(set(files)):
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"打包依赖缺失: {path}")
            archive.write(path, f"{prefix}/{relative}")
            manifest["files"][relative] = file_sha256(path)
        documentation = ROOT / "docs/cross_section86_runbook.md"
        archive.write(documentation, f"{prefix}/README.md")
        if not code_only:
            archive.write(source_index, f"{prefix}/data/cross_sections/profiles/profile_index.csv")
            with source_index.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                relative = row["json_path"]
                source = (index_root / relative).resolve()
                if not source.is_relative_to(index_root.resolve()):
                    raise ValueError("剖面路径超出存档根目录")
                original_hash = file_sha256(source)
                payload = json.loads(source.read_text(encoding="utf-8"))
                clean = sanitized_profile(payload, original_hash)
                archive.writestr(
                    f"{prefix}/data/cross_sections/{relative}",
                    json.dumps(json_ready(clean), ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False))
                manifest["original_profile_hashes"][relative] = original_hash
        archive.writestr(f"{prefix}/package_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        failed = archive.testzip()
        if failed:
            raise RuntimeError(f"压缩包CRC检查失败: {failed}")
        count = len(archive.namelist())
    result = {"path": str(destination.resolve()), "bytes": destination.stat().st_size,
              "sha256": file_sha256(destination), "n_files": count,
              "n_profiles": len(manifest["original_profile_hashes"])}
    write_json(destination.with_suffix(".manifest.json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description="打包原86断面实验与原86缓存")
    p.add_argument("--output", type=Path,
                   default=ROOT / "dist/cross_section86_oneclick_20261014.zip")
    p.add_argument("--code-only", action="store_true")
    args = p.parse_args()
    print(json.dumps(build_package(args.output, args.code_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
