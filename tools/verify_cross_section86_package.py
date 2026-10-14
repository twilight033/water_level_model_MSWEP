"""解压独立副本，验证来源哈希、白名单与两阶段短运行。"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="验证原86一键包的独立运行")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.extract_dir.resolve()
    if target.exists():
        raise FileExistsError("独立验证目录必须是新目录，避免覆盖已有结果")
    target.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError("压缩包路径超出验证目录")
        archive.extractall(target)
    root = target / "cross_section86_oneclick"
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if digest(root / relative) != expected:
            raise AssertionError(f"打包文件哈希不一致: {relative}")
    profiles = list((root / "data/cross_sections/profiles").glob("WY*/*/*.json"))
    for path in profiles:
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in ("NWIS_Discharge", "Velocity", "NWIS_Area", "GageHeight")):
            raise AssertionError(f"断面文件残留禁止的测流字段: {path}")
    print(f"独立副本哈希通过，剖面白名单检查{len(profiles)}次。", flush=True)
    commands = [
        [sys.executable, "-X", "utf8", "-m", "unittest", "discover",
         "-s", "tests", "-p", "test_cross_section86.py", "-v"],
        [sys.executable, "-u", "-X", "utf8", "src/cross_section_experiment/runner.py",
         "--smoke", "--device", "cpu"],
        # 第二次相同调用验证续跑：不能重新拟合已完成任务。
        [sys.executable, "-u", "-X", "utf8", "src/cross_section_experiment/runner.py",
         "--smoke", "--device", "cpu"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    report = {
        "archive": str(args.archive.resolve()), "sha256": digest(args.archive),
        "n_profiles_checked": len(profiles), "source_file_hashes_verified": True,
        "standalone_unit_tests": "passed", "standalone_two_stage_smoke": "passed",
        "resume": "passed", "formal_training_executed": False,
    }
    (args.archive.parent / "cross_section86_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
