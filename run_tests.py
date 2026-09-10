"""跑全部测试并打印清晰摘要。

用法：

    python -X utf8 run_tests.py

存在的意义：直接用 ``python -m unittest`` 时，结果写在 stderr，PowerShell 会把它
当作错误记录渲染（红字、NativeCommandError），再叠加 ``*>`` 重定向和 ``-c``
单行脚本的引号拆行问题，常常看不到真正的结论。本脚本把摘要与完整错误栈都写到
stdout，不需要任何重定向技巧。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=0, stream=sys.stdout).run(suite)

    print("=" * 70)
    print(f"结果: 共 {result.testsRun} 条   失败 {len(result.failures)}   "
          f"错误 {len(result.errors)}   跳过 {len(result.skipped)}")

    for case, reason in result.skipped:
        print(f"  跳过 {case.id().split('.')[-1]}: {reason}")

    for label, items in (("错误", result.errors), ("失败", result.failures)):
        for case, trace in items:
            print("-" * 70)
            print(f"{label}: {case.id()}")
            print(trace)

    ok = not (result.failures or result.errors)
    print("=" * 70)
    print("全部通过（跳过项不影响）" if ok else "存在失败或错误，见上方错误栈")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
