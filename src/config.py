"""转发到项目根目录的 config.py，消除重复配置。

本文件原是根目录 config.py 的一份副本。两份并存时，`import config` 解析到哪一份
取决于 sys.path 的顺序：管线模块把根目录排在前面、测试文件曾把 src 排在前面，
于是训练读到支持环境变量的新版、测试读到硬编码路径的旧版。换机器时表现为
"按说明设了 CAMELSH_DATA_PATH 却不生效"，进而在未挂载的盘上静默卡死。

现在无论从哪个 sys.path 顺序进来，取值都来自同一处。按文件路径显式加载，
不走模块名解析，避免自我导入。
"""

import importlib.util as _importlib_util
from pathlib import Path as _Path

_ROOT_CONFIG = _Path(__file__).resolve().parents[1] / "config.py"

_spec = _importlib_util.spec_from_file_location("_root_config", _ROOT_CONFIG)
if _spec is None or _spec.loader is None:          # pragma: no cover - 结构损坏
    raise ImportError(f"无法加载根目录配置: {_ROOT_CONFIG}")
_root_config = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)

globals().update({k: v for k, v in vars(_root_config).items()
                  if not k.startswith("_")})

__all__ = [k for k in globals() if not k.startswith("_")]
