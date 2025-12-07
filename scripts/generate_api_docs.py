"""Generate API reference for the quant_system package.

用法：
    在项目根目录运行：
        python scripts/generate_api_docs.py
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REQUIRED_ROOT_CHILDREN = ("quant_system", "data_cache", "notebooks")
PACKAGE_NAME = "quant_system"


# ---------------------------------------------------------------------
# 路径定位与导入准备
# ---------------------------------------------------------------------

def find_project_root(start: Path, required_children: Tuple[str, ...]) -> Path:
    """向上递归查找，直到找到同时包含指定目录的项目根路径。"""
    for candidate in (start, *start.parents):
        if all((candidate / child).exists() for child in required_children):
            return candidate
    raise RuntimeError(
        f"未能从 {start} 向上找到包含 {required_children} 的项目根目录，请检查脚本位置。"
    )


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(SCRIPT_PATH.parent, REQUIRED_ROOT_CHILDREN)
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_PATH = DOCS_DIR / "api_reference.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------

def iter_modules_in_package(package_dir: Path, package_name: str) -> Iterable[str]:
    """遍历包下所有模块，返回模块 import 路径字符串。"""
    for _finder, name, _ispkg in pkgutil.walk_packages(
        [str(package_dir)], prefix=f"{package_name}."
    ):
        parts = name.split(".")
        if any(part.startswith("__") for part in parts):
            continue
        yield name


def format_docstring(doc: str | None) -> str:
    """格式化 docstring，缺失时返回占位说明。"""
    if not doc:
        return "(暂无说明，TODO)"
    return inspect.cleandoc(doc)


def safe_signature(obj) -> str:
    """安全地生成对象签名，失败时返回省略号形式。"""
    try:
        sig = inspect.signature(obj)
        return f"{obj.__name__}{sig}"
    except (TypeError, ValueError):
        return f"{obj.__name__}(...)"


def collect_public_members(module) -> Tuple[List, List]:
    """从模块中提取公开函数和公开类。"""
    public_funcs = []
    public_classes = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            public_funcs.append(obj)
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            public_classes.append(obj)

    public_funcs.sort(key=lambda f: f.__name__)
    public_classes.sort(key=lambda c: c.__name__)
    return public_funcs, public_classes


def collect_public_methods(cls) -> List[Tuple[str, object]]:
    """列出类中定义的公开方法（不以下划线开头）。"""
    methods: List[Tuple[str, object]] = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            continue
        if getattr(obj, "__module__", None) != cls.__module__:
            continue
        qualname = getattr(obj, "__qualname__", "")
        if "." in qualname and not qualname.startswith(cls.__name__ + "."):
            continue
        methods.append((name, obj))

    methods.sort(key=lambda item: item[0])
    return methods


# ---------------------------------------------------------------------
# 核心：生成 markdown
# ---------------------------------------------------------------------

def generate_api_markdown() -> str:
    """扫描 quant_system 包，生成完整 API 文档的 markdown 字符串。"""
    lines: list[str] = []

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# A股量化分析系统 API 说明文档")
    lines.append("")
    lines.append(
        f"> 本文档由 `scripts/generate_api_docs.py` 自动生成（生成时间：{now}）。"
    )
    lines.append("> 请勿手工修改本文件，如需更新请修改源码或脚本后重新生成。")
    lines.append("")
    lines.append("---")
    lines.append("")

    module_names = sorted(iter_modules_in_package(PACKAGE_DIR, PACKAGE_NAME))

    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 导入模块 {mod_name} 失败：{exc}")
            continue

        public_funcs, public_classes = collect_public_members(module)
        if not public_funcs and not public_classes:
            continue

        lines.append(f"## 模块 `{mod_name}`")

        module_doc = format_docstring(module.__doc__)
        if module_doc:
            lines.append("")
            lines.append(module_doc)
            lines.append("")

        if public_funcs:
            lines.append("")
            lines.append("### 函数")
            lines.append("")
            for func in public_funcs:
                lines.append(f"#### {func.__name__}")
                lines.append("")
                lines.append(f"- 签名：`{safe_signature(func)}`")
                lines.append("")
                lines.append("**说明：**")
                lines.append("")
                lines.append(format_docstring(func.__doc__))
                lines.append("")

        if public_classes:
            lines.append("")
            lines.append("### 类")
            lines.append("")
            for cls in public_classes:
                lines.append(f"#### {cls.__name__}")
                lines.append("")
                lines.append(f"- 定义：`class {cls.__name__}`")
                lines.append("")
                lines.append("**说明：**")
                lines.append("")
                lines.append(format_docstring(cls.__doc__))
                lines.append("")

                methods = collect_public_methods(cls)
                if methods:
                    lines.append("**公开方法：**")
                    lines.append("")
                    for name, method in methods:
                        lines.append(f"- `{safe_signature(method)}`")
                    lines.append("")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = generate_api_markdown()
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"[ok] API 文档已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
