#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""自动更新 docs/codex_context.md 中的项目目录结构（含中文注释）。

用法：
    在项目根目录运行：
        python scripts/update_directory_structure.py

功能说明：
    1. 自动向上查找项目根目录（要求包含 REQUIRED_ROOT_CHILDREN 中的目录）
    2. 读取 docs/codex_context.md 中
       “## 📁 Project Directory Structure” ~ “<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->”
       之间的 ```text 代码块
    3. 从该代码块解析出：
        - 旧的根目录注释
        - 每个相对路径对应的中文注释（例如：config/settings.yaml -> 全局配置 …）
    4. 基于当前真实文件系统重建目录树
    5. 对于路径相同的目录/文件，尽可能复用原来的中文注释
    6. 将新的目录树写回 codex_context.md 对应区域
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# 基本配置
# ---------------------------------------------------------------------

REQUIRED_ROOT_CHILDREN: Tuple[str, ...] = ("quant_system", "data_cache", "notebooks")
CODEX_FILENAME = "codex_context.md"
DOCS_DIRNAME = "docs"

HEADER_TITLE = "## 📁 Project Directory Structure"
MARKER = "<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->"

# 需要忽略的目录
IGNORED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
}


# ---------------------------------------------------------------------
# 路径定位
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
DOCS_DIR = PROJECT_ROOT / DOCS_DIRNAME
CODEX_PATH = DOCS_DIR / CODEX_FILENAME


# ---------------------------------------------------------------------
# 解析旧目录树中的中文注释
# ---------------------------------------------------------------------

def extract_tree_block(content: str) -> Tuple[str, str, str]:
    """从 codex_context.md 全文中抽取目录树代码块与前后部分。

    返回：
        before: 标题之前的文本
        tree_text: ```text 代码块内部的目录树文本
        after: MARKER（含）之后的文本
    """
    header_idx = content.find(HEADER_TITLE)
    marker_idx = content.find(MARKER)

    if header_idx == -1 or marker_idx == -1:
        raise ValueError(
            "未在 codex_context.md 中找到预期的标题或标记，请确认文件中存在：\n"
            f"  标题：{HEADER_TITLE}\n"
            f"  标记：{MARKER}"
        )

    before = content[:header_idx]
    middle = content[header_idx:marker_idx]
    after = content[marker_idx:]

    code_start = middle.find("```text")
    if code_start == -1:
        raise ValueError("在目录结构区域中未找到 ```text 代码块，请检查 codex_context.md 格式。")

    # 找到代码块起始和结束
    code_start = middle.find("\n", code_start)
    if code_start == -1:
        raise ValueError("```text 后未找到换行，请检查 codex_context.md 格式。")
    code_start += 1

    code_end = middle.find("```", code_start)
    if code_end == -1:
        raise ValueError("目录结构代码块未正确闭合，请检查 codex_context.md 格式。")

    tree_text = middle[code_start:code_end].strip("\n")
    return before, tree_text, after


def parse_existing_comments(tree_text: str) -> Tuple[str, Dict[str, str]]:
    """从旧的目录树文本中解析根目录注释和路径 -> 注释映射。

    tree_text 为 ```text 代码块内部的纯文本。
    """
    lines = [line.rstrip("\n") for line in tree_text.splitlines() if line.strip()]
    if not lines:
        return "", {}

    # 解析第一行根目录，例如：
    # quant-a-share/  # 项目根目录（Git 仓库名称）
    root_line = lines[0].strip()
    m = re.match(r"(?P<name>.+?)/(?:\s{2,}#\s*(?P<comment>.*))?", root_line)
    root_comment = ""
    if m:
        root_comment = (m.group("comment") or "").strip()

    comments: Dict[str, str] = {}

    # 栈用于记录每一层的路径，用来计算类似 "a/b/c" 这样的相对路径
    path_stack: List[str] = []

    for line in lines[1:]:
        # 典型结构示例：
        # ├── config/  # 全局配置（settings.yaml 等）
        # │   └── settings.yaml  # 配置文件入口
        if "├── " in line:
            connector = "├── "
        elif "└── " in line:
            connector = "└── "
        else:
            # 不符合结构的行直接跳过
            continue

        prefix, rest = line.split(connector, 1)
        # 每 4 个字符（"│   " 或 "    "）视为一层缩进
        depth = len(prefix) // 4

        rest = rest.strip()

        if "  #" in rest:
            name_part, comment_part = rest.split("  #", 1)
            comment = comment_part.strip()
        else:
            name_part = rest
            comment = ""

        name_clean = name_part.rstrip("/").strip()
        if not name_clean:
            continue

        # 根据 depth 和栈构造相对路径
        if depth == 0:
            rel_path = name_clean
            path_stack = [rel_path]
        else:
            # 保证栈长度 >= depth
            if depth > len(path_stack):
                # 比预期更深，兜底接在上一层后面
                parent = path_stack[-1]
            else:
                parent = path_stack[depth - 1]
                path_stack = path_stack[:depth]

            rel_path = f"{parent}/{name_clean}"
            path_stack.append(rel_path)

        if comment:
            comments[rel_path] = comment

    return root_comment, comments


# ---------------------------------------------------------------------
# 基于真实文件系统生成新的目录树
# ---------------------------------------------------------------------

def build_tree(
    root: Path,
    project_root: Path,
    comments: Dict[str, str],
    prefix: str = "",
) -> str:
    """递归构建目录树文本，并贴上已有中文注释。"""
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines: List[str] = []

    for idx, entry in enumerate(entries):
        name = entry.name

        # 过滤目录
        if entry.is_dir() and (name in IGNORED_DIRS or name.startswith(".")):
            continue

        connector = "└── " if idx == len(entries) - 1 else "├── "
        display_name = f"{name}/" if entry.is_dir() else name

        rel_path = entry.relative_to(project_root).as_posix()
        comment = comments.get(rel_path, "").strip()

        if comment:
            line = f"{prefix}{connector}{display_name}  # {comment}"
        else:
            line = f"{prefix}{connector}{display_name}"

        lines.append(line)

        if entry.is_dir():
            child_prefix = prefix + ("    " if idx == len(entries) - 1 else "│   ")
            subtree = build_tree(entry, project_root, comments, child_prefix)
            if subtree:
                lines.append(subtree)

    return "\n".join([l for l in lines if l])


def generate_directory_tree(
    project_root: Path,
    root_comment: str,
    comments: Dict[str, str],
) -> str:
    """生成完整目录树文本（首行是根目录，其余递归生成）。"""
    root_name = project_root.name
    first_line = f"{root_name}/"
    if root_comment:
        first_line = f"{first_line}  # {root_comment}"

    lines = [first_line]

    subtree = build_tree(project_root, project_root, comments)
    if subtree:
        lines.append(subtree)

    return "\n".join(lines)


# ---------------------------------------------------------------------
# 核心：更新 codex_context.md
# ---------------------------------------------------------------------

def update_codex_directory_structure() -> None:
    """读取 codex_context.md，更新目录树代码块。"""
    if not CODEX_PATH.exists():
        raise FileNotFoundError(f"未找到文件：{CODEX_PATH}")

    content = CODEX_PATH.read_text(encoding="utf-8")

    # 解析旧目录树与注释
    before, old_tree_text, after = extract_tree_block(content)
    root_comment, comments = parse_existing_comments(old_tree_text)

    # 基于真实文件系统生成新目录树
    new_tree_text = generate_directory_tree(PROJECT_ROOT, root_comment, comments)

    # 重新拼接中间段：标题 + 空行 + ```text 代码块 + 空行
    middle = (
        f"{HEADER_TITLE}\n\n"
        "```text\n"
        f"{new_tree_text}\n"
        "```\n\n"
    )

    new_content = before + middle + after
    CODEX_PATH.write_text(new_content, encoding="utf-8")

    print(f"[ok] 目录结构已更新：{CODEX_PATH}")
    print(f"[info] 根目录：{PROJECT_ROOT}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    update_codex_directory_structure()


if __name__ == "__main__":
    main()
