#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动更新 docs/codex_context.md 中的项目目录树（含中文注释）。

用法：
    在项目根目录运行：
        python scripts/update_directory_structure.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED_ROOT_CHILDREN: Tuple[str, ...] = ("quant_system", "data_cache", "notebooks")
CODEX_FILENAME = "codex_context.md"
DOCS_DIRNAME = "docs"

HEADER_TITLE = "## 📁 Project Directory Structure"
MARKER = "<!-- CODEX_UPDATE_DIRECTORY_STRUCTURE -->"

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

CONNECTORS: Tuple[str, ...] = ("├──", "└──", "â”œâ”€â”€", "â””â”€â”€")


# ---------------------------------------------------------------------
# 路径定位
# ---------------------------------------------------------------------
def find_project_root(start: Path, required_children: Tuple[str, ...]) -> Path:
    """向上递归查找，同时包含指定子目录的项目根路径。"""
    for candidate in (start, *start.parents):
        if all((candidate / child).exists() for child in required_children):
            return candidate
    raise RuntimeError(
        f"未能自 {start} 向上找到包含 {required_children} 的项目根目录，请检查脚本位置。"
    )


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(SCRIPT_PATH.parent, REQUIRED_ROOT_CHILDREN)
DOCS_DIR = PROJECT_ROOT / DOCS_DIRNAME
CODEX_PATH = DOCS_DIR / CODEX_FILENAME


# ---------------------------------------------------------------------
# 解析旧目录树的中文注释
# ---------------------------------------------------------------------
def extract_tree_block(content: str) -> Tuple[str, str, str]:
    """从 codex_context.md 中抽取目录树代码块及前后部分。"""
    header_idx = content.find(HEADER_TITLE)
    marker_idx = content.find(MARKER)
    if header_idx == -1 or marker_idx == -1:
        raise ValueError(
            "未找到目录结构标题或标记，请确认文档包含：\n"
            f"  标题：{HEADER_TITLE}\n"
            f"  标记：{MARKER}"
        )

    before = content[:header_idx]
    middle = content[header_idx:marker_idx]
    after = content[marker_idx:]

    code_start = middle.find("```text")
    if code_start == -1:
        raise ValueError("目录结构区域缺少 ```text 代码块，请检查文档格式。")
    code_start = middle.find("\n", code_start)
    if code_start == -1:
        raise ValueError("```text 行后未找到换行，请检查文档格式。")
    code_start += 1

    code_end = middle.find("```", code_start)
    if code_end == -1:
        raise ValueError("目录结构代码块未正确闭合，请检查文档格式。")

    tree_text = middle[code_start:code_end].strip("\n")
    return before, tree_text, after


def _detect_depth(prefix: str) -> int:
    """根据前缀字符估算树的深度（每 4 个字符视为一级缩进）。"""
    clean = prefix.replace("│", " ").replace("â”‚", " ")
    return len(clean) // 4


def parse_existing_comments(tree_text: str) -> Tuple[str, Dict[str, str]]:
    """从旧的目录树文本解析根目录注释与 路径->注释 的映射。"""
    lines = [line.rstrip("\n") for line in tree_text.splitlines() if line.strip()]
    if not lines:
        return "", {}

    root_line = lines[0].strip()
    m = re.match(r"(?P<name>.+?)/(?:\s{2,}#\s*(?P<comment>.*))?", root_line)
    root_comment = (m.group("comment").strip() if m and m.group("comment") else "")

    comments: Dict[str, str] = {}
    path_stack: List[str] = []

    for line in lines[1:]:
        connector_idx = None
        for c in CONNECTORS:
            idx = line.find(c)
            if idx != -1:
                connector_idx = idx
                connector = c
                break
        if connector_idx is None:
            continue

        prefix = line[:connector_idx]
        rest = line[connector_idx + len(connector):].strip()
        depth = _detect_depth(prefix)

        if "  #" in rest:
            name_part, comment_part = rest.split("  #", 1)
            comment = comment_part.strip()
        else:
            name_part, comment = rest, ""

        name_clean = name_part.rstrip("/").strip()
        if not name_clean:
            continue

        if depth == 0:
            rel_path = name_clean
            path_stack = [rel_path]
        else:
            if depth > len(path_stack):
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
def build_tree(root: Path, project_root: Path, comments: Dict[str, str], prefix: str = "") -> str:
    """递归构建目录树文本，并贴上已有中文注释。"""
    entries = sorted(
        [p for p in root.iterdir() if not (p.is_dir() and (p.name in IGNORED_DIRS or p.name.startswith(".")))],
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    lines: List[str] = []
    total = len(entries)

    for idx, entry in enumerate(entries):
        connector = "└── " if idx == total - 1 else "├── "
        display_name = f"{entry.name}/" if entry.is_dir() else entry.name
        rel_path = entry.relative_to(project_root).as_posix()
        comment = comments.get(rel_path, "").strip()
        line = f"{prefix}{connector}{display_name}"
        if comment:
            line += f"  # {comment}"
        lines.append(line)

        if entry.is_dir():
            child_prefix = prefix + ("    " if idx == total - 1 else "│   ")
            subtree = build_tree(entry, project_root, comments, child_prefix)
            if subtree:
                lines.append(subtree)

    return "\n".join(lines)


def generate_directory_tree(project_root: Path, root_comment: str, comments: Dict[str, str]) -> str:
    """生成完整目录树文本（首行为根目录，其余递归生成）。"""
    root_name = project_root.name
    first_line = f"{root_name}/"
    if root_comment:
        first_line += f"  # {root_comment}"

    subtree = build_tree(project_root, project_root, comments)
    if subtree:
        return "\n".join([first_line, subtree])
    return first_line


# ---------------------------------------------------------------------
# 核心：更新 codex_context.md
# ---------------------------------------------------------------------
def update_directory_structure() -> None:
    """读取 codex_context.md，更新目录树代码块。"""
    if not CODEX_PATH.exists():
        raise FileNotFoundError(f"未找到文件：{CODEX_PATH}")

    content = CODEX_PATH.read_text(encoding="utf-8")
    before, old_tree_text, after = extract_tree_block(content)
    root_comment, comments = parse_existing_comments(old_tree_text)

    new_tree_text = generate_directory_tree(PROJECT_ROOT, root_comment, comments)
    middle = (
        f"{HEADER_TITLE}\n\n"
        "```text\n"
        f"{new_tree_text}\n"
        "```\n\n"
    )

    new_content = before + middle + after
    CODEX_PATH.write_text(new_content, encoding="utf-8")
    print(f"[ok] 目录结构已更新：{CODEX_PATH}")
    print(f"[info] 项目根目录：{PROJECT_ROOT}")


def main() -> None:
    update_directory_structure()


if __name__ == "__main__":
    main()
