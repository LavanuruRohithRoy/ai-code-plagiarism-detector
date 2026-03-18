from __future__ import annotations

import re

from src.pipeline.ast_analyzer import ASTAnalyzer


def heuristic_structure_features(code: str) -> dict[str, int]:
    """Language-agnostic structural approximation for non-Python code."""
    non_empty = [line.strip() for line in code.splitlines() if line.strip()]
    loop_count = len(re.findall(r"\b(for|while|do)\b", code))
    cond_count = len(re.findall(r"\b(if|else if|switch|case)\b", code))
    fn_like = len(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_<>:\[\]]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{",
            code,
        )
    )

    depth = 0
    max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)

    if max_depth == 0:
        indent_depth = 0
        for line in non_empty:
            leading_spaces = len(line) - len(line.lstrip(" "))
            indent_depth = max(indent_depth, leading_spaces // 4)
        max_depth = indent_depth

    return {
        "num_functions": fn_like,
        "num_loops": loop_count,
        "num_conditionals": cond_count,
        "max_nesting_depth": max_depth,
    }


def extract_structure_features(
    code: str,
    language: str | None,
    ast_analyzer: ASTAnalyzer,
) -> dict[str, int]:
    """Use Python AST when available, otherwise use language-agnostic heuristics."""
    normalized_language = language.lower() if language else None

    if normalized_language in (None, "python"):
        ast_features = ast_analyzer.analyze(code)
        if any(ast_features.values()):
            return ast_features

    return heuristic_structure_features(code)
