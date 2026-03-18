from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


MAX_CODE_CHARS = 50000
MAX_FILE_BYTES = 200 * 1024
MAX_FILES_PER_BATCH = 25
MAX_NON_EMPTY_LOC = 1500
ALLOWED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".cpp",
    ".c",
    ".go",
    ".rs",
}

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
}

LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "rs": "rust",
}

SUPPORTED_LANGUAGES = sorted(set(LANGUAGE_ALIASES.values()))


def detect_language_from_filename(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext)


def normalize_language(language: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if language is None:
        return None, None

    normalized = LANGUAGE_ALIASES.get(language.strip().lower())
    if normalized is None:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        return None, f"Unsupported language '{language}'. Supported: {supported}"

    return normalized, None


def validate_file_name(filename: str) -> Optional[str]:
    if not filename or not filename.strip():
        return "Missing filename"

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    return None


def validate_file_content(content: bytes) -> tuple[Optional[str], Optional[str]]:
    if not content:
        return None, "File is empty"

    if len(content) > MAX_FILE_BYTES:
        return None, f"File size exceeds {MAX_FILE_BYTES} bytes"

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        return None, "Only UTF-8 encoded text files are supported"

    if len(decoded.strip()) == 0:
        return None, "File contains only whitespace"

    if len(decoded) > MAX_CODE_CHARS:
        return None, f"Code length exceeds {MAX_CODE_CHARS} characters"

    return decoded, None


def compute_code_metrics(code: str) -> dict[str, int]:
    lines = code.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    comment_lines = [
        line for line in non_empty_lines if line.strip().startswith(("#", "//", "/*", "*"))
    ]
    token_count_estimate = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", code))

    return {
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "char_count": len(code),
        "token_count_estimate": token_count_estimate,
        "comment_lines_estimate": len(comment_lines),
    }


def validate_code_text(code: str) -> tuple[dict[str, int], Optional[str]]:
    metrics = compute_code_metrics(code)

    if metrics["non_empty_lines"] == 0:
        return metrics, "Code contains no executable lines"

    if metrics["non_empty_lines"] > MAX_NON_EMPTY_LOC:
        return metrics, f"Code exceeds max LOC ({MAX_NON_EMPTY_LOC} non-empty lines)"

    return metrics, None
