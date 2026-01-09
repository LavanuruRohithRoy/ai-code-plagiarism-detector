import re
from typing import Tuple


class CodeNormalizer:
    def normalize(self, code: str) -> Tuple[str, dict]:
        original_length = len(code)

        # Normalize line endings
        code = code.replace("\r\n", "\n").replace("\r", "\n")

        # Remove single-line comments (# and //)
        code = re.sub(r"(#.*?$|//.*?$)", "", code, flags=re.MULTILINE)

        # Remove multi-line comments (/* */)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # IMPORTANT: DO NOT destroy indentation for Python
        # Normalize trailing spaces only
        code = "\n".join(line.rstrip() for line in code.split("\n"))

        # Remove excessive empty lines (keep structure)
        code = re.sub(r"\n{3,}", "\n\n", code)

        normalized_length = len(code)

        meta = {
            "original_length": original_length,
            "normalized_length": normalized_length
        }

        return code, meta
