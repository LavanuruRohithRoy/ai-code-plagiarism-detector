import re
from typing import Set


class TokenSimilarity:
    """
    Computes surface-level similarity using token overlap.
    """

    def tokenize(self, code: str) -> Set[str]:
        # Simple tokenizer: words + symbols
        tokens = re.findall(r"[A-Za-z_]+|\d+|==|!=|<=|>=|[^\s]", code)
        return set(tokens)

    def jaccard_similarity(self, code_a: str, code_b: str) -> float:
        tokens_a = self.tokenize(code_a)
        tokens_b = self.tokenize(code_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)

        return len(intersection) / len(union)
