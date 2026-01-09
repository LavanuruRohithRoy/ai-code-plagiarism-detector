import ast
from typing import Dict


class ASTAnalyzer:
    """
    Safely extracts structural features from Python code.
    No execution, parse-only.
    """

    def analyze(self, code: str) -> Dict[str, int]:
        features = {
            "num_functions": 0,
            "num_loops": 0,
            "num_conditionals": 0,
            "max_nesting_depth": 0
        }

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Invalid Python → return empty but safe features
            return features

        def walk(node, depth=0):
            features["max_nesting_depth"] = max(
                features["max_nesting_depth"], depth
            )

            for child in ast.iter_child_nodes(node):

                if isinstance(child, ast.FunctionDef):
                    features["num_functions"] += 1

                if isinstance(child, (ast.For, ast.While)):
                    features["num_loops"] += 1

                if isinstance(child, ast.If):
                    features["num_conditionals"] += 1

                walk(child, depth + 1)

        walk(tree)
        return features
