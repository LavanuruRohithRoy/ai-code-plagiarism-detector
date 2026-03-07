import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.append(os.path.abspath("."))

from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.scorer import ScoreAggregator
from src.pipeline.token_similarity import TokenSimilarity


HUMAN_DIR = "data/raw/human"
AI_DIR = "data/raw/ai"
OUTPUT_FILE = "data/results/evaluation_results.csv"


@dataclass
class Sample:
    path: str
    label: str
    normalized_code: str
    ast_features: dict
    embedding: np.ndarray


def read_code(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def ast_similarity(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0

    keys = set(a.keys()) | set(b.keys())
    diff = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
    return 1 / (1 + diff)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def collect_paths(root_dir: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                paths.append(os.path.join(root, file))
    return sorted(paths)


def prepare_samples(limit_per_label: int | None = None) -> list[Sample]:
    normalizer = CodeNormalizer()
    ast_analyzer = ASTAnalyzer()
    embedder = EmbeddingGenerator()

    labeled_paths: list[tuple[str, str]] = []
    human_paths = collect_paths(HUMAN_DIR)
    ai_paths = collect_paths(AI_DIR)

    if limit_per_label is not None:
        human_paths = human_paths[:limit_per_label]
        ai_paths = ai_paths[:limit_per_label]

    labeled_paths.extend((p, "HUMAN") for p in human_paths)
    labeled_paths.extend((p, "AI") for p in ai_paths)

    samples: list[Sample] = []

    print(f"Preparing embeddings/features for {len(labeled_paths)} files...")
    for path, label in labeled_paths:
        raw_code = read_code(path)
        if not raw_code or not raw_code.strip():
            continue

        normalized_code, _ = normalizer.normalize(raw_code)
        ast_features = ast_analyzer.analyze(normalized_code)
        embedding = embedder.generate(normalized_code).detach().numpy().astype(np.float32)

        samples.append(
            Sample(
                path=path,
                label=label,
                normalized_code=normalized_code,
                ast_features=ast_features,
                embedding=embedding,
            )
        )

    return samples


def evaluate_samples(samples: list[Sample], output_file: str) -> None:
    scorer = ScoreAggregator()
    token_similarity = TokenSimilarity()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "label",
                "top_match_file",
                "top_match_label",
                "plagiarism_percentage",
                "ai_probability",
                "confidence",
                "token_similarity",
                "semantic_similarity",
                "structure_similarity",
                "code_lines",
                "size_penalty_applied",
            ],
        )
        writer.writeheader()

        for i, sample in enumerate(samples):
            token_sim = 0.0
            semantic_sim = 0.0
            structure_sim = 0.0
            best_idx = -1
            best_composite = -1.0

            for j, candidate in enumerate(samples):
                if i == j:
                    continue

                t = scorer.thresholds.plagiarism_token_weight * (
                    token_similarity.jaccard_similarity(sample.normalized_code, candidate.normalized_code)
                )
                s = scorer.thresholds.plagiarism_semantic_weight * (
                    cosine_similarity(sample.embedding, candidate.embedding)
                )
                a = scorer.thresholds.plagiarism_structure_weight * (
                    ast_similarity(sample.ast_features, candidate.ast_features)
                )
                composite = t + s + a

                if composite > best_composite:
                    best_composite = composite
                    best_idx = j
                    token_sim = t / scorer.thresholds.plagiarism_token_weight if scorer.thresholds.plagiarism_token_weight else 0.0
                    semantic_sim = s / scorer.thresholds.plagiarism_semantic_weight if scorer.thresholds.plagiarism_semantic_weight else 0.0
                    structure_sim = a / scorer.thresholds.plagiarism_structure_weight if scorer.thresholds.plagiarism_structure_weight else 0.0

            plagiarism_score = scorer.compute_plagiarism_score(token_sim, semantic_sim, structure_sim)
            ai_probability = scorer.compute_ai_probability(semantic_sim, structure_sim)

            line_count = len([l for l in sample.normalized_code.splitlines() if l.strip()])
            size_penalty = False
            if line_count <= 4:
                plagiarism_score = min(plagiarism_score, 35.0)
                size_penalty = True
            elif line_count <= 8:
                plagiarism_score = min(plagiarism_score, 60.0)
                size_penalty = True

            confidence = scorer.compute_confidence(
                token_similarity=token_sim,
                semantic_similarity=semantic_sim,
                structure_similarity=structure_sim,
                line_count=line_count,
            )

            top_match_file = ""
            top_match_label = ""
            if best_idx >= 0:
                top_match_file = samples[best_idx].path
                top_match_label = samples[best_idx].label

            writer.writerow(
                {
                    "file": sample.path,
                    "label": sample.label,
                    "top_match_file": top_match_file,
                    "top_match_label": top_match_label,
                    "plagiarism_percentage": round(plagiarism_score, 2),
                    "ai_probability": round(ai_probability, 2),
                    "confidence": confidence,
                    "token_similarity": round(token_sim, 4),
                    "semantic_similarity": round(semantic_sim, 4),
                    "structure_similarity": round(structure_sim, 4),
                    "code_lines": line_count,
                    "size_penalty_applied": size_penalty,
                }
            )

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(samples)} files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline dataset evaluation")
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=None,
        help="Optional limit for quick validation runs.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = prepare_samples(limit_per_label=args.limit_per_label)

    if len(samples) < 2:
        print("Not enough samples to evaluate.")
        return

    evaluate_samples(samples, output_file=args.output)
    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
