import argparse
import csv
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

sys.path.append(os.path.abspath("."))

from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.scorer import ScoreAggregator
from src.pipeline.structure_features import extract_structure_features
from src.pipeline.token_similarity import TokenSimilarity
from src.api.file_validation import EXTENSION_LANGUAGE_MAP, normalize_language


HUMAN_DIR = "data/raw/human"
AI_DIR = "data/raw/ai"
OUTPUT_FILE = "data/results/evaluation_results.csv"


@dataclass
class Sample:
    path: str
    label: str
    language: str
    normalized_code: str
    ast_features: dict
    embedding: np.ndarray


@dataclass
class PreparedSample:
    sample: Sample
    token_set: set[str]
    ast_vector: np.ndarray
    line_count: int


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


def detect_language_from_path(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    language = EXTENSION_LANGUAGE_MAP.get(ext)
    normalized_language, _ = normalize_language(language)
    return normalized_language


def collect_paths(root_dir: str, include_languages: set[str] | None = None) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            language = detect_language_from_path(path)
            if language is None:
                continue
            if include_languages and language not in include_languages:
                continue
            paths.append((path, language))
    return sorted(paths, key=lambda item: item[0])


def _ast_to_vector(features: dict) -> np.ndarray:
    return np.array(
        [
            float(features.get("num_functions", 0)),
            float(features.get("num_loops", 0)),
            float(features.get("num_conditionals", 0)),
            float(features.get("max_nesting_depth", 0)),
        ],
        dtype=np.float32,
    )


def _token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a) + len(tokens_b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _structure_similarity_vec(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    diff = float(np.abs(vec_a - vec_b).sum())
    return 1.0 / (1.0 + diff)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _iter_sample_indices(
    samples: list[PreparedSample],
    mode: str,
) -> Iterable[tuple[str, list[int]]]:
    if mode == "global":
        return [("global", list(range(len(samples))))]

    by_language: dict[str, list[int]] = {}
    for idx, prepared in enumerate(samples):
        by_language.setdefault(prepared.sample.language, []).append(idx)
    return sorted(by_language.items(), key=lambda item: item[0])


def _iter_cross_label_groups(
    samples: list[PreparedSample],
    mode: str,
) -> Iterable[tuple[str, list[int], dict[str, list[int]]]]:
    if mode == "global":
        indices = list(range(len(samples)))
        labels = {
            "AI": [idx for idx in indices if samples[idx].sample.label == "AI"],
            "HUMAN": [idx for idx in indices if samples[idx].sample.label == "HUMAN"],
        }
        return [("global", indices, labels)]

    by_language: dict[str, dict[str, list[int]]] = {}
    for idx, prepared in enumerate(samples):
        language = prepared.sample.language
        label = prepared.sample.label
        if language not in by_language:
            by_language[language] = {"AI": [], "HUMAN": []}
        by_language[language].setdefault(label, []).append(idx)

    groups: list[tuple[str, list[int], dict[str, list[int]]]] = []
    for language, labels in sorted(by_language.items(), key=lambda item: item[0]):
        indices = sorted(labels.get("AI", []) + labels.get("HUMAN", []))
        groups.append((language, indices, labels))
    return groups


def build_prepared_samples(samples: list[Sample], token_similarity: TokenSimilarity) -> list[PreparedSample]:
    prepared: list[PreparedSample] = []
    for sample in samples:
        line_count = len([line for line in sample.normalized_code.splitlines() if line.strip()])
        prepared.append(
            PreparedSample(
                sample=sample,
                token_set=token_similarity.tokenize(sample.normalized_code),
                ast_vector=_ast_to_vector(sample.ast_features),
                line_count=line_count,
            )
        )
    return prepared


def prepare_samples(
    limit_per_label: int | None = None,
    include_languages: set[str] | None = None,
) -> list[Sample]:
    normalizer = CodeNormalizer()
    ast_analyzer = ASTAnalyzer()
    embedder = EmbeddingGenerator()

    labeled_paths: list[tuple[str, str, str]] = []
    human_paths = collect_paths(HUMAN_DIR, include_languages=include_languages)
    ai_paths = collect_paths(AI_DIR, include_languages=include_languages)

    if limit_per_label is not None:
        human_paths = human_paths[:limit_per_label]
        ai_paths = ai_paths[:limit_per_label]

    labeled_paths.extend((p, "HUMAN", lang) for p, lang in human_paths)
    labeled_paths.extend((p, "AI", lang) for p, lang in ai_paths)

    samples: list[Sample] = []

    total = len(labeled_paths)
    print(f"[1/3] Preparing embeddings/features for {total} files...")
    for i, (path, label, language) in enumerate(labeled_paths, start=1):
        raw_code = read_code(path)
        if not raw_code or not raw_code.strip():
            continue

        normalized_code, _ = normalizer.normalize(raw_code)
        ast_features = extract_structure_features(
            code=normalized_code,
            language=language,
            ast_analyzer=ast_analyzer,
        )
        embedding = embedder.generate(normalized_code).detach().numpy().astype(np.float32)

        samples.append(
            Sample(
                path=path,
                label=label,
                language=language,
                normalized_code=normalized_code,
                ast_features=ast_features,
                embedding=embedding,
            )
        )

        if i % 50 == 0 or i == total:
            print(f"  Embedded {i}/{total}")

    return samples


def evaluate_samples(
    samples: list[Sample],
    output_file: str,
    mode: str = "within-language",
    semantic_top_k: int = 200,
    cross_label: bool = False,
    no_size_penalty: bool = False,
) -> None:
    scorer = ScoreAggregator()
    token_similarity = TokenSimilarity()
    prepared = build_prepared_samples(samples, token_similarity)

    print(f"[2/3] Scoring {len(prepared)} samples (mode={mode}, semantic_top_k={semantic_top_k})...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file",
            "label",
            "language",
            "top_match_file",
            "top_match_label",
            "top_match_language",
            "plagiarism_percentage",
            "confidence",
            "token_similarity",
            "semantic_similarity",
            "structure_similarity",
            "code_lines",
            "size_penalty_applied",
            "pairing_scope",
        ]
        if cross_label:
            fieldnames.extend(["ai_affinity", "same_label_sim", "cross_label_sim"])
        else:
            fieldnames.append("ai_probability")

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        total_samples = len(prepared)
        processed = 0
        if cross_label:
            groups: Iterable[tuple[str, list[int], dict[str, list[int]]]] = _iter_cross_label_groups(prepared, mode)
        else:
            groups = [(group_name, group_indices, {}) for group_name, group_indices in _iter_sample_indices(prepared, mode)]

        for group_name, group_indices, group_labels in groups:
            group_size = len(group_indices)
            has_cross_pairs = True
            if cross_label:
                has_cross_pairs = bool(group_labels.get("AI")) and bool(group_labels.get("HUMAN"))

            if group_size < 2 or not has_cross_pairs:
                for idx in group_indices:
                    entry = prepared[idx]
                    row = {
                        "file": entry.sample.path,
                        "label": entry.sample.label,
                        "language": entry.sample.language,
                        "top_match_file": "",
                        "top_match_label": "",
                        "top_match_language": "",
                        "plagiarism_percentage": 0.0,
                        "confidence": "low",
                        "token_similarity": 0.0,
                        "semantic_similarity": 0.0,
                        "structure_similarity": 0.0,
                        "code_lines": entry.line_count,
                        "size_penalty_applied": False,
                        "pairing_scope": "cross-label" if cross_label else mode,
                    }
                    if cross_label:
                        row["ai_affinity"] = 0.0
                        row["same_label_sim"] = 0.0
                        row["cross_label_sim"] = 0.0
                    else:
                        row["ai_probability"] = 0.0

                    writer.writerow(row)
                processed += group_size
                print(f"  Group '{group_name}': skipped (size={group_size})")
                continue

            print(f"  Group '{group_name}': computing semantic matrix for {group_size} samples")
            vectors = np.stack([prepared[idx].sample.embedding for idx in group_indices]).astype(np.float32)
            vectors = _normalize_rows(vectors)
            semantic_matrix = vectors @ vectors.T

            top_k = max(1, min(semantic_top_k, group_size - 1))

            for local_i, global_i in enumerate(group_indices):
                entry = prepared[global_i]
                semantic_row = semantic_matrix[local_i].copy()
                semantic_row[local_i] = -np.inf

                best_idx = -1
                best_composite = -np.inf
                token_sim = 0.0
                semantic_sim = 0.0
                structure_sim = 0.0

                same_label_best = 0.0
                cross_label_best = 0.0

                def _top_locals_for_label(target_label: str) -> list[int]:
                    label_locals = [
                        local_j
                        for local_j in range(group_size)
                        if local_j != local_i and prepared[group_indices[local_j]].sample.label == target_label
                    ]
                    if not label_locals:
                        return []
                    if len(label_locals) <= top_k:
                        return label_locals

                    label_scores = np.array([semantic_row[local_j] for local_j in label_locals], dtype=np.float32)
                    top_offsets = np.argpartition(label_scores, -top_k)[-top_k:]
                    return [label_locals[int(offset)] for offset in top_offsets]

                if cross_label:
                    entry_label = entry.sample.label
                    opposite_label = "HUMAN" if entry_label == "AI" else "AI"

                    same_locals = _top_locals_for_label(entry_label)
                    cross_locals = _top_locals_for_label(opposite_label)

                    for local_j in same_locals:
                        semantic_value = float(semantic_row[int(local_j)])
                        if semantic_value > same_label_best:
                            same_label_best = semantic_value

                    for local_j in cross_locals:
                        candidate_global = group_indices[int(local_j)]
                        candidate = prepared[candidate_global]

                        semantic_value = float(semantic_row[int(local_j)])
                        token_value = _token_jaccard(entry.token_set, candidate.token_set)
                        structure_value = _structure_similarity_vec(entry.ast_vector, candidate.ast_vector)

                        composite = (
                            scorer.thresholds.plagiarism_token_weight * token_value
                            + scorer.thresholds.plagiarism_semantic_weight * semantic_value
                            + scorer.thresholds.plagiarism_structure_weight * structure_value
                        )

                        if semantic_value > cross_label_best:
                            cross_label_best = semantic_value

                        if composite > best_composite:
                            best_composite = composite
                            best_idx = candidate_global
                            token_sim = token_value
                            semantic_sim = semantic_value
                            structure_sim = structure_value
                else:
                    shortlist_local = np.argpartition(semantic_row, -top_k)[-top_k:]

                    for local_j in shortlist_local:
                        candidate_global = group_indices[int(local_j)]
                        candidate = prepared[candidate_global]

                        semantic_value = float(semantic_row[int(local_j)])
                        token_value = _token_jaccard(entry.token_set, candidate.token_set)
                        structure_value = _structure_similarity_vec(entry.ast_vector, candidate.ast_vector)

                        composite = (
                            scorer.thresholds.plagiarism_token_weight * token_value
                            + scorer.thresholds.plagiarism_semantic_weight * semantic_value
                            + scorer.thresholds.plagiarism_structure_weight * structure_value
                        )

                        if composite > best_composite:
                            best_composite = composite
                            best_idx = candidate_global
                            token_sim = token_value
                            semantic_sim = semantic_value
                            structure_sim = structure_value

                plagiarism_score = scorer.compute_plagiarism_score(token_sim, semantic_sim, structure_sim)

                ai_probability = 0.0
                ai_affinity = 0.0
                if cross_label:
                    # Better metric: difference between same-label and cross-label similarity (not ratio)
                    # Positive = prefers same label, Negative = prefers opposite label
                    affinity_delta = (same_label_best - cross_label_best) * 100.0
                    # Scale to 0-100 range for interpretability
                    ai_affinity = 50.0 + (affinity_delta / 2.0)  # Maps -100:100 delta to 0:100 score
                    ai_affinity = max(0.0, min(100.0, ai_affinity))
                else:
                    ai_probability = scorer.compute_ai_probability(semantic_sim, structure_sim)

                size_penalty = False
                if not no_size_penalty:
                    if entry.line_count <= 4:
                        plagiarism_score = min(plagiarism_score, 35.0)
                        size_penalty = True
                    elif entry.line_count <= 8:
                        plagiarism_score = min(plagiarism_score, 60.0)
                        size_penalty = True

                confidence = scorer.compute_confidence(
                    token_similarity=token_sim,
                    semantic_similarity=semantic_sim,
                    structure_similarity=structure_sim,
                    line_count=entry.line_count,
                )

                top_match_file = ""
                top_match_label = ""
                top_match_language = ""
                if best_idx >= 0:
                    top_match = prepared[best_idx].sample
                    top_match_file = top_match.path
                    top_match_label = top_match.label
                    top_match_language = top_match.language

                row = {
                    "file": entry.sample.path,
                    "label": entry.sample.label,
                    "language": entry.sample.language,
                    "top_match_file": top_match_file,
                    "top_match_label": top_match_label,
                    "top_match_language": top_match_language,
                    "plagiarism_percentage": round(plagiarism_score, 2),
                    "confidence": confidence,
                    "token_similarity": round(token_sim, 4),
                    "semantic_similarity": round(semantic_sim, 4),
                    "structure_similarity": round(structure_sim, 4),
                    "code_lines": entry.line_count,
                    "size_penalty_applied": size_penalty,
                    "pairing_scope": "cross-label" if cross_label else mode,
                }
                if cross_label:
                    row["ai_affinity"] = round(ai_affinity, 2)
                    row["same_label_sim"] = round(same_label_best, 4)
                    row["cross_label_sim"] = round(cross_label_best, 4)
                else:
                    row["ai_probability"] = round(ai_probability, 2)

                writer.writerow(row)

                processed += 1
                if processed % 50 == 0 or processed == total_samples:
                    print(f"  Scored {processed}/{total_samples}")


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
    parser.add_argument(
        "--mode",
        choices=["within-language", "global"],
        default="within-language",
        help="Candidate matching scope for nearest-neighbor evaluation.",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated language filter (e.g. python,java,cpp).",
    )
    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=200,
        help="How many top semantic candidates to re-rank with token+structure per sample.",
    )
    parser.add_argument(
        "--cross-label",
        action="store_true",
        help="Match each sample against opposite-label candidates only (AI vs HUMAN).",
    )
    parser.add_argument(
        "--no-size-penalty",
        action="store_true",
        help="Disable short-code plagiarism caps (recommended for offline analysis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_languages: set[str] | None = None
    if args.languages:
        include_languages = {
            language.strip().lower()
            for language in args.languages.split(",")
            if language.strip()
        }

    samples = prepare_samples(
        limit_per_label=args.limit_per_label,
        include_languages=include_languages,
    )

    if len(samples) < 2:
        print("Not enough samples to evaluate.")
        return

    labels = {sample.label for sample in samples}
    if "AI" not in labels:
        print("Warning: no AI samples found. Evaluation will proceed with available labels.")

    language_counts: dict[str, int] = {}
    for sample in samples:
        language_counts[sample.language] = language_counts.get(sample.language, 0) + 1

    print("Language counts:")
    for language, count in sorted(language_counts.items()):
        print(f"  - {language}: {count}")

    evaluate_samples(
        samples,
        output_file=args.output,
        mode=args.mode,
        semantic_top_k=args.semantic_top_k,
        cross_label=args.cross_label,
        no_size_penalty=args.no_size_penalty,
    )
    print(f"[3/3] Results written -> {args.output}")
    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
