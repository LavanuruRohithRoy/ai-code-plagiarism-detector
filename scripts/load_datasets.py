from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath("."))

from src.api.dependencies import sync_faiss_with_db
from src.api.file_validation import EXTENSION_LANGUAGE_MAP, normalize_language
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.structure_features import extract_structure_features
from src.storage.repository import AnalysisRepository


def detect_language(path: Path) -> str | None:
	ext = path.suffix.lower()
	language = EXTENSION_LANGUAGE_MAP.get(ext)
	normalized_language, language_error = normalize_language(language)
	if language_error or not normalized_language:
		return None
	return normalized_language


def collect_from_filesystem() -> list[tuple[Path, str]]:
	roots = [Path("data/raw/ai"), Path("data/raw/human")]
	collected: list[tuple[Path, str]] = []

	for root in roots:
		if not root.exists():
			continue

		for path in root.rglob("*"):
			if not path.is_file():
				continue
			language = detect_language(path)
			if not language:
				continue
			collected.append((path, language))

	return sorted(collected, key=lambda item: str(item[0]))


def collect_from_csv(csv_path: Path) -> list[tuple[Path, str]]:
	if not csv_path.exists():
		return []

	collected: list[tuple[Path, str]] = []
	seen: set[str] = set()

	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			file_path = (row.get("file") or "").strip()
			if not file_path:
				continue

			normalized_path = file_path.replace("\\", "/")
			if normalized_path in seen:
				continue

			path = Path(file_path)
			if not path.exists() or not path.is_file():
				continue

			language = detect_language(path)
			if not language:
				continue

			seen.add(normalized_path)
			collected.append((path, language))

	return sorted(collected, key=lambda item: str(item[0]))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Incremental dataset loader")
	parser.add_argument(
		"--source",
		choices=["auto", "csv", "filesystem"],
		default="auto",
		help="Source of file list. auto prefers CSV then falls back to filesystem.",
	)
	parser.add_argument(
		"--csv-path",
		default="data/results/evaluation_results.csv",
		help="CSV file path used when source is csv/auto.",
	)
	parser.add_argument(
		"--rebuild-faiss",
		action="store_true",
		help="Rebuild FAISS after loading. Off by default for faster incremental loads.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	started = time.time()

	csv_path = Path(args.csv_path)
	if args.source == "csv":
		candidates = collect_from_csv(csv_path)
		source_used = f"csv:{csv_path}"
	elif args.source == "filesystem":
		candidates = collect_from_filesystem()
		source_used = "filesystem"
	else:
		candidates = collect_from_csv(csv_path)
		source_used = f"csv:{csv_path}" if candidates else "filesystem"
		if not candidates:
			candidates = collect_from_filesystem()

	normalizer = CodeNormalizer()
	ast_analyzer = ASTAnalyzer()
	embedder = EmbeddingGenerator()
	repository = AnalysisRepository()

	total = len(candidates)
	inserted = 0
	skipped_existing = 0
	skipped_invalid = 0

	print(f"Loading dataset candidates={total} source={source_used}")

	for index, (path, language) in enumerate(candidates, start=1):
		try:
			raw_code = path.read_text(encoding="utf-8", errors="ignore")
		except Exception:
			skipped_invalid += 1
			continue

		if not raw_code or not raw_code.strip():
			skipped_invalid += 1
			continue

		code_hash = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()

		if repository.exists_code_hash(code_hash):
			skipped_existing += 1
			continue

		normalized_code, _ = normalizer.normalize(raw_code)
		ast_features = extract_structure_features(
			code=normalized_code,
			language=language,
			ast_analyzer=ast_analyzer,
		)
		embedding = embedder.generate(normalized_code).detach().numpy()

		did_insert = repository.save_result(
			code_hash=code_hash,
			plagiarism_score=0.0,
			ai_probability=0.0,
			normalized_code=normalized_code,
			ast_features=ast_features,
			embedding_vector=embedding,
		)

		if did_insert:
			inserted += 1
		else:
			skipped_existing += 1

		if index % 50 == 0 or index == total:
			print(
				f"  progress {index}/{total} inserted={inserted} "
				f"existing={skipped_existing} invalid={skipped_invalid}"
			)

	faiss_size = "unchanged"
	if args.rebuild_faiss and inserted > 0:
		faiss_size = str(sync_faiss_with_db())

	elapsed = time.time() - started
	print(
		"Dataset load complete. "
		f"total={total}, inserted={inserted}, skipped_existing={skipped_existing}, "
		f"skipped_invalid={skipped_invalid}, faiss_vectors={faiss_size}, elapsed_sec={elapsed:.1f}"
	)


if __name__ == "__main__":
	main()
