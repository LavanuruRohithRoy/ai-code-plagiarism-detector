from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(os.path.abspath("."))

from src.pipeline.dataset_matcher import DatasetMatcher
from src.api.dependencies import sync_faiss_with_db


def main() -> None:
	db_path = Path("plagiarism.db")
	if not db_path.exists():
		print("❌ plagiarism.db not found")
		return

	conn = sqlite3.connect(str(db_path))
	cur = conn.cursor()

	cur.execute("SELECT COUNT(*) FROM analysis_results")
	total_rows = cur.fetchone()[0]

	cur.execute("SELECT COUNT(*) FROM analysis_results WHERE embedding IS NULL")
	null_embeddings = cur.fetchone()[0]

	cur.execute("SELECT COUNT(*) FROM analysis_results WHERE normalized_code IS NULL OR TRIM(normalized_code) = ''")
	empty_code = cur.fetchone()[0]

	cur.execute("SELECT COUNT(*) FROM analysis_results WHERE code_hash IS NULL OR TRIM(code_hash) = ''")
	empty_hash = cur.fetchone()[0]

	conn.close()

	matcher = DatasetMatcher()
	faiss_vectors = sync_faiss_with_db()

	print("✅ Sanity check")
	print(f"  db_rows={total_rows}")
	print(f"  db_null_embeddings={null_embeddings}")
	print(f"  db_empty_code={empty_code}")
	print(f"  db_empty_hash={empty_hash}")
	print(f"  dataset_matcher_entries={matcher.size()}")
	print(f"  faiss_vectors_after_sync={faiss_vectors}")


if __name__ == "__main__":
	main()
