import os
import sys

sys.path.append(os.path.abspath("."))

from src.api.dependencies import sync_faiss_with_db


def main() -> None:
	total = sync_faiss_with_db()
	print(f"FAISS rebuilt from SQLite. vectors={total}")


if __name__ == "__main__":
	main()
