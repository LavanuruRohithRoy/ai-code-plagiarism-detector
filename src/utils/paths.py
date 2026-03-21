from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _in_container() -> bool:
    return Path("/app").exists()


def get_data_dir() -> Path:
    override = os.getenv("APP_DATA_DIR")
    if override:
        return Path(override)

    if _in_container():
        return Path("/app/data")

    return PROJECT_ROOT / "runtime" / "data"


def get_db_path() -> Path:
    override = os.getenv("APP_DB_PATH")
    if override:
        return Path(override)

    if _in_container():
        return Path("/app/db/plagiarism.db")

    return PROJECT_ROOT / "runtime" / "db" / "plagiarism.db"


def get_embeddings_dir() -> Path:
    override = os.getenv("APP_EMBEDDINGS_DIR")
    if override:
        return Path(override)

    return get_data_dir() / "embeddings"


def get_raw_data_dir() -> Path:
    override = os.getenv("APP_RAW_DATA_DIR")
    if override:
        return Path(override)

    return get_data_dir() / "raw"
