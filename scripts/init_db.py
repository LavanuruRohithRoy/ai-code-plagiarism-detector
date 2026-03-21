import os
import sys

sys.path.append(os.path.abspath("."))

from src.storage.db import engine
from src.storage.models import Base


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully.")


if __name__ == "__main__":
    init_db()
