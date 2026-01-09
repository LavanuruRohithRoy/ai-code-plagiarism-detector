from src.storage.db import engine
from src.storage.models import Base

def init():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init()
