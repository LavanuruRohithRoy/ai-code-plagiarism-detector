from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.dependencies import sync_faiss_with_db
from src.api.routes import router
from src.utils.config import load_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    sync_faiss_with_db()
    yield


def create_app() -> FastAPI:
    settings = load_settings()

    app = FastAPI(
        title="AI Code Plagiarism Detector",
        version="0.2.0",
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan,
    )

    app.include_router(router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()