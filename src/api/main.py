from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()