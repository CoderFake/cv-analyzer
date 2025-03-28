import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.database import engine, Base


if settings.USE_OLLAMA:
    try:
        from app.services.ollama_llm_service import ollama_llm_service as active_llm_service

        print(f"Using Ollama LLM service with model {settings.OLLAMA_MODEL}")
    except ImportError:
        from app.services.llm_service import llm_service as active_llm_service

        print(f"Ollama LLM service not available, falling back to local LLM")
else:
    from app.services.llm_service import llm_service as active_llm_service

    print(f"Using local LLM service with model at {settings.LLM_MODEL_PATH}")

llm_service = active_llm_service

try:
    from app.utils.memory_monitor import memory_monitor

    use_memory_monitor = settings.ENABLE_MEMORY_MONITORING
except ImportError:
    use_memory_monitor = False
    print("Memory monitoring not available")


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with engine.begin() as conn:
        if settings.DEBUG:
            await conn.run_sync(Base.metadata.create_all)

    if use_memory_monitor:
        memory_monitor.start()
        print(f"Memory monitoring started with limit: {settings.MAX_MEMORY_USAGE_MB}MB")

    yield

    if use_memory_monitor:
        memory_monitor.stop()

    if settings.USE_OLLAMA:
        try:
            await active_llm_service.close()
        except:
            pass


app = FastAPI(
    title="CV Analyzer API",
    description="API for analyzing and grading CVs using LLM and web search",
    version="1.0.0",
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    llm_info = {
        "type": "ollama" if settings.USE_OLLAMA else "local",
        "model": settings.OLLAMA_MODEL if settings.USE_OLLAMA else settings.LLM_MODEL_PATH
    }

    return {
        "message": "Welcome to CV Analyzer API",
        "status": "OK",
        "environment": settings.ENVIRONMENT,
        "llm_service": llm_info
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)