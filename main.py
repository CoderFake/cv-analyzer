import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

try:
    from app.services.ollama_llm_service import ollama_llm_service

    llm_service = ollama_llm_service
    logger.info(f"Sử dụng Ollama LLM service với mô hình {settings.OLLAMA_MODEL}")
except ImportError:
    from app.services.llm_service import llm_service

    logger.info(f"Sử dụng dịch vụ LLM cục bộ với mô hình tại {settings.LLM_MODEL_PATH}")

try:
    if settings.ENABLE_MEMORY_MONITORING:
        from app.utils.memory_monitor import memory_monitor

        use_memory_monitor = True
        logger.info(f"Giám sát bộ nhớ đã được bật với giới hạn: {settings.MAX_MEMORY_USAGE_MB}MB")
    else:
        use_memory_monitor = False
except ImportError:
    use_memory_monitor = False
    logger.warning("Không thể sử dụng giám sát bộ nhớ")


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with engine.begin() as conn:
        if settings.DEBUG:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Đã tạo bảng dữ liệu")


    if use_memory_monitor:
        memory_monitor.start()
        logger.info("Đã bắt đầu giám sát bộ nhớ")

    yield

    if use_memory_monitor:
        memory_monitor.stop()
        logger.info("Đã dừng giám sát bộ nhớ")

    if hasattr(llm_service, 'close'):
        await llm_service.close()
        logger.info("Đã đóng kết nối dịch vụ LLM")


app = FastAPI(
    title="CV Analyzer API",
    description="API phân tích và đánh giá CV sử dụng LLM và tìm kiếm web",
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
        "message": "Chào mừng đến với CV Analyzer API",
        "status": "OK",
        "environment": settings.ENVIRONMENT,
        "llm_service": llm_info
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)