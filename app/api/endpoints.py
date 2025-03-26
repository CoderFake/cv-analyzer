from fastapi import APIRouter

from app.api import auth, candidates, chat, files, knowledge

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["authentication"])
router.include_router(candidates.router, prefix="/candidates", tags=["candidates"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(files.router, prefix="/files", tags=["files"])
router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])