import os
import sys


import importlib.util
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web-search-adapter")


def is_llm_web_search_installed():
    try:
        spec = importlib.util.find_spec("llm_web_search")
        return spec is not None
    except ImportError:
        return False

try:
    if is_llm_web_search_installed():
        from llm_web_search import retrieve_from_duckduckgo, Generator
        from retrieval import DocumentRetriever

        document_retriever = DocumentRetriever(
            device="cpu",
            num_results=10,
            similarity_threshold=0.5,
            chunk_size=500,
            ensemble_weighting=0.5,
            keyword_retriever="bm25",
            chunking_method="character-based"
        )

        logger.info("LLM_Web_search được tìm thấy. Sử dụng tích hợp với LLM_Web_search.")
        USING_LLM_WEB_SEARCH = True
    else:
        logger.warning("LLM_Web_search không được tìm thấy. Sử dụng tìm kiếm web tích hợp sẵn.")
        USING_LLM_WEB_SEARCH = False
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo LLM_Web_search: {str(e)}")
    USING_LLM_WEB_SEARCH = False


async def search_with_llm_web_search(query: str, max_results: int = 5):
    if not USING_LLM_WEB_SEARCH:
        logger.error("LLM_Web_search không được cài đặt. Không thể sử dụng.")
        return []

    try:
        gen = Generator(retrieve_from_duckduckgo(
            query,
            document_retriever,
            max_results=max_results,
            instant_answers=True,
            simple_search=False
        ))

        messages = []
        for status_message in gen:
            messages.append(status_message)

        search_results = gen.retval

        results = []
        for doc in search_results:
            results.append({
                "title": doc.metadata.get("source", "Unknown Source"),
                "content": doc.page_content,
                "url": doc.metadata.get("source", "#")
            })

        return results
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm với LLM_Web_search: {str(e)}")
        return []


async def search_web(query: str, max_results: int = 5):
    if USING_LLM_WEB_SEARCH:
        return await search_with_llm_web_search(query, max_results)
    else:
        from app.services.web_search_service import web_search_service
        return await web_search_service.search(query, max_results)