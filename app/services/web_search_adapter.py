import logging
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web-search-adapter")


try:
    from app.lib.llm_web_search.llm_web_search import retrieve_from_duckduckgo, Generator
    from app.lib.llm_web_search.retrieval import DocumentRetriever, docs_to_pretty_str
    USING_LLM_WEB_SEARCH = True
    print("Sử dụng LLM_Web_search thành công!")
except ImportError as e:
    print(f"Chi tiết lỗi import: {str(e)}")
    USING_LLM_WEB_SEARCH = False


async def search_with_llm_web_search(query: str, max_results: int = 5):
    if not USING_LLM_WEB_SEARCH:
        logger.error("LLM_Web_search không được cài đặt hoặc không thể sử dụng.")
        return []

    try:
        if settings.SEARCH_BACKEND == "searxng" and settings.SEARXNG_URL:
            gen = Generator(retrieve_from_searxng(
                query,
                settings.SEARXNG_URL,
                document_retriever,
                max_results=max_results,
                instant_answers=True,
                simple_search=False
            ))
        else:
            # Mặc định sử dụng DuckDuckGo với region Việt Nam
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