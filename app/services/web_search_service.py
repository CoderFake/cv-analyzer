from typing import List, Dict, Any, Optional
import asyncio
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from app.services.web_search_adapter import search_web, USING_LLM_WEB_SEARCH


class WebSearchService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=5.0)
        self.max_concurrent_requests = 3
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    async def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        results = []
        try:
            if USING_LLM_WEB_SEARCH:
                return await search_web(query, max_results)
            else:
                with DDGS() as ddgs:
                    count = 0
                    for result in ddgs.text(query, region='vn-vi', safesearch='moderate', max_results=max_results * 2):
                        if count >= max_results:
                            break

                        if result.get("body") and len(result["body"]) > 50:
                            results.append({
                                "title": result["title"],
                                "content": result["body"],
                                "url": result["href"]
                            })
                            count += 1

                return results
        except Exception as e:
            print(f"Error searching with DuckDuckGo: {e}")
            return []

    async def get_webpage_content(self, url: str) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
                }

                async with httpx.AsyncClient(headers=headers, timeout=5.0, follow_redirects=True) as client:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                    except (httpx.TimeoutException, httpx.HTTPError, httpx.ConnectError):
                        if "://" in url:
                            base_url = url.split("://", 1)[1]
                            try:
                                response = await client.get(f"https://{base_url}")
                                response.raise_for_status()
                            except:
                                return None
                        else:
                            return None

                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("text/html"):
                        return None

                    soup = BeautifulSoup(response.text, 'html.parser')
                    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                        script_or_style.decompose()

                    title = soup.title.string if soup.title else "No title"

                    main_content = ""
                    main_elements = soup.select("main, article, section, .content, .main-content")
                    if main_elements:
                        for element in main_elements:
                            paragraphs = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                            for p in paragraphs:
                                text = p.get_text(strip=True)
                                if text and len(text) > 20:  # Chỉ lấy đoạn có nghĩa
                                    main_content += text + "\n\n"
                    else:
                        paragraphs = soup.find_all('p')
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if text and len(text) > 20:
                                main_content += text + "\n\n"

                    if len(main_content) < 100:
                        main_content = "\n\n".join([
                            p.get_text(strip=True)
                            for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                            if len(p.get_text(strip=True)) > 20
                        ])

                    if len(main_content) > 1500:
                        main_content = main_content[:1500] + "..."

                    return {
                        "title": title,
                        "url": url,
                        "content": main_content.strip()
                    }

            except Exception as e:
                print(f"Error extracting content from webpage {url}: {e}")
                return None

    async def process_search_results(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        try:
            search_task = asyncio.create_task(self.search(query, max_results))
            search_results = await asyncio.wait_for(search_task, timeout=3.0)

            extraction_tasks = []
            for result in search_results[:2]:
                task = asyncio.create_task(self.get_webpage_content(result["url"]))
                extraction_tasks.append(task)

            extracted_contents = await asyncio.wait_for(
                asyncio.gather(*extraction_tasks, return_exceptions=True),
                timeout=4.0
            )

            extracted_content = [
                content for content in extracted_contents
                if content and not isinstance(content, Exception) and len(content.get("content", "")) > 100
            ]

            return {
                "query": query,
                "search_results": search_results,
                "extracted_content": extracted_content
            }
        except asyncio.TimeoutError:
            return {
                "query": query,
                "search_results": search_results if 'search_results' in locals() else [],
                "extracted_content": []
            }
        except Exception as e:
            print(f"Error in process_search_results: {e}")
            return {
                "query": query,
                "search_results": [],
                "extracted_content": []
            }

    async def close(self):
        await self.client.aclose()


web_search_service = WebSearchService()