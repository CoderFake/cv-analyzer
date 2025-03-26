from typing import List, Dict, Any, Optional

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class WebSearchService:
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = []
        try:
            with DDGS() as ddgs:
                for result in ddgs.text(query, region='vn-vi', safesearch='moderate', max_results=max_results):
                    results.append({
                        "title": result["title"],
                        "content": result["body"],
                        "url": result["href"]
                    })
            return results
        except Exception as e:
            print(f"Error searching with DuckDuckGo: {e}")
            return []

    async def get_webpage_content(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }

            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()

                title = soup.title.string if soup.title else "No title"
                main_content = ""
                main_elements = soup.select("main, article, section, .content, .main-content")
                if main_elements:
                    for element in main_elements:
                        main_content += element.get_text(separator=" ", strip=True) + "\n\n"
                else:
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        main_content += p.get_text(separator=" ", strip=True) + "\n"
                if not main_content.strip():
                    main_content = soup.get_text(separator=" ", strip=True)

                return {
                    "title": title,
                    "url": url,
                    "content": main_content.strip()
                }

        except Exception as e:
            print(f"Error extracting content from webpage {url}: {e}")
            return None

    async def process_search_results(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        search_results = await self.search(query, max_results)
        extracted_content = []
        if search_results:
            for result in search_results[:3]:
                content = await self.get_webpage_content(result["url"])
                if content:
                    extracted_content.append(content)

        return {
            "query": query,
            "search_results": search_results,
            "extracted_content": extracted_content
        }

web_search_service = WebSearchService()