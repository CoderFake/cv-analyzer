import asyncio
import json
import httpx
from typing import List, Dict, Any, Optional
import re
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings


class OllamaLLMService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.system_prompt = """Bạn là trợ lý AI chuyên về tuyển dụng và đánh giá CV. Bạn phân tích CV của các ứng viên để đưa ra đánh giá khách quan, hữu ích. Bạn luôn phân tích kỹ càng thông tin trước khi đưa ra nhận xét. Khi cần thiết, bạn sẽ tìm kiếm thông tin trên web để đánh giá chính xác hơn. Hãy ưu tiên sử dụng tiếng Việt trong các tìm kiếm và phân tích."""
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.client = httpx.AsyncClient(timeout=60.0)

        self._check_ollama_connection()

    def _check_ollama_connection(self):
        import requests
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
            print(f"Successfully connected to Ollama at {self.base_url}")

            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            if not any(model.get("name") == self.model for model in models):
                print(f"Model {self.model} not found, attempting to pull...")
                requests.post(f"{self.base_url}/api/pull", json={"name": self.model})
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure the Ollama service is running and accessible")

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7
    ) -> str:
        if system_prompt is None:
            system_prompt = self.system_prompt

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["message"]["content"]
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."

    async def evaluate_cv(
            self,
            cv_content: str,
            position: str,
            use_search: bool = True
    ) -> Dict[str, Any]:
        from app.services.web_search_service import web_search_service

        if use_search:
            search_task = asyncio.create_task(
                web_search_service.process_search_results(f"Yêu cầu tuyển dụng vị trí {position} và kỹ năng cần thiết")
            )

            search_results = await search_task

            search_text = ""
            for idx, result in enumerate(search_results.get("search_results", []), 1):
                search_text += f"{idx}. {result['title']}: {result['content']}\n\n"

            prompt = f"""Hãy phân tích CV này cho vị trí {position} dựa trên các yếu tố: học vấn, kinh nghiệm, kỹ năng, các thành tựu và sự phù hợp với vị trí. 

            Nội dung CV:
            {cv_content}

            Thông tin thị trường tuyển dụng:
            {search_text}

            Vui lòng đánh giá theo thang điểm sau:
            A: Xuất sắc - Vượt trội, phù hợp hoàn hảo với vị trí
            B: Tốt - Đáp ứng tốt yêu cầu, có nhiều điểm mạnh
            C: Trung bình - Đáp ứng được các yêu cầu cơ bản
            D: Dưới trung bình - Còn thiếu nhiều kỹ năng/kinh nghiệm cần thiết
            E: Không phù hợp - Không đáp ứng được các yêu cầu của vị trí

            Trả về kết quả theo cấu trúc:
            - Đánh giá tổng thể (thang điểm A-E): [thang điểm]
            - Nhận xét chi tiết: [nhận xét]
            - Tóm tắt: [tóm tắt ngắn gọn]
            - Điểm mạnh: [điểm mạnh]
            - Điểm yếu: [điểm yếu]
            - Khuyến nghị: [khuyến nghị cải thiện]
            """
        else:
            prompt = f"""Hãy phân tích CV này cho vị trí {position} dựa trên các yếu tố: học vấn, kinh nghiệm, kỹ năng, các thành tựu và sự phù hợp với vị trí. 

            Nội dung CV:
            {cv_content}

            Vui lòng đánh giá theo thang điểm sau:
            A: Xuất sắc - Vượt trội, phù hợp hoàn hảo với vị trí
            B: Tốt - Đáp ứng tốt yêu cầu, có nhiều điểm mạnh
            C: Trung bình - Đáp ứng được các yêu cầu cơ bản
            D: Dưới trung bình - Còn thiếu nhiều kỹ năng/kinh nghiệm cần thiết
            E: Không phù hợp - Không đáp ứng được các yêu cầu của vị trí

            Trả về kết quả theo cấu trúc:
            - Đánh giá tổng thể (thang điểm A-E): [thang điểm]
            - Nhận xét chi tiết: [nhận xét]
            - Tóm tắt: [tóm tắt ngắn gọn]
            - Điểm mạnh: [điểm mạnh]
            - Điểm yếu: [điểm yếu]
            - Khuyến nghị: [khuyến nghị cải thiện]
            """

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.3
        )

        eval_result = self._parse_cv_evaluation(response)
        return eval_result

    async def chat_completion(
            self,
            question: str,
            chat_history: List[Dict[str, str]] = None,
            cv_content: Optional[str] = None
    ) -> str:
        chat_history_text = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                chat_history_text += f"{role.capitalize()}: {content}\n"

        prompt = f"""Dựa trên thông tin CV và cuộc hội thoại trước đó, hãy trả lời câu hỏi sau một cách chuyên nghiệp và hữu ích.

        Thông tin CV:
        {cv_content or "Không có thông tin CV"}

        Lịch sử trò chuyện:
        {chat_history_text}

        Câu hỏi: {question}

        Trả lời:"""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._sync_generate_response(prompt, temperature=0.7)
        )

    def _sync_generate_response(self, prompt, temperature=0.7):
        """Synchronous version of generate_response for thread pool execution"""
        import requests

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            return response_data["message"]["content"]
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."

    async def answer_with_knowledge(
            self,
            question: str,
            context: str,
            category: Optional[str] = None
    ) -> str:
        prompt = f"""Hãy trả lời câu hỏi dựa trên thông tin trong các tài liệu sau. 
        Nếu thông tin trong tài liệu không đủ để trả lời, hãy cho biết bạn không có đủ thông tin và trả lời dựa trên kiến thức của bạn.

        Tài liệu:
        {context}

        Câu hỏi: {question}

        Trả lời:"""

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.3
        )

        return response

    def _parse_cv_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        result = {
            "grade": "C",
            "evaluation": evaluation_text,
            "summary": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        grade_match = re.search(r"(Đánh giá tổng thể|thang điểm).*?[:]\s*([A-E])", evaluation_text, re.IGNORECASE)
        if grade_match:
            result["grade"] = grade_match.group(2).upper()

        summary_match = re.search(r"(Tóm tắt|Summary)[:]\s*(.*?)(?:\n|$)", evaluation_text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            result["summary"] = summary_match.group(2).strip()

        strengths_section = re.search(
            r"(Điểm mạnh|Strengths)[:]\s*(.*?)(?=Điểm yếu|Weaknesses|Khuyến nghị|Recommendations|$)", evaluation_text,
            re.IGNORECASE | re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(2).strip()
            strengths = re.findall(r"(?:^|\n)[-•*]?\s*(.*?)(?:\n|$)", strengths_text)
            result["strengths"] = [s.strip() for s in strengths if s.strip()]

        weaknesses_section = re.search(r"(Điểm yếu|Weaknesses)[:]\s*(.*?)(?=Khuyến nghị|Recommendations|$)",
                                       evaluation_text, re.IGNORECASE | re.DOTALL)
        if weaknesses_section:
            weaknesses_text = weaknesses_section.group(2).strip()
            weaknesses = re.findall(r"(?:^|\n)[-•*]?\s*(.*?)(?:\n|$)", weaknesses_text)
            result["weaknesses"] = [w.strip() for w in weaknesses if w.strip()]

        recommendations_section = re.search(r"(Khuyến nghị|Recommendations)[:]\s*(.*?)(?=$)", evaluation_text,
                                            re.IGNORECASE | re.DOTALL)
        if recommendations_section:
            recommendations_text = recommendations_section.group(2).strip()
            recommendations = re.findall(r"(?:^|\n)[-•*]?\s*(.*?)(?:\n|$)", recommendations_text)
            result["recommendations"] = [r.strip() for r in recommendations if r.strip()]

        return result

    async def close(self):
        await self.client.aclose()


ollama_llm_service = OllamaLLMService()