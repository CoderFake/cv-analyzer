import asyncio
import os
import re
import base64
import json
import tempfile
from typing import List, Dict, Any, Optional, Union
import httpx
import requests

from app.core.config import settings
from app.services.context_classifier import context_classifier
from app.services.web_search_service import web_search_service
from app.services.llm_service import llm_service
from langdetect import detect, LangDetectException

# Cấu hình API Gemini
GEMINI_API_KEY = settings.GEMINI_API_KEY or ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"


class IntegratedLLMService:
    """Dịch vụ tích hợp Ollama Vision, Gemini và tìm kiếm web để phân tích CV"""

    def __init__(self):
        self.ollama_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=30.0)
        self.system_prompt_vi = """Bạn là trợ lý AI chuyên về tuyển dụng và đánh giá CV. Bạn phân tích CV của các ứng viên để đưa ra đánh giá khách quan, hữu ích. Bạn luôn phân tích kỹ càng thông tin trước khi đưa ra nhận xét."""
        self.system_prompt_en = """You are an AI assistant specializing in recruitment and CV evaluation. You analyze candidates' CVs to provide objective and helpful assessments. You always analyze information thoroughly before giving feedback."""
        self.use_gemini = bool(GEMINI_API_KEY) and settings.USE_GEMINI

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return "en" if lang == "en" else "vi"
        except LangDetectException:
            return "vi"  # Mặc định tiếng Việt

    async def process_image_with_ollama(self, file_path: str) -> str:
        """Sử dụng Ollama Vision API để trích xuất thông tin từ hình ảnh"""
        try:
            # Đọc file ảnh và chuyển thành base64
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Chuẩn bị payload cho API Ollama
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hãy trích xuất tất cả thông tin quan trọng từ CV này, bao gồm thông tin cá nhân, học vấn, kinh nghiệm làm việc, kỹ năng, chứng chỉ và các thông tin liên quan khác. Trả về dưới dạng văn bản có cấu trúc.",
                        "images": [base64_image]
                    }
                ],
                "stream": False
            }

            # Gọi API Ollama Vision
            response = await self.client.post(
                f"{self.ollama_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            return result["message"]["content"]
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh với Ollama Vision: {e}")
            return f"Không thể trích xuất thông tin từ ảnh CV: {str(e)}"

    async def process_file_content(self, file_path: str, file_content: str, file_type: str, query: str,
                                   cv_content: Optional[str] = None, knowledge_content: Optional[str] = None) -> str:
        """Xử lý nội dung file dựa vào loại file"""

        # Xử lý file ảnh bằng Ollama Vision
        if file_type.lower() in ['.jpg', '.jpeg', '.png']:
            extracted_text = await self.process_image_with_ollama(file_path)
        else:
            # Đối với các file văn bản, sử dụng nội dung đã được trích xuất
            extracted_text = file_content

        # Tìm kiếm thông tin bổ sung từ web nếu cần
        needs_web_search = context_classifier.needs_web_search(query)
        if needs_web_search:
            search_results = await web_search_service.process_search_results(query)
            search_text = ""
            for idx, result in enumerate(search_results.get("search_results", []), 1):
                search_text += f"{idx}. {result['title']}: {result['content']}\n\n"
        else:
            search_text = ""

        # Xác định ngôn ngữ
        language = self.detect_language(query + " " + extracted_text[:200])

        # Kiểm tra nếu có CV
        has_cv = cv_content is not None and len(cv_content.strip()) > 0

        # Kiểm tra nếu có knowledge
        has_knowledge = knowledge_content is not None and len(knowledge_content.strip()) > 0

        # Tạo prompt dựa vào ngôn ngữ
        if language == "en":
            prompt = f"""Please analyze the following content and answer the user's query.

            File Content:
            {extracted_text}

            {"CV Content:" + cv_content if has_cv else ""}
            {"Knowledge Base:" + knowledge_content if has_knowledge else ""}
            {"Web Search Results:" + search_text if search_text else ""}

            User Query:
            {query}

            Answer:"""
            system_prompt = self.system_prompt_en
        else:
            prompt = f"""Hãy phân tích nội dung sau và trả lời câu hỏi của người dùng.

            Nội dung File:
            {extracted_text}

            {"Nội dung CV:" + cv_content if has_cv else ""}
            {"Cơ sở kiến thức:" + knowledge_content if has_knowledge else ""}
            {"Kết quả tìm kiếm web:" + search_text if search_text else ""}

            Câu hỏi của người dùng:
            {query}

            Trả lời:"""
            system_prompt = self.system_prompt_vi

        # Sử dụng Gemini nếu có cấu hình, hoặc fallback sang LLM service
        if self.use_gemini:
            response = await self.call_gemini_api(prompt, system_prompt)
        else:
            response = await llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

        return response

    async def chat_with_knowledge(
            self,
            question: str,
            chat_history: List[Dict[str, str]] = None,
            cv_content: Optional[str] = None,
            file_path: Optional[str] = None,
            file_type: Optional[str] = None,
            knowledge_content: Optional[str] = None
    ) -> str:
        """Xử lý chat với CV, file đính kèm và knowledge base"""

        # Xác định ngôn ngữ
        language = self.detect_language(question)
        system_prompt = self.system_prompt_vi if language == "vi" else self.system_prompt_en

        # Kiểm tra xem có file đính kèm không
        has_file = file_path is not None and os.path.exists(file_path)

        # Kiểm tra xem có CV không
        has_cv = cv_content is not None and len(cv_content.strip()) > 0

        # Kiểm tra xem có knowledge không
        has_knowledge = knowledge_content is not None and len(knowledge_content.strip()) > 0

        # Tìm kiếm web nếu cần
        needs_web_search = context_classifier.needs_web_search(question)

        # Nếu câu hỏi không liên quan đến CV hoặc không có CV
        if not self._is_cv_related(question) and not has_cv and not has_file and not has_knowledge:
            # Trả lời mặc định
            if language == "en":
                return "I'm sorry, I can only answer questions about CVs, recruitment, and topics in our knowledge base. If you have a CV to analyze, please upload it or ask a question related to job applications and recruitment."
            else:
                return "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về CV, tuyển dụng và chủ đề trong cơ sở kiến thức của chúng tôi. Nếu bạn có CV cần phân tích, vui lòng tải lên hoặc đặt câu hỏi liên quan đến ứng tuyển và tuyển dụng."

        # Chuẩn bị lịch sử trò chuyện
        chat_history_text = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                chat_history_text += f"{role.capitalize()}: {content}\n"

        # Tìm kiếm thông tin từ web nếu cần
        search_results_text = ""
        if needs_web_search:
            try:
                search_task = asyncio.create_task(web_search_service.process_search_results(question))
                search_results = await asyncio.wait_for(search_task, timeout=5.0)

                for idx, result in enumerate(search_results.get("search_results", []), 1):
                    search_results_text += f"{idx}. {result['title']}: {result['content']}\n\n"
            except Exception as e:
                print(f"Lỗi khi tìm kiếm web: {e}")

        # Xử lý file nếu có
        file_content = ""
        if has_file:
            # Xử lý file dựa vào loại file
            if file_type.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    file_content = await self.process_image_with_ollama(file_path)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh: {e}")
                    file_content = "Không thể trích xuất thông tin từ file ảnh."

        # Tạo prompt dựa vào ngôn ngữ và dữ liệu có sẵn
        if language == "en":
            prompt = f"""Based on the available information, please answer the user's question.

            {"CV Content:" + cv_content if has_cv else ""}
            {"File Content:" + file_content if file_content else ""}
            {"Knowledge Base:" + knowledge_content if has_knowledge else ""}
            {"Chat History:" + chat_history_text if chat_history_text else ""}
            {"Web Search Results:" + search_results_text if search_results_text else ""}

            User Question: {question}

            Answer:"""
        else:
            prompt = f"""Dựa trên thông tin có sẵn, hãy trả lời câu hỏi của người dùng."

            "{"Nội dung CV:" + cv_content if has_cv else ""}"
            "{"Nội dung File:" + file_content if file_content else ""}"
            "{"Cơ sở kiến thức:" + knowledge_content if has_knowledge else ""}"
            "{"Lịch sử trò chuyện:" + chat_history_text if chat_history_text else ""}"
            "{"Kết quả tìm kiếm web:" + search_results_text if search_results_text else ""}"

            Câu hỏi: {question}

            Trả lời:"""

        # Sử dụng Gemini nếu có cấu hình, hoặc fallback sang LLM service
        if self.use_gemini:
            response = await self.call_gemini_api(prompt, system_prompt)
        else:
            response = await llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3 if (has_cv or file_content or search_results_text or has_knowledge) else 0.7
            )

        return response

    async def call_gemini_api(self, prompt: str, system_prompt: str) -> str:
        """Gọi API Gemini để phân tích và trả lời"""
        if not GEMINI_API_KEY:
            # Fallback sang LLM service nếu không có API key
            return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)

        try:
            # Chuẩn bị payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {"text": system_prompt}
                    ]
                },
                "generationConfig": {
                    "temperature": 0.4,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 4096
                }
            }

            # Gọi API
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url=url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                # Xử lý phản hồi
                if response.status_code == 200:
                    resp_data = response.json()
                    if "candidates" in resp_data and len(resp_data["candidates"]) > 0:
                        text_parts = []
                        for part in resp_data["candidates"][0]["content"]["parts"]:
                            if "text" in part:
                                text_parts.append(part["text"])
                        return "".join(text_parts)
                    else:
                        return "Không nhận được phản hồi hợp lệ từ API."
                else:
                    # Xử lý lỗi API
                    error_detail = f"Lỗi API ({response.status_code}): {response.text}"
                    print(error_detail)
                    # Fallback sang LLM service
                    return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)

        except Exception as e:
            print(f"Lỗi khi gọi Gemini API: {e}")
            # Fallback sang LLM service
            return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _is_cv_related(self, question: str) -> bool:
        """Kiểm tra xem câu hỏi có liên quan đến CV và tuyển dụng không"""
        question = question.lower()
        cv_keywords = [
            'cv', 'resume', 'job', 'career', 'interview', 'skill', 'experience',
            'education', 'qualification', 'recruitment', 'hiring', 'application',
            'hồ sơ', 'ứng tuyển', 'tuyển dụng', 'việc làm', 'kỹ năng', 'kinh nghiệm',
            'học vấn', 'phỏng vấn', 'nghề nghiệp', 'công việc', 'ứng viên'
        ]

        for keyword in cv_keywords:
            if keyword in question:
                return True

        return False


# Singleton instance
integrated_llm_service = IntegratedLLMService()