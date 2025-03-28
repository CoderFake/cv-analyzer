import asyncio
import re
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json

from llama_cpp import Llama

from app.core.config import settings
from app.services.context_classifier import context_classifier
from app.services.web_search_service import web_search_service
from langdetect import detect, LangDetectException


class LLMService:

    def __init__(self):
        self.model = Llama(
            model_path=settings.LLM_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            n_batch=512
        )

        self.system_prompt = """Bạn là trợ lý AI chuyên về tuyển dụng và đánh giá CV. Bạn phân tích CV của các ứng viên để đưa ra đánh giá khách quan, hữu ích. Bạn luôn phân tích kỹ càng thông tin trước khi đưa ra nhận xét. Khi cần thiết, bạn sẽ tìm kiếm thông tin trên web để đánh giá chính xác hơn. Hãy ưu tiên sử dụng tiếng Việt trong các tìm kiếm và phân tích."""

        self.cv_evaluation_prompt = """Hãy phân tích CV này cho vị trí {position} dựa trên các yếu tố: học vấn, kinh nghiệm, kỹ năng, các thành tựu và sự phù hợp với vị trí. 

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

        self.cv_evaluation_with_search_prompt = """Hãy phân tích CV này cho vị trí {position} dựa trên các yếu tố: học vấn, kinh nghiệm, kỹ năng, các thành tựu và sự phù hợp với vị trí. 

        Nội dung CV:
        {cv_content}
        
        Thông tin thị trường tuyển dụng:
        {search_results}
        
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

        self.chat_prompt = """Dựa trên thông tin CV và cuộc hội thoại trước đó, hãy trả lời câu hỏi sau một cách chuyên nghiệp và hữu ích.

        Thông tin CV:
        {cv_content}
        
        Lịch sử trò chuyện:
        {chat_history}
        
        Câu hỏi: {question}
        
        Trả lời:"""

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return "en" if lang == "en" else "vi"
        except LangDetectException:
            return "vi"

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7
    ) -> str:

        language = self.detect_language(prompt)

        if system_prompt is None:
            if language == "en":
                system_prompt = """You are an AI assistant specializing in resume analysis and evaluation. You analyze candidates' CVs to provide objective and helpful assessments. You always analyze information thoroughly before giving feedback. When necessary, you search the web for information to provide more accurate evaluations. Please respond in English."""
            else:
                system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        reply = response["choices"][0]["message"]["content"]
        return reply

    async def evaluate_cv(
            self,
            cv_content: str,
            position: str,
            use_search: bool = True
    ) -> Dict[str, Any]:
        if use_search:
            search_query = f"Yêu cầu tuyển dụng vị trí {position} kỹ năng cần thiết gần đây nhất"

            search_task = asyncio.create_task(web_search_service.process_search_results(search_query))

            base_prompt = self.cv_evaluation_prompt.format(
                position=position,
                cv_content=cv_content
            )

            try:
                search_results = await asyncio.wait_for(search_task, timeout=5.0)

                search_text = ""
                for idx, result in enumerate(search_results.get("search_results", []), 1):
                    search_text += f"{idx}. {result['title']}: {result['content']}\n\n"

                if len(search_text) < 100:
                    prompt = base_prompt
                else:
                    prompt = self.cv_evaluation_with_search_prompt.format(
                        position=position,
                        cv_content=cv_content,
                        search_results=search_text
                    )
            except asyncio.TimeoutError:
                prompt = base_prompt
        else:
            prompt = self.cv_evaluation_prompt.format(
                position=position,
                cv_content=cv_content
            )

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
        language = self.detect_language(question)

        needs_web_search = context_classifier.needs_web_search(question)
        needs_knowledge = context_classifier.should_use_knowledge_base(question)

        chat_history_text = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                chat_history_text += f"{role.capitalize()}: {content}\n"

        search_results_text = ""
        if needs_web_search:
            try:
                search_task = asyncio.create_task(web_search_service.process_search_results(question))
                search_results = await asyncio.wait_for(search_task, timeout=4.0)

                for idx, result in enumerate(search_results.get("extracted_content", []), 1):
                    search_results_text += f"\nKết quả tìm kiếm {idx}:\n"
                    search_results_text += f"Tiêu đề: {result.get('title', 'Không có tiêu đề')}\n"
                    search_results_text += f"Nội dung: {result.get('content', 'Không có nội dung')}\n"
                    search_results_text += f"Nguồn: {result.get('url', 'Không có URL')}\n"
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Web search error or timeout: {e}")
                search_results_text = ""

        base_prompt = f"""Dựa trên thông tin CV và cuộc hội thoại trước đó, hãy trả lời câu hỏi sau một cách chuyên nghiệp và hữu ích.

        Thông tin CV:
        {cv_content or "Không có thông tin CV"}

        Lịch sử trò chuyện:
        {chat_history_text}
        """

        if needs_web_search and search_results_text:
            web_info = f"""
            Thông tin tìm kiếm web:
            {search_results_text}
            """
            base_prompt += web_info

        if language == "en":
            prompt = f"""Based on the CV information and previous conversation, please answer the following question professionally and helpfully.

            CV Information:
            {cv_content or "No CV information available"}

            Conversation history:
            {chat_history_text}
            """

            if needs_web_search and search_results_text:
                prompt += f"""
                Web search information:
                {search_results_text}
                """

            prompt += f"""
            Question: {question}

            Answer:"""
        else:
            prompt = base_prompt + f"""
            Câu hỏi: {question}

            Trả lời:"""

        temperature = 0.5 if search_results_text else 0.7

        response = await self.generate_response(
            prompt=prompt,
            temperature=temperature
        )

        return response

    async def process_document(
            self,
            document_content: str,
            document_type: str,
            user_query: str
    ) -> str:
        doc_type_explanation = ""
        if document_type.lower() in ['.pdf', '.docx', '.doc']:
            doc_type_explanation = "CV hoặc tài liệu văn bản"
        elif document_type.lower() in ['.jpg', '.jpeg', '.png']:
            doc_type_explanation = "hình ảnh"
        else:
            doc_type_explanation = "tài liệu"

        if len(document_content) > 10000:
            document_content = document_content[:10000] + "...(nội dung đã bị cắt để tiết kiệm bộ nhớ)..."

        prompt = f"""Dưới đây là nội dung từ {doc_type_explanation} được tải lên. Phân tích và trả lời câu hỏi người dùng một cách ngắn gọn, rõ ràng.

        Nội dung tài liệu:
        {document_content}

        Yêu cầu của người dùng:
        {user_query}

        Phân tích và trả lời:"""

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1024
        )

        return response

    async def process_document(
            self,
            document_content: str,
            document_type: str,
            user_query: str
    ) -> str:

        doc_type_description = "tài liệu"
        if document_type.lower() in ['.pdf', '.docx', '.doc']:
            doc_type_description = "tài liệu văn bản"
        elif document_type.lower() in ['.jpg', '.jpeg', '.png']:
            doc_type_description = "hình ảnh"
        elif document_type.lower() == '.txt':
            doc_type_description = "tệp văn bản"

        max_content_length = 10000
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "...[nội dung bị cắt do quá dài]"

        prompt = f"""Dưới đây là nội dung từ {doc_type_description} được tải lên. 
        Hãy phân tích và trả lời yêu cầu của người dùng một cách rõ ràng, chi tiết.

        Nội dung tài liệu:
        {document_content}

        Yêu cầu của người dùng:
        {user_query}

        Hãy cung cấp phân tích chi tiết, rõ ràng và hữu ích:"""

        language = self.detect_language(document_content + " " + user_query)
        system_prompt = self.system_prompt if language == "vi" else """You are an AI assistant specializing in document analysis. You analyze and provide helpful insights on documents uploaded by users. Always analyze thoroughly before giving feedback."""

        response = await self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        return response

llm_service = LLMService()