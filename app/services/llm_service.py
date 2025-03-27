import re
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json

from llama_cpp import Llama

from app.core.config import settings
from app.services.web_search_service import web_search_service


class LLMService:

    def __init__(self):
        self.model = Llama(
            model_path=settings.LLM_MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1
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

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7
    ) -> str:
        if system_prompt is None:
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
            search_query = f"Yêu cầu tuyển dụng vị trí {position} và kỹ năng cần thiết"

            search_results = await web_search_service.process_search_results(search_query)

            search_text = ""
            for idx, result in enumerate(search_results.get("search_results", []), 1):
                search_text += f"{idx}. {result['title']}: {result['content']}\n\n"
            prompt = self.cv_evaluation_with_search_prompt.format(
                position=position,
                cv_content=cv_content,
                search_results=search_text
            )
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
        chat_history_text = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                chat_history_text += f"{role.capitalize()}: {content}\n"

        prompt = self.chat_prompt.format(
            cv_content=cv_content or "Không có thông tin CV",
            chat_history=chat_history_text,
            question=question
        )
        response = await self.generate_response(
            prompt=prompt,
            temperature=0.7
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

llm_service = LLMService()