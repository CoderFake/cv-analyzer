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

# Import thư viện từ lib/llm_web_search
try:
    from app.lib.llm_web_search.llm_web_search import retrieve_from_duckduckgo, Generator
    from app.lib.llm_web_search.retrieval import DocumentRetriever, docs_to_pretty_str
    from app.lib.llm_web_search.retrievers.faiss_retriever import FaissRetriever
    from app.lib.llm_web_search.retrievers.bm25_retriever import BM25Retriever
    from app.lib.llm_web_search.utils import Document, MySentenceTransformer

    USING_LLM_WEB_SEARCH = True
except ImportError as e:
    print(f"Lỗi import thư viện llm_web_search: {e}")
    USING_LLM_WEB_SEARCH = False

# Cấu hình API Gemini
GEMINI_API_KEY = settings.GEMINI_API_KEY or ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"


class IntegratedLLMService:
    """Dịch vụ tích hợp LLM để phân tích CV và trả lời câu hỏi liên quan đến tuyển dụng"""

    def __init__(self):
        self.ollama_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=30.0)
        self.system_prompt_vi = """Bạn là trợ lý AI chuyên về tuyển dụng và đánh giá CV. Bạn phân tích CV của các ứng viên để đưa ra đánh giá khách quan, hữu ích. Bạn luôn phân tích kỹ càng thông tin trước khi đưa ra nhận xét."""
        self.system_prompt_en = """You are an AI assistant specializing in recruitment and CV evaluation. You analyze candidates' CVs to provide objective and helpful assessments. You always analyze information thoroughly before giving feedback."""
        self.use_gemini = bool(GEMINI_API_KEY) and settings.USE_GEMINI

        # Khởi tạo DocumentRetriever và embedding model nếu dùng llm_web_search
        self.document_retriever = None
        self.embedding_model = None

        if USING_LLM_WEB_SEARCH:
            try:
                import torch
                from app.lib.llm_web_search.utils import MySentenceTransformer

                print("Đang khởi tạo MySentenceTransformer...")
                self.embedding_model = MySentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu",
                    model_kwargs={"torch_dtype": torch.float32}
                )

                print("Đang khởi tạo DocumentRetriever...")
                from app.lib.llm_web_search.retrieval import DocumentRetriever
                self.document_retriever = DocumentRetriever(
                    device="cpu",
                    num_results=5,
                    similarity_threshold=0.5,
                    chunk_size=500,
                    keyword_retriever="bm25"
                )
                print("Đã khởi tạo DocumentRetriever và embedding model thành công")
            except Exception as e:
                print(f"Lỗi khởi tạo retriever: {e}")
                self.document_retriever = None
                self.embedding_model = None

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return "en" if lang == "en" else "vi"
        except LangDetectException:
            return "vi"  # Mặc định tiếng Việt

    async def extract_content_from_image(self, file_path: str) -> str:
        """Sử dụng Ollama Vision API để trích xuất thông tin từ hình ảnh"""
        try:
            print(f"===== BẮT ĐẦU TRÍCH XUẤT NỘI DUNG TỪ ẢNH: {file_path} =====")
            # Đọc file ảnh và chuyển thành base64
            with open(file_path, "rb") as image_file:
                file_data = image_file.read()
                file_size = len(file_data)
                print(f"Kích thước file: {file_size} bytes")
                base64_image = base64.b64encode(file_data).decode("utf-8")
                print(f"Đã chuyển đổi thành base64, độ dài chuỗi: {len(base64_image)}")

            # Chuẩn bị payload cho API Ollama
            prompt_message = "Hãy trích xuất tất cả thông tin quan trọng từ CV này, bao gồm thông tin cá nhân, học vấn, kinh nghiệm làm việc, kỹ năng, chứng chỉ và các thông tin liên quan khác. Trả về dưới dạng văn bản có cấu trúc."
            print(f"Prompt: {prompt_message}")

            payload = {
                "model": self.ollama_model,
                "prompt": prompt_message,
                "images": [base64_image],
                "stream": False
            }
            print(f"Sử dụng model: {self.ollama_model}")

            # Gọi API Ollama
            print(f"Gửi request đến: {self.ollama_url}/api/generate")
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "")
            print(f"Nội dung trích xuất được ({len(content)} ký tự):")
            print("============ BẮT ĐẦU NỘI DUNG TRÍCH XUẤT ============")
            print(content[:500] + "..." if len(content) > 500 else content)
            print("============= KẾT THÚC NỘI DUNG TRÍCH XUẤT =============")

            return content
        except Exception as e:
            error_message = f"Lỗi khi xử lý ảnh với Ollama: {e}"
            print(f"===== LỖI TRÍCH XUẤT: {error_message} =====")
            # Thử phương án dự phòng
            try:
                print("Thử phương án dự phòng với OCR đơn giản...")
                # Nếu có thể, sử dụng một cách đơn giản hơn để trích xuất text
                from PIL import Image
                import pytesseract

                image = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(image)

                print(f"OCR đã trích xuất được ({len(extracted_text)} ký tự):")
                print("============ BẮT ĐẦU NỘI DUNG OCR ============")
                print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                print("============= KẾT THÚC NỘI DUNG OCR =============")

                if len(extracted_text) > 50:
                    return extracted_text
            except Exception as ocr_error:
                print(f"Phương án dự phòng OCR cũng thất bại: {ocr_error}")

            return f"Không thể trích xuất thông tin từ ảnh CV: {str(e)}"

    async def search_with_lib(self, query: str, content: str = None) -> str:
        """Tìm kiếm thông tin dùng thư viện llm_web_search"""
        print(f"Tìm kiếm thông tin cho query: {query}")

        # Luôn bắt đầu với web_search_service làm phương án an toàn
        try:
            print("Bắt đầu tìm kiếm với web_search_service")
            results = await web_search_service.process_search_results(query)
            basic_search_text = ""
            for idx, result in enumerate(results.get("search_results", []), 1):
                basic_search_text += f"{idx}. {result['title']}: {result['content']}\n\n"
            print(f"Tìm kiếm cơ bản hoàn tất, độ dài kết quả: {len(basic_search_text)}")
        except Exception as e:
            print(f"Lỗi khi tìm kiếm cơ bản: {e}")
            basic_search_text = ""

        # Nếu không sử dụng llm_web_search, trả về kết quả cơ bản
        if not USING_LLM_WEB_SEARCH or not self.document_retriever:
            print("Không sử dụng llm_web_search, trả về kết quả tìm kiếm cơ bản")
            return basic_search_text

        # Thử tìm kiếm với llm_web_search
        try:
            from app.lib.llm_web_search.utils import Document
            import torch  # Đảm bảo torch được import

            print("Bắt đầu tìm kiếm với llm_web_search")

            # Tìm kiếm web với lib
            search_generator = Generator(retrieve_from_duckduckgo(
                query,
                self.document_retriever,
                max_results=5,
                instant_answers=True,
                simple_search=False
            ))

            # Collect status messages
            for status in search_generator:
                print(f"Tìm kiếm web: {status}")

            # Get search results
            search_results = search_generator.retval

            # Nếu có content, thử tìm kiếm thông tin từ content
            if content and len(content) > 100:
                try:
                    print("Phân tích nội dung đã cung cấp")
                    # Tạo documents từ content
                    content_chunks = []
                    text_splitter = self.document_retriever._merge_splits([content], "\n")
                    for chunk in text_splitter:
                        content_chunks.append(Document(page_content=chunk, metadata={"source": "cv_content"}))

                    # Sử dụng BM25 để tìm kiếm trong content
                    print(f"Tạo BM25Retriever với {len(content_chunks)} chunks")
                    retriever = BM25Retriever.from_documents(content_chunks)
                    content_results = retriever.get_relevant_documents(query)
                    print(f"Tìm thấy {len(content_results)} kết quả liên quan trong nội dung")

                    # Kết hợp kết quả
                    search_results.extend(content_results)
                except Exception as e:
                    print(f"Lỗi khi tìm kiếm trong content: {e}")

            # Format kết quả
            if search_results:
                print(f"Định dạng {len(search_results)} kết quả tìm kiếm")
                formatted_results = docs_to_pretty_str(search_results)
                print(f"Kết quả tìm kiếm nâng cao, độ dài: {len(formatted_results)}")
                return formatted_results
            else:
                print("Không có kết quả tìm kiếm nâng cao, trả về kết quả cơ bản")
                return basic_search_text

        except Exception as e:
            print(f"Lỗi khi tìm kiếm với llm_web_search: {e}")
            # Trả về kết quả cơ bản nếu có lỗi với llm_web_search
            return basic_search_text

    async def process_file_content(self, file_path: str, file_content: str, file_type: str, query: str,
                                   cv_content: Optional[str] = None, knowledge_content: Optional[str] = None) -> str:
        """Xử lý nội dung file với LLM local và web search trước khi tổng hợp kết quả"""
        print(f"Bắt đầu xử lý file: {file_path}")

        # Bước 1: Trích xuất nội dung file
        extracted_text = ""
        if file_type.lower() in ['.jpg', '.jpeg', '.png']:
            # Đối với file ảnh, sử dụng Ollama Vision để trích xuất
            extracted_text = await self.extract_content_from_image(file_path)
            print(f"Đã trích xuất nội dung từ ảnh, độ dài: {len(extracted_text)}")
        else:
            # Đối với các file văn bản, sử dụng nội dung đã được trích xuất
            extracted_text = file_content
            print(f"Đã lấy nội dung từ file văn bản, độ dài: {len(extracted_text)}")

            # Chi tiết hóa nội dung để gỡ lỗi
            print("======= CHI TIẾT NỘI DUNG FILE =======")
            print(extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text)
            print("======= KẾT THÚC CHI TIẾT NỘI DUNG FILE =======")

        # Nếu không thể trích xuất được nội dung, trả về thông báo lỗi
        if not extracted_text or len(extracted_text) < 20:
            return "Không thể trích xuất nội dung từ file. Vui lòng tải lên lại hoặc cung cấp file khác."

        # Tạo query cho việc tìm kiếm
        position = ""
        # Trích xuất vị trí công việc từ nội dung CV
        position_matches = re.search(r"(?:vị trí|position|job title)[:\s]*([^\n.,]+)", extracted_text, re.IGNORECASE)
        if position_matches:
            position = position_matches.group(1).strip()

        search_query = position or query
        if not search_query or len(search_query) < 5:
            search_query = "Yêu cầu tuyển dụng CV kỹ năng cần thiết"

        print(f"Tìm kiếm với query: {search_query}")

        # Bước 2: Tìm kiếm thông tin bổ sung từ web sử dụng lib
        try:
            search_text = await self.search_with_lib(search_query, extracted_text)
            print(f"Kết quả tìm kiếm, độ dài: {len(search_text)}")
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")
            search_text = "Không thể tìm kiếm thông tin bổ sung do lỗi hệ thống."

        # Bước 3: Phân tích nội dung CV bằng LLM local trước
        try:
            analysis_prompt = f"Phân tích chi tiết CV sau đây, đưa ra những điểm mạnh và điểm yếu chính: {extracted_text[:4000]}"
            local_analysis = await llm_service.generate_response(
                prompt=analysis_prompt,
                temperature=0.3
            )
            print(f"Đã phân tích sơ bộ bằng LLM local, độ dài: {len(local_analysis)}")
        except Exception as e:
            print(f"Lỗi khi phân tích CV bằng LLM local: {e}")
            local_analysis = "Không thể thực hiện phân tích sơ bộ do lỗi hệ thống."

        # Bước 4: Tạo prompt đánh giá tổng hợp
        language = self.detect_language(query + " " + extracted_text[:200])

        # Sử dụng prompt có sẵn để đánh giá CV, kèm phân tích từ LLM local
        cv_evaluation_prompt = f"""Hãy phân tích CV này dựa trên các yếu tố: học vấn, kinh nghiệm, kỹ năng, các thành tựu và sự phù hợp với vị trí. 

        Phân tích sơ bộ:
        {local_analysis}

        Thông tin thị trường tuyển dụng:
        {search_text}

        Vui lòng đánh giá theo thang điểm sau:
        A: Xuất sắc - Vượt trội, phù hợp hoàn hảo với vị trí
        B: Tốt - Đáp ứng tốt yêu cầu, có nhiều điểm mạnh
        C: Trung bình - Đáp ứng được các yêu cầu cơ bản
        D: Dưới trung bình - Còn thiếu nhiều kỹ năng/kinh nghiệm cần thiết
        F: Không phù hợp - Không đáp ứng được các yêu cầu của vị trí

        Trả về kết quả theo cấu trúc:
        - Đánh giá tổng thể (thang điểm A-E): [thang điểm]
        - Nhận xét chi tiết: [nhận xét]
        - Tóm tắt: [tóm tắt ngắn gọn]
        - Điểm mạnh: [điểm mạnh]
        - Điểm yếu: [điểm yếu]
        - Khuyến nghị: [khuyến nghị cải thiện]
        """

        if language == "en":
            cv_evaluation_prompt = f"""Please analyze this CV based on: education, experience, skills, achievements, and fit for the position.

            Preliminary analysis:
            {local_analysis}

            Job Market Information:
            {search_text}

            Please rate according to the following scale:
            A: Excellent - Outstanding, perfect fit for the position
            B: Good - Meets requirements well, has many strengths
            C: Average - Meets basic requirements
            D: Below average - Lacks many necessary skills/experience
            F: Not suitable - Does not meet position requirements

            Return results in the following structure:
            - Overall evaluation (grade A-E): [grade]
            - Detailed comments: [comments]
            - Summary: [brief summary]
            - Strengths: [strengths]
            - Weaknesses: [weaknesses]
            - Recommendations: [improvement recommendations]
            """

        print(f"Tạo kết quả cuối cùng, độ dài prompt: {len(cv_evaluation_prompt)}")

        # Sử dụng Gemini cho kết quả cuối nếu được cấu hình, nếu không sử dụng LLM local
        if self.use_gemini:
            final_response = await self.call_gemini_api(cv_evaluation_prompt,
                                                        self.system_prompt_vi if language == "vi" else self.system_prompt_en)
        else:
            final_response = await llm_service.generate_response(
                prompt=cv_evaluation_prompt,
                temperature=0.3
            )

        print(f"Quá trình xử lý file hoàn tất, độ dài kết quả: {len(final_response)}")
        return final_response

    async def chat_with_knowledge(
            self,
            question: str,
            chat_history: List[Dict[str, str]] = None,
            cv_content: Optional[str] = None,
            file_path: Optional[str] = None,
            file_type: Optional[str] = None,
            knowledge_content: Optional[str] = None
    ) -> str:
        """Xử lý câu hỏi người dùng kết hợp với tri thức"""
        print(f"Bắt đầu xử lý câu hỏi: {question}")

        # Kiểm tra xem có file hay không
        has_file = file_path is not None and os.path.exists(file_path)
        has_cv = cv_content is not None and len(cv_content.strip()) > 0
        has_knowledge = knowledge_content is not None and len(knowledge_content.strip()) > 0

        # Nếu có file đã tải lên, chuyển hướng đến process_file_content
        if has_file:
            print(f"Phát hiện file được tải lên, chuyển hướng đến process_file_content. File: {file_path}")
            file_content = ""
            # Đọc nội dung file nếu chưa được cung cấp
            if file_type.lower() in ['.jpg', '.jpeg', '.png']:
                file_content = await self.extract_content_from_image(file_path)
            else:
                try:
                    print(f"Đọc nội dung file văn bản: {file_path}")

                    # Thử đọc với nhiều encoding khác nhau
                    encodings = ['utf-8', 'latin1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                                file_content = f.read()
                                print(f"Đọc thành công với encoding {encoding}, độ dài: {len(file_content)}")
                                break
                        except UnicodeDecodeError:
                            print(f"Không thể đọc với encoding {encoding}, thử encoding khác")

                    # Nếu vẫn không đọc được, thử đọc dưới dạng binary
                    if not file_content:
                        print("Thử đọc file dưới dạng binary")
                        with open(file_path, 'rb') as f:
                            binary_content = f.read()

                        # Đối với file PDF
                        if file_path.lower().endswith('.pdf'):
                            try:
                                import PyPDF2
                                reader = PyPDF2.PdfReader(file_path)
                                file_content = ""
                                for page in reader.pages:
                                    file_content += page.extract_text() + "\n"
                                print(f"Đã trích xuất từ PDF, độ dài: {len(file_content)}")
                            except Exception as pdf_error:
                                print(f"Lỗi khi đọc PDF: {pdf_error}")

                        # Đối với file DOC/DOCX
                        elif file_path.lower().endswith(('.doc', '.docx')):
                            try:
                                import textract
                                file_content = textract.process(file_path).decode('utf-8', errors='ignore')
                                print(f"Đã trích xuất từ DOC/DOCX, độ dài: {len(file_content)}")
                            except Exception as doc_error:
                                print(f"Lỗi khi đọc DOC/DOCX: {doc_error}")

                except Exception as e:
                    print(f"Không thể đọc nội dung file: {e}")

            # In ra thông tin về nội dung đã trích xuất
            print(f"Nội dung file ({len(file_content)} ký tự):")
            print("======= TRÍCH XUẤT NỘI DUNG ======")
            print(file_content[:500] + "..." if len(file_content) > 500 else file_content)
            print("======= KẾT THÚC TRÍCH XUẤT ======")

            # Xử lý file thông qua hàm process_file_content
            return await self.process_file_content(
                file_path=file_path,
                file_content=file_content,
                file_type=file_type,
                query=question,
                cv_content=cv_content,
                knowledge_content=knowledge_content
            )

        # Nếu không có file, tiếp tục xử lý câu hỏi bình thường
        language = self.detect_language(question)
        print(f"Ngôn ngữ phát hiện: {language}")

        # Bước 1: Chuẩn bị lịch sử trò chuyện
        chat_history_text = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                chat_history_text += f"{role.capitalize()}: {content}\n"

        # Bước 2: Tìm kiếm thông tin từ web
        search_results_text = ""
        try:
            # Sử dụng lib để tìm kiếm
            search_results_text = await self.search_with_lib(question)
            print(f"Kết quả tìm kiếm web, độ dài: {len(search_results_text)}")
        except Exception as e:
            print(f"Lỗi khi tìm kiếm web: {e}")

        # Bước 3: Nếu có CV, cần phân tích trước khi hỏi
        cv_analysis = ""
        if has_cv and len(cv_content) > 100:
            try:
                analysis_prompt = f"Phân tích ngắn gọn CV sau đây, nêu những điểm nổi bật: {cv_content[:4000]}"
                cv_analysis = await llm_service.generate_response(
                    prompt=analysis_prompt,
                    temperature=0.3
                )
                print(f"Đã phân tích CV, độ dài: {len(cv_analysis)}")
            except Exception as e:
                print(f"Lỗi khi phân tích CV: {e}")

        # Bước 4: Tạo prompt cuối cùng và gửi cho LLM
        if language == "en":
            prompt = f"""Based on the available information, please answer the user's question.

            {"CV Content Summary:" + cv_analysis if cv_analysis else ""}
            {"CV Content:" + cv_content[:2000] if has_cv and not cv_analysis else ""}
            {"Knowledge Base:" + knowledge_content if has_knowledge else ""}
            {"Chat History:" + chat_history_text if chat_history_text else ""}
            {"Web Search Results:" + search_results_text if search_results_text else ""}

            User Question: {question}

            Answer:"""
        else:
            prompt = f"""Dựa trên thông tin có sẵn, hãy trả lời câu hỏi của người dùng.

            {"Tóm tắt CV:" + cv_analysis if cv_analysis else ""}
            {"Nội dung CV:" + cv_content[:2000] if has_cv and not cv_analysis else ""}
            {"Cơ sở kiến thức:" + knowledge_content if has_knowledge else ""}
            {"Lịch sử trò chuyện:" + chat_history_text if chat_history_text else ""}
            {"Kết quả tìm kiếm web:" + search_results_text if search_results_text else ""}

            Câu hỏi: {question}

            Trả lời:"""

        print(f"Tạo prompt cuối cùng, độ dài: {len(prompt)}")

        # Sử dụng LLM local nếu có dữ liệu phức tạp hoặc không dùng Gemini
        should_use_local = has_cv or len(search_results_text) > 200 or cv_analysis

        if should_use_local or not self.use_gemini:
            print("Sử dụng LLM local để trả lời")
            response = await llm_service.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt_vi if language == "vi" else self.system_prompt_en,
                temperature=0.3 if (has_cv or search_results_text or has_knowledge) else 0.7
            )
        else:
            print("Sử dụng Gemini để trả lời")
            response = await self.call_gemini_api(prompt,
                                                  self.system_prompt_vi if language == "vi" else self.system_prompt_en)

        print(f"Hoàn tất xử lý câu hỏi, độ dài câu trả lời: {len(response)}")
        return response

    async def call_gemini_api(self, prompt: str, system_prompt: str) -> str:
        """Gọi API Gemini để xử lý câu hỏi"""

        if not GEMINI_API_KEY:
            return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)

        try:
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

            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url=url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

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
                    error_detail = f"Lỗi API ({response.status_code}): {response.text}"
                    print(error_detail)
                    return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)

        except Exception as e:
            print(f"Lỗi khi gọi Gemini API: {e}")

            return await llm_service.generate_response(prompt=prompt, system_prompt=system_prompt)


# Singleton instance
integrated_llm_service = IntegratedLLMService()