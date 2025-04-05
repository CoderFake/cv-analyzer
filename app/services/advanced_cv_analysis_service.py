import os
import re
import json
import asyncio
import tempfile
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

import httpx
import requests
from langdetect import detect, LangDetectException
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cv-analysis-service")


class AdvancedCVAnalysisService:
    """
    Dịch vụ phân tích CV nâng cao sử dụng Ollama Vision, LlamaIndex và tìm kiếm Web
    để cung cấp phân tích chi tiết về CV và tính phù hợp với thị trường
    """

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

        # System prompts cho các bước phân tích
        self.system_prompts = {
            "cv_extraction": """Bạn là trợ lý AI chuyên về phân tích CV. Nhiệm vụ của bạn là trích xuất tất cả thông tin quan trọng từ CV, bao gồm:
                - Thông tin cá nhân: tên, email, số điện thoại, địa chỉ
                - Học vấn: trường, bằng cấp, chuyên ngành, thời gian học
                - Kinh nghiệm làm việc: công ty, vị trí, thời gian, trách nhiệm, thành tựu
                - Kỹ năng: kỹ năng chuyên môn, kỹ năng mềm, ngôn ngữ
                - Chứng chỉ và các khóa học
                - Các dự án đã tham gia
                - Sở thích và hoạt động ngoại khóa
            Trả về dưới dạng cấu trúc rõ ràng với các mục riêng biệt.""",

            "position_analysis": """Bạn là chuyên gia tuyển dụng. Nhiệm vụ của bạn là phân tích yêu cầu cho vị trí ứng tuyển dựa trên thông tin thị trường.
            Hãy xác định:
                - Yêu cầu kỹ năng chính
                - Yêu cầu kinh nghiệm
                - Yêu cầu bằng cấp
                - Mức lương phổ biến trên thị trường
                - Các công nghệ/kỹ năng đang được ưa chuộng
                - Xu hướng tuyển dụng hiện tại""",

            "cv_evaluation": """Bạn là nhà tuyển dụng chuyên nghiệp. Nhiệm vụ của bạn là đánh giá CV và đưa ra nhận xét khách quan về:
                - Tính phù hợp với vị trí ứng tuyển
                - Điểm mạnh và điểm yếu của ứng viên
                - Mức độ cạnh tranh so với thị trường
                - Khả năng đáp ứng yêu cầu công việc
                - Gợi ý cải thiện CV
            Đánh giá theo thang điểm A-E, trong đó A là xuất sắc và E là không đạt yêu cầu."""
        }

    def detect_language(self, text: str) -> str:
        """Xác định ngôn ngữ của văn bản"""
        try:
            lang = detect(text)
            return "en" if lang == "en" else "vi"
        except LangDetectException:
            return "vi"  # Mặc định tiếng Việt

    async def extract_content_from_image(self, file_path: str) -> str:
        """Sử dụng Ollama Vision API để trích xuất thông tin từ hình ảnh CV"""
        try:
            logger.info(f"Extracting content from image: {file_path}")

            # Đọc tệp ảnh và chuyển thành base64
            with open(file_path, "rb") as image_file:
                file_data = image_file.read()
                file_size = len(file_data)
                logger.info(f"File size: {file_size} bytes")
                base64_image = base64.b64encode(file_data).decode("utf-8")

            # Chuẩn bị prompt để trích xuất CV
            prompt_message = "Hãy trích xuất tất cả thông tin từ CV này thành văn bản có cấu trúc, bao gồm: thông tin cá nhân, học vấn, kinh nghiệm làm việc, kỹ năng, chứng chỉ và các thông tin liên quan khác."

            payload = {
                "model": self.ollama_model,
                "prompt": prompt_message,
                "images": [base64_image],
                "stream": False
            }

            # Gọi API Ollama
            url = f"{self.ollama_base_url}/api/generate"
            logger.info(f"Calling Ollama Vision API at {url}")

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "")
            logger.info(f"Extracted content length: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Error extracting content from image: {e}")

            # Thử phương án dự phòng
            try:
                from PIL import Image
                import pytesseract

                logger.info("Attempting fallback OCR extraction")
                image = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(image)

                if len(extracted_text) > 50:
                    logger.info(f"Fallback OCR successful, extracted {len(extracted_text)} characters")
                    return extracted_text
            except Exception as ocr_error:
                logger.error(f"Fallback OCR failed: {ocr_error}")

            return f"Error extracting content from image: {e}"

    async def extract_text_from_document(self, file_path: str, file_type: str) -> str:
        """Trích xuất văn bản từ tệp tài liệu (PDF, DOCX, TXT, ...)"""
        try:
            logger.info(f"Extracting text from document: {file_path} (type: {file_type})")

            # Kiểm tra tệp có tồn tại không
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Xử lý theo loại tệp
            if file_type.lower() in ['.jpg', '.jpeg', '.png']:
                # Sử dụng Ollama Vision cho hình ảnh
                return await self.extract_content_from_image(file_path)

            elif file_type.lower() == '.pdf':
                # Xử lý PDF
                try:
                    import PyPDF2

                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n\n"

                        logger.info(f"Extracted {len(text)} characters from PDF")
                        return text
                except Exception as pdf_error:
                    logger.error(f"Error extracting PDF: {pdf_error}")
                    # Thử phương pháp khác nếu PyPDF2 thất bại

            elif file_type.lower() in ['.doc', '.docx']:
                # Xử lý Word documents
                try:
                    import textract
                    text = textract.process(file_path).decode('utf-8', errors='ignore')
                    logger.info(f"Extracted {len(text)} characters from DOC/DOCX")
                    return text
                except Exception as doc_error:
                    logger.error(f"Error extracting DOC/DOCX: {doc_error}")

            elif file_type.lower() == '.txt':
                # Xử lý file text đơn giản
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    logger.info(f"Extracted {len(text)} characters from TXT")
                    return text

            # Thử sử dụng unstructured nếu các phương pháp trên thất bại
            try:
                from unstructured.partition.auto import partition
                elements = partition(filename=file_path)
                text = "\n\n".join([str(element) for element in elements])
                logger.info(f"Extracted {len(text)} characters using unstructured")
                return text
            except Exception as unstruct_error:
                logger.error(f"Error using unstructured: {unstruct_error}")

            # Nếu tất cả phương pháp thất bại, thử đọc dưới dạng văn bản thô
            try:
                # Thử các encoding khác nhau
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                            text = file.read()
                            logger.info(f"Read {len(text)} characters with {encoding} encoding")
                            return text
                    except UnicodeDecodeError:
                        continue
            except Exception as raw_error:
                logger.error(f"Error reading raw file: {raw_error}")

            raise Exception("Could not extract text from document using any available method")

        except Exception as e:
            logger.error(f"Error in extract_text_from_document: {e}")
            return f"Error extracting document: {str(e)}"

    async def analyze_cv_content(self, cv_content: str, prompt_type: str = "cv_extraction") -> str:
        """Phân tích nội dung CV và trích xuất thông tin có cấu trúc"""
        try:
            logger.info(f"Analyzing CV content with prompt type: {prompt_type}")

            system_prompt = self.system_prompts.get(prompt_type, self.system_prompts["cv_extraction"])

            # Sử dụng Ollama để phân tích CV
            url = f"{self.ollama_base_url}/api/chat"

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cv_content}
                ],
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            analysis = result.get("message", {}).get("content", "")
            logger.info(f"CV analysis complete, {len(analysis)} characters")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing CV content: {e}")
            return f"Error analyzing CV: {str(e)}"

    async def search_job_market(self, position: str, skills: List[str] = None) -> List[Dict[str, str]]:
        """Tìm kiếm thông tin thị trường việc làm sử dụng DuckDuckGo"""
        try:
            logger.info(f"Searching job market for position: {position}")
            results = []

            # Tạo queries tìm kiếm
            search_queries = [
                f"yêu cầu tuyển dụng vị trí {position}",
                f"kỹ năng cần thiết cho {position}",
                f"mức lương {position} tại Việt Nam",
                f"xu hướng tuyển dụng {position} 2025"
            ]

            if skills and len(skills) > 0:
                # Thêm tìm kiếm với 3 kỹ năng hàng đầu
                top_skills = skills[:3]
                for skill in top_skills:
                    search_queries.append(f"{position} với kỹ năng {skill}")

            # Thực hiện tìm kiếm song song
            search_tasks = []
            for query in search_queries:
                search_tasks.append(self._perform_duckduckgo_search(query))

            search_results = await asyncio.gather(*search_tasks)

            # Gộp kết quả
            for query_results in search_results:
                results.extend(query_results)

            logger.info(f"Found {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"Error searching job market: {e}")
            return []

    async def _perform_duckduckgo_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Thực hiện tìm kiếm DuckDuckGo và trả về kết quả"""
        try:
            logger.info(f"Performing DuckDuckGo search for: {query}")
            results = []

            with DDGS() as ddgs:
                for result in ddgs.text(
                        query,
                        region='vn-vi',
                        safesearch='moderate',
                        max_results=max_results
                ):
                    # Trích xuất nội dung từ URL
                    try:
                        content = result.get("body", "")
                        # Nếu muốn lấy thêm nội dung từ trang web
                        # extended_content = await self._get_webpage_content(result.get("href", ""))
                        # if extended_content:
                        #     content = extended_content

                        results.append({
                            "title": result.get("title", ""),
                            "content": content,
                            "url": result.get("href", "")
                        })
                    except Exception as ex:
                        logger.error(f"Error processing search result: {ex}")

            return results

        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return []

    async def _get_webpage_content(self, url: str) -> str:
        """Lấy và xử lý nội dung trang web"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
            }

            async with httpx.AsyncClient(headers=headers, timeout=5.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                # Loại bỏ script, style, header, footer...
                for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                    element.decompose()

                # Lấy nội dung văn bản
                paragraphs = []
                for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Chỉ lấy đoạn có nghĩa
                        paragraphs.append(text)

                content = "\n\n".join(paragraphs)
                if len(content) > 1500:
                    content = content[:1500] + "..."

                return content

        except Exception as e:
            logger.error(f"Error getting webpage content: {e}")
            return ""

    async def extract_skills_from_analysis(self, cv_analysis: str) -> List[str]:
        """Trích xuất danh sách kỹ năng từ phân tích CV"""
        try:
            # Tìm kiếm phần kỹ năng trong phân tích CV
            skills_section = re.search(r"(?:Kỹ năng|Skills)[:\s]+(.*?)(?=\n\s*\n|$)", cv_analysis,
                                       re.IGNORECASE | re.DOTALL)
            if skills_section:
                skills_text = skills_section.group(1)
                # Tách các kỹ năng theo dấu phẩy, dấu chấm, hoặc xuống dòng
                skills_list = re.split(r"[,.:;•\n]+", skills_text)
                # Làm sạch và lọc
                skills = [skill.strip() for skill in skills_list if skill.strip()]
                return skills

            # Nếu không tìm thấy phần kỹ năng rõ ràng, sử dụng Ollama để trích xuất
            prompt = f"""
            Từ phân tích CV sau, hãy trích xuất danh sách các kỹ năng (cả kỹ năng kỹ thuật và kỹ năng mềm).
            Trả về dưới dạng danh sách với mỗi kỹ năng trên một dòng, không có thêm thông tin khác.

            CV:
            {cv_analysis}
            """

            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            skills_text = result.get("response", "")
            skills_list = skills_text.strip().split("\n")
            skills = [skill.strip() for skill in skills_list if skill.strip()]

            return skills

        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []

    async def extract_position_from_cv(self, cv_content: str) -> str:
        """Trích xuất vị trí ứng tuyển từ nội dung CV"""
        try:
            # Tìm kiếm vị trí trong CV
            position_match = re.search(r"(?:vị trí|position|job title)[:\s]*([^\n.,]+)", cv_content, re.IGNORECASE)
            if position_match:
                return position_match.group(1).strip()

            # Nếu không tìm thấy, sử dụng Ollama để trích xuất
            prompt = f"""
            Từ CV sau, hãy xác định vị trí mà ứng viên đang ứng tuyển hoặc nghề nghiệp chính.
            Chỉ trả về tên vị trí/nghề nghiệp, không thêm thông tin khác.

            CV:
            {cv_content[:2000]}  # Giới hạn độ dài CV để xử lý nhanh hơn
            """

            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            position = result.get("response", "").strip()
            return position

        except Exception as e:
            logger.error(f"Error extracting position: {e}")
            return ""

    async def generate_comprehensive_analysis(
            self,
            cv_content: str,
            position: str,
            market_data: List[Dict[str, str]],
            chat_history: List[Dict[str, str]] = None
    ) -> str:
        """Tạo phân tích toàn diện về CV sử dụng Gemini API"""
        try:
            logger.info(f"Generating comprehensive analysis for position: {position}")

            # Phân tích CV
            cv_analysis = await self.analyze_cv_content(cv_content)

            # Trích xuất kỹ năng
            skills = await self.extract_skills_from_analysis(cv_analysis)
            skills_text = "\n".join([f"- {skill}" for skill in skills])

            # Chuẩn bị dữ liệu thị trường
            market_info = ""
            for i, data in enumerate(market_data, 1):
                market_info += f"Thông tin thị trường {i}:\n"
                market_info += f"Tiêu đề: {data.get('title', '')}\n"
                market_info += f"Nội dung: {data.get('content', '')[:500]}...\n"
                market_info += f"Nguồn: {data.get('url', '')}\n\n"

            # Chuẩn bị lịch sử chat
            chat_history_text = ""
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    chat_history_text += f"{role.capitalize()}: {content}\n"

            # Phát hiện ngôn ngữ
            language = self.detect_language(cv_content)

            # Chuẩn bị prompt cho Gemini
            if language == "en":
                prompt = f"""
                I need a comprehensive analysis of this CV/resume for a {position} position.

                CV ANALYSIS:
                {cv_analysis}

                SKILLS IDENTIFIED:
                {skills_text}

                JOB MARKET INFORMATION:
                {market_info}

                Please provide a structured analysis covering:
                1. Overall evaluation (grade A-E, where A is excellent)
                2. Strengths and weaknesses 
                3. Fit for the {position} position
                4. Market competitiveness
                5. Suggestions for improvement

                Base your analysis on the CV content and market research provided.
                """
            else:
                prompt = f"""
                Tôi cần một bản phân tích toàn diện về CV này cho vị trí {position}.

                PHÂN TÍCH CV:
                {cv_analysis}

                KỸ NĂNG XÁC ĐỊNH:
                {skills_text}

                THÔNG TIN THỊ TRƯỜNG VIỆC LÀM:
                {market_info}

                Vui lòng cung cấp phân tích theo cấu trúc sau:
                1. Đánh giá tổng thể (thang điểm A-E, trong đó A là xuất sắc)
                2. Điểm mạnh và điểm yếu
                3. Mức độ phù hợp với vị trí {position}
                4. Khả năng cạnh tranh trên thị trường
                5. Đề xuất cải thiện

                Dựa phân tích của bạn trên nội dung CV và nghiên cứu thị trường được cung cấp.
                """

            # Gọi Gemini API
            if self.gemini_api_key:
                logger.info("Using Gemini API for comprehensive analysis")
                analysis = await self._call_gemini_api(prompt)
            else:
                logger.info("Using Ollama for comprehensive analysis")
                analysis = await self._call_ollama_chat(prompt)

            return analysis

        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return f"Error generating analysis: {str(e)}"

    async def _call_gemini_api(self, prompt: str) -> str:
        """Gọi Gemini API để xử lý yêu cầu"""
        try:
            system_instruction = """
            You are an AI assistant specializing in CV analysis and recruitment. 
            You provide objective, detailed, and helpful evaluations of resumes and CVs.
            Your analysis is comprehensive and includes both strengths and areas for improvement.
            """

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
                        {"text": system_instruction}
                    ]
                },
                "generationConfig": {
                    "temperature": 0.4,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 4096
                }
            }

            url = f"{self.gemini_api_url}?key={self.gemini_api_key}"

            response = await self.client.post(
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
                logger.error(error_detail)
                return await self._call_ollama_chat(prompt)

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return await self._call_ollama_chat(prompt)

    async def _call_ollama_chat(self, prompt: str) -> str:
        """Gọi Ollama API để xử lý yêu cầu khi Gemini không khả dụng"""
        try:
            system_prompt = """
            Bạn là trợ lý AI chuyên về phân tích CV và tuyển dụng.
            Bạn cung cấp đánh giá khách quan, chi tiết và hữu ích về CV của ứng viên.
            Phân tích của bạn toàn diện và bao gồm cả điểm mạnh và các lĩnh vực cần cải thiện.
            """

            url = f"{self.ollama_base_url}/api/chat"

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            return result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Error calling Ollama Chat API: {e}")
            return f"Lỗi phân tích CV: {str(e)}"

    async def process_cv_file(
            self,
            file_path: str,
            file_type: str,
            position: Optional[str] = None,
            chat_history: List[Dict[str, str]] = None
    ) -> Tuple[str, bool]:
        """Xử lý tệp CV toàn diện từ trích xuất đến phân tích thị trường và đưa ra đánh giá

        Args:
            file_path: Đường dẫn đến tệp CV
            file_type: Loại tệp (.pdf, .docx, .jpg, ...)
            position: Vị trí ứng tuyển (nếu biết)
            chat_history: Lịch sử trò chuyện

        Returns:
            Tuple[str, bool]: (Phân tích CV, Cần hỏi vị trí)
        """
        try:
            logger.info(f"Processing CV file: {file_path} (type: {file_type})")

            # Kiểm tra file có tồn tại không
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return "Lỗi: File không tồn tại hoặc không thể truy cập.", False

            # 1. Trích xuất nội dung từ file
            cv_content = await self.extract_text_from_document(file_path, file_type)
            logger.info(f"Extracted {len(cv_content)} characters from document")

            # Log một phần nội dung để debug
            logger.debug(f"CV content excerpt: {cv_content[:500]}...")

            # 2. Kiểm tra vị trí ứng tuyển
            needs_position = False
            if not position:
                # Thử trích xuất vị trí từ CV
                position = await self.extract_position_from_cv(cv_content)
                logger.info(f"Extracted position from CV: {position}")

                # Kiểm tra trong lịch sử trò chuyện
                if not position and chat_history:
                    position_from_chat = self._extract_position_from_chat(chat_history)
                    if position_from_chat:
                        position = position_from_chat
                        logger.info(f"Extracted position from chat history: {position}")

            # Nếu vẫn không có vị trí, cần hỏi người dùng
            if not position:
                logger.info("Position not found, will ask user")
                needs_position = True
                # Trả về phản hồi tạm thời và cờ báo cần hỏi vị trí
                return "Tôi đã phân tích CV của bạn, nhưng cần biết bạn đang ứng tuyển vị trí gì để đưa ra đánh giá phù hợp nhất. Vui lòng cho tôi biết vị trí bạn đang ứng tuyển.", True

            # 3. Phân tích sơ bộ CV
            cv_analysis = await self.analyze_cv_content(cv_content)
            logger.info("Initial CV analysis complete")

            # 4. Trích xuất kỹ năng
            skills = await self.extract_skills_from_analysis(cv_analysis)
            logger.info(f"Extracted {len(skills)} skills from CV")

            # 5. Tìm kiếm thông tin thị trường
            market_data = await self.search_job_market(position, skills)
            logger.info(f"Market search returned {len(market_data)} results")

            # 6. Tổng hợp và phân tích toàn diện
            comprehensive_analysis = await self.generate_comprehensive_analysis(
                cv_content=cv_content,
                position=position,
                market_data=market_data,
                chat_history=chat_history
            )
            logger.info("Comprehensive analysis complete")

            return comprehensive_analysis, False

        except Exception as e:
            logger.error(f"Error processing CV file: {e}")
            return f"Đã xảy ra lỗi khi xử lý CV: {str(e)}", False

    def _extract_position_from_chat(self, chat_history: List[Dict[str, str]]) -> str:
        """Trích xuất vị trí ứng tuyển từ lịch sử trò chuyện"""
        try:
            position_keywords = [
                "vị trí", "position", "job", "ứng tuyển", "apply",
                "nghề nghiệp", "career", "công việc", "chức danh"
            ]

            # Tìm trong tin nhắn gần đây nhất trước tiên
            for msg in reversed(chat_history):
                content = msg.get("content", "").lower()

                # Tìm cấu trúc "vị trí X" hoặc "apply for X position"
                for keyword in position_keywords:
                    pattern = rf"{keyword}[:\s]+([^\n.,?!]+)"
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        position = match.group(1).strip()
                        return position

            return ""
        except Exception as e:
            logger.error(f"Error extracting position from chat: {e}")
            return ""

    async def continue_analysis_with_position(
            self,
            file_path: str,
            file_type: str,
            position: str,
            chat_history: List[Dict[str, str]] = None
    ) -> str:
        """Tiếp tục phân tích CV sau khi đã có vị trí ứng tuyển"""
        try:
            logger.info(f"Continuing analysis with position: {position}")

            # Gọi lại hàm process_cv_file với vị trí đã biết
            analysis, _ = await self.process_cv_file(file_path, file_type, position, chat_history)
            return analysis
        except Exception as e:
            logger.error(f"Error continuing analysis: {e}")
            return f"Đã xảy ra lỗi khi phân tích CV với vị trí {position}: {str(e)}"

    async def close(self):
        """Đóng các kết nối"""
        await self.client.aclose()

    async def answer_general_question(
            self,
            question: str,
            chat_history: List[Dict[str, str]] = None,
            cv_content: Optional[str] = None,
            knowledge_content: Optional[str] = None
    ) -> str:
        """Trả lời câu hỏi chung, không liên quan trực tiếp đến phân tích CV

        Args:
            question: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện
            cv_content: Nội dung CV (nếu có)
            knowledge_content: Nội dung từ knowledge base (nếu có)

        Returns:
            str: Câu trả lời
        """
        try:
            logger.info(f"Answering general question: {question}")

            # Phát hiện ngôn ngữ
            language = self.detect_language(question)

            # Chuẩn bị lịch sử chat
            chat_history_text = ""
            if chat_history:
                for message in chat_history:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    chat_history_text += f"{role.capitalize()}: {content}\n"

            # Chuẩn bị context liên quan (nếu cần)
            cv_context = ""
            if cv_content and len(cv_content.strip()) > 0:
                # Nếu có CV, chúng ta có thể tóm tắt nó
                cv_analysis_prompt = f"""
                Hãy tóm tắt ngắn gọn CV sau đây, tập trung vào thông tin quan trọng nhất:

                {cv_content[:3000]}  # Giới hạn độ dài để xử lý nhanh
                """

                cv_summary = await self._call_ollama_chat(cv_analysis_prompt)
                cv_context = f"Tóm tắt CV:\n{cv_summary}\n\n"

            # Tìm kiếm thông tin từ web nếu câu hỏi liên quan đến thị trường việc làm
            search_results_text = ""
            if "thị trường" in question.lower() or "lương" in question.lower() or "tuyển dụng" in question.lower():
                try:
                    # Thực hiện tìm kiếm
                    search_results = await self._perform_duckduckgo_search(question, max_results=3)

                    # Định dạng kết quả
                    for i, result in enumerate(search_results, 1):
                        search_results_text += f"Kết quả tìm kiếm {i}:\n"
                        search_results_text += f"Tiêu đề: {result.get('title', '')}\n"
                        search_results_text += f"Nội dung: {result.get('content', '')[:300]}...\n"
                        search_results_text += f"Nguồn: {result.get('url', '')}\n\n"
                except Exception as e:
                    logger.error(f"Error in web search: {e}")

            # Chuẩn bị nội dung knowledge base
            knowledge_context = ""
            if knowledge_content and len(knowledge_content.strip()) > 0:
                knowledge_context = f"Knowledge Base:\n{knowledge_content}\n\n"

            # Tạo prompt dựa trên ngôn ngữ
            if language == "en":
                prompt = f"""
                Please answer the following question professionally and helpfully.

                {cv_context}
                {knowledge_context}
                {search_results_text}

                Previous conversation:
                {chat_history_text}

                Question: {question}

                Answer:
                """
            else:
                prompt = f"""
                Vui lòng trả lời câu hỏi sau một cách chuyên nghiệp và hữu ích.

                {cv_context}
                {knowledge_context}
                {search_results_text}

                Cuộc trò chuyện trước đó:
                {chat_history_text}

                Câu hỏi: {question}

                Trả lời:
                """

            # Sử dụng Gemini nếu có API key, nếu không sử dụng Ollama
            if self.gemini_api_key:
                logger.info("Using Gemini API to answer general question")
                answer = await self._call_gemini_api(prompt)
            else:
                logger.info("Using Ollama to answer general question")
                answer = await self._call_ollama_chat(prompt)

            return answer

        except Exception as e:
            logger.error(f"Error answering general question: {e}")
            return f"Xin lỗi, tôi không thể trả lời câu hỏi của bạn lúc này. Lỗi: {str(e)}"