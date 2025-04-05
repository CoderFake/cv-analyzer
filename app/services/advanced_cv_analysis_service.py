import os
import re
import json
import base64
import asyncio
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import UUID

import httpx
from langdetect import detect, LangDetectException
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from app.core.config import settings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cv-analysis-service")


class AdvancedCVAnalysisService:
    """
    Advanced CV analysis service using Ollama for CV processing and analysis,
    DuckDuckGo for market research, and Gemini for final user-facing responses
    """

    def __init__(self):

        self.client = httpx.AsyncClient(timeout=60.0)

        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

        self.debug_mode = os.environ.get("DEBUG", "false").lower() == "true"

        self.system_prompts = {
            "cv_extraction": """You are an AI assistant specializing in CV analysis. Your task is to extract all important information from CVs, including:
                - Personal information: name, email, phone, address
                - Education: schools, degrees, fields of study, time periods
                - Work experience: companies, positions, time periods, responsibilities, achievements
                - Skills: technical skills, soft skills, languages
                - Certifications and courses
                - Projects participated in
                - Interests and extracurricular activities
            Return in a clear structured format with separate sections.""",

            "market_analysis": """You are a recruitment expert. Your task is to analyze the requirements for the applied position based on market information.
            Identify:
                - Key skill requirements
                - Experience requirements
                - Education requirements
                - Common salary range
                - Popular technologies/skills
                - Current recruitment trends""",

            "cv_evaluation": """You are a professional recruiter. Your task is to evaluate the CV and provide objective feedback on:
                - Suitability for the applied position
                - Candidate's strengths and weaknesses
                - Competitiveness in the market
                - Ability to meet job requirements
                - Suggestions for CV improvement
            Evaluate on an A-E scale, where A is excellent and E is unsatisfactory."""
        }

        try:

            from app.lib.llm_web_search.retrieval import DocumentRetriever
            from app.lib.llm_web_search.utils import Document, MySentenceTransformer
            import torch

            logger.info("Initializing LlamaIndex components")
            self.llamaindex_available = True

            self.embedding_model = MySentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
                model_kwargs={"torch_dtype": torch.float32}
            )

            self.document_retriever = DocumentRetriever(
                device="cpu",
                num_results=5,
                similarity_threshold=0.5,
                chunk_size=500,
                keyword_retriever="bm25"
            )

            self.Document = Document
            logger.info("LlamaIndex components initialized successfully")

        except Exception as e:
            logger.warning(f"LlamaIndex components not available: {e}")
            self.llamaindex_available = False
            self.embedding_model = None
            self.document_retriever = None
            self.Document = None

    def detect_language(self, text: str) -> str:
        """Detect the language of a text (returns 'en' or 'vi')"""
        try:
            lang = detect(text)
            return "en" if lang == "en" else "vi"
        except LangDetectException:

            return "vi"

    async def extract_content_from_image(self, file_path: str) -> str:
        """Extract text from CV images using Ollama Vision API"""
        try:
            logger.info(f"Extracting content from image: {file_path}")

            with open(file_path, "rb") as image_file:
                file_data = image_file.read()
                file_size = len(file_data)
                logger.info(f"File size: {file_size} bytes")
                base64_image = base64.b64encode(file_data).decode("utf-8")

            prompt_message = "Extract all information from this CV into structured text, including: personal information, education, work experience, skills, certifications and other relevant information."

            payload = {
                "model": self.ollama_model,
                "prompt": prompt_message,
                "images": [base64_image],
                "stream": False
            }

            url = f"{self.ollama_base_url}/api/generate"
            logger.info(f"Calling Ollama Vision API at {url}")

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "")
            logger.info(f"Extracted content length: {len(content)} characters")

            if len(content) < 50:
                logger.warning("Extracted content too short, might indicate a failure")

            return content

        except Exception as e:
            logger.error(f"Error extracting content from image: {e}")

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
        """Extract text from document files (PDF, DOCX, TXT, etc.)"""
        try:
            logger.info(f"Extracting text from document: {file_path} (type: {file_type})")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_type.lower() in ['.jpg', '.jpeg', '.png']:

                return await self.extract_content_from_image(file_path)

            elif file_type.lower() == '.pdf':

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


            elif file_type.lower() in ['.doc', '.docx']:

                try:
                    import textract
                    text = textract.process(file_path).decode('utf-8', errors='ignore')
                    logger.info(f"Extracted {len(text)} characters from DOC/DOCX")
                    return text
                except Exception as doc_error:
                    logger.error(f"Error extracting DOC/DOCX: {doc_error}")

            elif file_type.lower() == '.txt':

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    logger.info(f"Extracted {len(text)} characters from TXT")
                    return text

            try:
                from unstructured.partition.auto import partition
                elements = partition(filename=file_path)
                text = "\n\n".join([str(element) for element in elements])
                logger.info(f"Extracted {len(text)} characters using unstructured")
                return text
            except Exception as unstruct_error:
                logger.error(f"Error using unstructured: {unstruct_error}")

            try:
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

    async def analyze_cv_with_ollama(self, cv_content: str, prompt_type: str = "cv_extraction") -> str:
        """Analyze CV content using Ollama and extract structured information"""
        try:
            logger.info(f"Analyzing CV content with Ollama using prompt type: {prompt_type}")

            system_prompt = self.system_prompts.get(prompt_type, self.system_prompts["cv_extraction"])

            url = f"{self.ollama_base_url}/api/generate"

            combined_prompt = f"{system_prompt}\n\n{cv_content}"

            payload = {
                "model": self.ollama_model,
                "prompt": combined_prompt,
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            analysis = result.get("response", "")
            logger.info(f"CV analysis complete, {len(analysis)} characters")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing CV content with Ollama: {e}")
            return f"Error analyzing CV: {str(e)}"

    async def search_job_market(self, position: str, skills: List[str] = None) -> List[Dict[str, str]]:
        """Search job market information using DuckDuckGo"""
        try:
            logger.info(f"Searching job market for position: {position}")
            results = []

            search_queries = [
                f"job requirements for {position} position",
                f"skills needed for {position}",
                f"{position} salary in Vietnam",
                f"{position} recruitment trends 2025",
                f"{position} top skills market demand"
            ]

            if skills and len(skills) > 0:
                top_skills = skills[:3]
                for skill in top_skills:
                    search_queries.append(f"{position} with {skill} skill requirements")

            search_tasks = []
            for query in search_queries:
                search_tasks.append(self._perform_duckduckgo_search(query))

            search_results = await asyncio.gather(*search_tasks)

            for query_results in search_results:
                results.extend(query_results)

            logger.info(f"Found {len(results)} search results for job market")
            return results

        except Exception as e:
            logger.error(f"Error searching job market: {e}")
            return []

    async def _perform_duckduckgo_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Perform DuckDuckGo search and return results"""
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

                    try:

                        content = result.get("body", "")





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
        """Get and process webpage content"""
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

                for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                    element.decompose()

                paragraphs = []
                for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Only include meaningful paragraphs
                        paragraphs.append(text)

                content = "\n\n".join(paragraphs)

                if len(content) > 1500:
                    content = content[:1500] + "..."

                return content

        except Exception as e:
            logger.error(f"Error getting webpage content: {e}")
            return ""

    async def extract_skills_from_analysis(self, cv_analysis: str) -> List[str]:
        """Extract skills list from CV analysis"""
        try:

            skills_section = re.search(r"(?:Skills|Kỹ năng)[:\s]+(.*?)(?=\n\s*\n|\n#|\n##|$)",
                                       cv_analysis, re.IGNORECASE | re.DOTALL)

            if skills_section:
                skills_text = skills_section.group(1)

                skills_list = re.split(r"[,.:;•\n]+", skills_text)

                skills = [skill.strip() for skill in skills_list if skill.strip()]
                return skills

            prompt = f"""
            From the following CV analysis, extract a list of skills (both technical and soft skills).
            Return as a list with one skill per line, no additional information.

            CV Analysis:
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
        """Extract job position from CV content"""
        try:

            position_patterns = [
                r"(?:vị trí|position|job title|applying for)[:\s]*([^\n.,]+)",
                r"(?:career objective|objective)[:\s]*.*?([^\n.,]{3,30})\s*(?:position|role|job)",
                r"(?:desired position|desired role)[:\s]*([^\n.,]+)"
            ]

            for pattern in position_patterns:
                position_match = re.search(pattern, cv_content, re.IGNORECASE)
                if position_match:
                    return position_match.group(1).strip()

            prompt = f"""
            From the following CV, identify the position that the candidate is applying for or their main profession.
            Return only the position/profession name, no additional information.

            CV:
            {cv_content[:2000]}  # Limit CV length for faster processing
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

    async def generate_final_analysis_with_gemini(
            self,
            cv_content: str,
            position: str,
            market_data: List[Dict[str, str]],
            cv_analysis: str,
            skills: List[str],
            chat_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate comprehensive CV analysis using Gemini API for final user-facing response"""
        try:
            logger.info(f"Generating final analysis with Gemini for position: {position}")

            skills_text = "\n".join([f"- {skill}" for skill in skills])

            market_info = ""
            for i, data in enumerate(market_data, 1):
                market_info += f"Market Info {i}:\n"
                market_info += f"Title: {data.get('title', '')}\n"
                market_info += f"Content: {data.get('content', '')[:500]}...\n"
                market_info += f"Source: {data.get('url', '')}\n\n"

            chat_history_text = ""
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    chat_history_text += f"{role.capitalize()}: {content}\n"

            language = self.detect_language(cv_content)

            if language == "en":
                prompt = f"""
                I need a comprehensive analysis of this CV/resume for a {position} position.

                CV ANALYSIS (from Ollama):
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
                Make your response conversational and helpful, as this will be shown directly to the user.
                """
            else:
                prompt = f"""
                Tôi cần một bản phân tích toàn diện về CV này cho vị trí {position}.

                PHÂN TÍCH CV (từ Ollama):
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
                Hãy làm cho câu trả lời mang tính trò chuyện và hữu ích, vì nó sẽ được hiển thị trực tiếp cho người dùng.
                """

            logger.info("Calling Gemini API for final analysis")
            analysis = await self._call_gemini_api(prompt)

            if not analysis or len(analysis) < 100:
                logger.warning(f"Gemini analysis too short or empty: '{analysis}'")

                logger.info("Falling back to Ollama for final analysis")
                analysis = await self._call_ollama_chat(prompt)

            return analysis

        except Exception as e:
            logger.error(f"Error generating final analysis with Gemini: {e}")

            logger.info("Falling back to Ollama due to Gemini error")
            try:
                return await self._call_ollama_chat(prompt)
            except:
                return f"Error generating CV analysis: {str(e)}"

    async def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API to process request"""
        try:

            if not self.gemini_api_key:
                logger.warning("No Gemini API key provided, falling back to Ollama")
                return await self._call_ollama_chat(prompt)

            system_instruction = """
            You are an AI assistant specializing in CV analysis and recruitment. 
            You provide objective, detailed, and helpful evaluations of resumes and CVs.
            Your analysis is comprehensive and includes both strengths and areas for improvement.
            Your responses are conversational and directly helpful to job seekers.
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
                    return "Could not get valid response from Gemini API."
            else:
                error_detail = f"API error ({response.status_code}): {response.text}"
                logger.error(error_detail)
                return await self._call_ollama_chat(prompt)

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return await self._call_ollama_chat(prompt)

    async def _call_ollama_chat(self, prompt: str) -> str:
        """Call Ollama Generate API as fallback when Gemini is unavailable"""
        try:
            system_prompt = """
            You are an AI assistant specializing in CV analysis and recruitment.
            You provide objective, detailed, and helpful evaluations of CVs.
            Your analysis is comprehensive and includes both strengths and areas for improvement.
            Your responses are conversational and directly helpful to job seekers.
            """

            url = f"{self.ollama_base_url}/api/generate"

            combined_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "model": self.ollama_model,
                "prompt": combined_prompt,
                "stream": False
            }

            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            return result.get("response", "")

        except Exception as e:
            logger.error(f"Error calling Ollama Chat API: {e}")
            return f"Error analyzing CV: {str(e)}"

    async def process_cv_file(
            self,
            file_path: str,
            file_type: str,
            position: Optional[str] = None,
            chat_history: List[Dict[str, str]] = None
    ) -> Tuple[str, bool]:
        """Process CV file from extraction to market research and final evaluation

        Args:
            file_path: Path to CV file
            file_type: File type (.pdf, .docx, .jpg, etc.)
            position: Applied position (if known)
            chat_history: Chat history

        Returns:
            Tuple[str, bool]: (CV analysis, Need to ask for position)
        """
        try:
            logger.info(f"Processing CV file: {file_path} (type: {file_type})")

            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return "Error: File does not exist or cannot be accessed.", False

            cv_content = await self.extract_text_from_document(file_path, file_type)
            logger.info(f"Extracted {len(cv_content)} characters from document")

            if len(cv_content) < 100:
                logger.warning("Extracted content too short to analyze effectively")
                return "The file appears to be empty or couldn't be processed properly. Please upload a different file with more content.", False

            needs_position = False
            if not position:

                position = await self.extract_position_from_cv(cv_content)
                logger.info(f"Extracted position from CV: {position}")

                if not position and chat_history:
                    position_from_chat = self._extract_position_from_chat(chat_history)
                    if position_from_chat:
                        position = position_from_chat
                        logger.info(f"Extracted position from chat history: {position}")

            if not position:
                logger.info("Position not found, will ask user")
                needs_position = True
                return "I've analyzed your CV but need to know what position you're applying for to provide the most relevant assessment. Please let me know the specific role you're interested in.", True

            cv_analysis = await self.analyze_cv_with_ollama(cv_content)
            logger.info("Initial CV analysis complete")

            skills = await self.extract_skills_from_analysis(cv_analysis)
            logger.info(f"Extracted {len(skills)} skills from CV")

            market_data = await self.search_job_market(position, skills)
            logger.info(f"Market search returned {len(market_data)} results")

            comprehensive_analysis = await self.generate_final_analysis_with_gemini(
                cv_content=cv_content,
                position=position,
                market_data=market_data,
                cv_analysis=cv_analysis,
                skills=skills,
                chat_history=chat_history
            )
            logger.info("Comprehensive analysis complete")

            return comprehensive_analysis, False

        except Exception as e:
            logger.error(f"Error processing CV file: {e}")
            return f"An error occurred while analyzing your CV: {str(e)}", False

    def _extract_position_from_chat(self, chat_history: List[Dict[str, str]]) -> str:
        """Extract position from chat history"""
        try:
            position_keywords = [
                "vị trí", "position", "job", "ứng tuyển", "apply",
                "nghề nghiệp", "career", "công việc", "chức danh"
            ]

            for msg in reversed(chat_history):
                content = msg.get("content", "").lower()

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
        """Continue CV analysis after position is provided"""
        try:
            logger.info(f"Continuing analysis with position: {position}")

            analysis, _ = await self.process_cv_file(file_path, file_type, position, chat_history)
            return analysis
        except Exception as e:
            logger.error(f"Error continuing analysis: {e}")
            return f"An error occurred while analyzing your CV for the {position} position: {str(e)}"

    async def answer_general_question(
            self,
            question: str,
            chat_history: List[Dict[str, str]] = None,
            cv_content: Optional[str] = None,
            knowledge_content: Optional[str] = None
    ) -> str:
        """Answer general questions not directly related to CV analysis
        Always use Gemini for user-facing responses"""
        try:
            logger.info(f"Answering general question: {question}")

            language = self.detect_language(question)

            chat_history_text = ""
            if chat_history:
                for message in chat_history:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    chat_history_text += f"{role.capitalize()}: {content}\n"

            cv_context = ""
            if cv_content and len(cv_content.strip()) > 0:


                cv_summary_prompt = f"""
                Briefly summarize this CV, focusing on the most important information:

                {cv_content[:3000]}  # Limit length for faster processing
                """

                url = f"{self.ollama_base_url}/api/generate"
                payload = {
                    "model": self.ollama_model,
                    "prompt": cv_summary_prompt,
                    "stream": False
                }

                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                cv_summary = result.get("response", "")
                cv_context = f"CV Summary:\n{cv_summary}\n\n"

            search_results_text = ""
            job_market_terms = ["market", "salary", "recruitment", "thị trường", "lương", "tuyển dụng"]
            if any(term in question.lower() for term in job_market_terms):
                try:

                    search_results = await self._perform_duckduckgo_search(question, max_results=3)

                    for i, result in enumerate(search_results, 1):
                        search_results_text += f"Search Result {i}:\n"
                        search_results_text += f"Title: {result.get('title', '')}\n"
                        search_results_text += f"Content: {result.get('content', '')[:300]}...\n"
                        search_results_text += f"Source: {result.get('url', '')}\n\n"
                except Exception as e:
                    logger.error(f"Error in web search: {e}")

            knowledge_context = ""
            if knowledge_content and len(knowledge_content.strip()) > 0:
                knowledge_context = f"Knowledge Base:\n{knowledge_content}\n\n"

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

            if self.gemini_api_key:
                logger.info("Using Gemini API to answer general question")
                answer = await self._call_gemini_api(prompt)

                if not answer or len(answer.strip()) < 20:
                    logger.warning("Gemini response was empty or too short, using fallback")
                    answer = await self._call_ollama_chat(prompt)
            else:

                logger.warning("No Gemini API key available - user interaction will use Ollama as fallback")
                answer = await self._call_ollama_chat(prompt)

            return answer

        except Exception as e:
            logger.error(f"Error answering general question: {e}")
            return f"I'm sorry, I couldn't answer your question right now. Error: {str(e)}"

    async def close(self):
        """Close connections"""
        await self.client.aclose()