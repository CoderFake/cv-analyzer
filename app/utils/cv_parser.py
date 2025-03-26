import os
import re
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title


class CVParser:

    @staticmethod
    async def extract_text_from_file(file_path: str) -> str:
        try:
            elements = partition(
                filename=file_path,
                include_metadata=True,
                strategy="hi_res" if file_path.lower().endswith(('.jpg', '.jpeg', '.png')) else "auto"
            )

            text = "\n\n".join([str(element) for element in elements])
            return text

        except Exception as e:
            print(f"Error extracting text from file {file_path}: {e}")
            return ""

    @staticmethod
    async def parse_cv_data(cv_text: str) -> Dict[str, Any]:
        elements = []
        for line in cv_text.split('\n'):
            if line.strip():
                elements.append(line)

        chunks = chunk_by_title(elements)

        cv_data = {
            "personal_info": {},
            "education": [],
            "work_experience": [],
            "skills": [],
            "languages": [],
            "certifications": [],
            "projects": [],
            "achievements": []
        }

        cv_data["personal_info"] = CVParser._parse_personal_info(cv_text)
        cv_data["education"] = CVParser._parse_education(cv_text)
        cv_data["work_experience"] = CVParser._parse_work_experience(cv_text)
        cv_data["skills"] = CVParser._parse_skills(cv_text)
        cv_data["languages"] = CVParser._parse_languages(cv_text)
        cv_data["certifications"] = CVParser._parse_certifications(cv_text)

        return cv_data

    @staticmethod
    def _parse_personal_info(text: str) -> Dict[str, Any]:
        info = {}
        patterns = {
            "name": r"(?:^|\n)(?:name|họ tên|full name)[:\s]+([A-Za-zÀ-ỹ\s]+)",
            "email": r"(?:^|\n)(?:email|e-mail)[:\s]+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
            "phone": r"(?:^|\n)(?:phone|điện thoại|tel|mobile|số điện thoại)[:\s]+((?:\+?\d{1,3}[-\.\s]?)?\d{3,}[-\.\s]?\d{3,}[-\.\s]?\d{3,})",
            "address": r"(?:^|\n)(?:address|địa chỉ)[:\s]+([^\n]+)",
            "dob": r"(?:^|\n)(?:dob|date of birth|ngày sinh|birthday)[:\s]+([^\n]+)",
            "linkedin": r"(?:linkedin\.com\/in\/[a-zA-Z0-9_-]+)",
            "github": r"(?:github\.com\/[a-zA-Z0-9_-]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if key in ["email", "linkedin", "github"]:
                    info[key] = match.group(1) if key == "email" else match.group(0)
                else:
                    info[key] = match.group(1).strip()

        return info

    @staticmethod
    def _parse_education(text: str) -> List[Dict[str, Any]]:
        education = []

        education_section = CVParser._extract_section(
            text,
            ["education", "học vấn", "academic background", "quá trình học tập"]
        )

        if education_section:
            edu_entries = re.split(r"\n(?=\d{4}|\d{2}\/\d{4}|[A-Za-z]+\s\d{4})", education_section)

            for entry in edu_entries:
                if not entry.strip():
                    continue

                edu_info = {}
                time_match = re.search(r"(\d{2}\/\d{4}|\d{4})\s*[-–—]\s*(\d{2}\/\d{4}|\d{4}|present|hiện tại)", entry,
                                       re.IGNORECASE)
                if time_match:
                    edu_info["time_period"] = time_match.group(0).strip()

                school_match = re.search(r"(?:university|trường|college|academy|học viện)[:\s]+([^\n,]+)", entry,
                                         re.IGNORECASE)
                if school_match:
                    edu_info["school"] = school_match.group(1).strip()
                else:
                    lines = entry.split('\n')
                    for line in lines:
                        if "university" in line.lower() or "college" in line.lower() or "trường" in line.lower():
                            edu_info["school"] = line.strip()
                            break

                major_match = re.search(r"(?:major|ngành|chuyên ngành|field)[:\s]+([^\n,]+)", entry, re.IGNORECASE)
                if major_match:
                    edu_info["major"] = major_match.group(1).strip()

                degree_match = re.search(r"(?:degree|bằng|bachelor|master|tiến sĩ|doctor|phd)[:\s]+([^\n,]+)", entry,
                                         re.IGNORECASE)
                if degree_match:
                    edu_info["degree"] = degree_match.group(1).strip()

                edu_info["raw_content"] = entry.strip()

                if edu_info:
                    education.append(edu_info)

        return education

    @staticmethod
    def _parse_work_experience(text: str) -> List[Dict[str, Any]]:
        experiences = []

        experience_section = CVParser._extract_section(
            text,
            ["work experience", "experience", "employment history", "kinh nghiệm làm việc", "kinh nghiệm"]
        )

        if experience_section:
            exp_entries = re.split(r"\n(?=\d{4}|\d{2}\/\d{4}|[A-Za-z]+\s\d{4})", experience_section)

            for entry in exp_entries:
                if not entry.strip():
                    continue

                exp_info = {}
                time_match = re.search(r"(\d{2}\/\d{4}|\d{4})\s*[-–—]\s*(\d{2}\/\d{4}|\d{4}|present|hiện tại)", entry,
                                       re.IGNORECASE)
                if time_match:
                    exp_info["time_period"] = time_match.group(0).strip()

                company_match = re.search(r"(?:company|công ty|organization|tổ chức)[:\s]+([^\n,]+)", entry,
                                          re.IGNORECASE)
                if company_match:
                    exp_info["company"] = company_match.group(1).strip()
                else:
                    lines = entry.split('\n')
                    for line in lines:
                        if "company" in line.lower() or "công ty" in line.lower() or "corporation" in line.lower():
                            exp_info["company"] = line.strip()
                            break

                position_match = re.search(r"(?:position|title|role|vị trí|chức danh)[:\s]+([^\n,]+)", entry,
                                           re.IGNORECASE)
                if position_match:
                    exp_info["position"] = position_match.group(1).strip()
                else:
                    lines = entry.split('\n')
                    if lines and not exp_info.get("company"):
                        exp_info["position"] = lines[0].strip()

                description_lines = []
                found_desc = False
                for line in entry.split('\n'):
                    if re.search(r"(?:description|responsibilities|achievements|mô tả|nhiệm vụ|thành tựu)", line,
                                 re.IGNORECASE):
                        found_desc = True
                        continue
                    if found_desc and line.strip():
                        description_lines.append(line.strip())

                if description_lines:
                    exp_info["description"] = "\n".join(description_lines)

                exp_info["raw_content"] = entry.strip()

                if exp_info:
                    experiences.append(exp_info)

        return experiences

    @staticmethod
    def _parse_skills(text: str) -> List[str]:
        skills = []

        skills_section = CVParser._extract_section(
            text,
            ["skills", "kỹ năng", "technical skills", "competencies"]
        )

        if skills_section:
            skill_entries = re.findall(r"(?:^|\n)(?:[-•*]\s*|\d+\.\s*)([^-•*\n]+)", skills_section)

            for entry in skill_entries:
                skill = entry.strip()
                if skill:
                    skills.append(skill)

            if not skills:
                skills_lines = skills_section.split('\n')
                for line in skills_lines:
                    if ':' in line:
                        skills_part = line.split(':', 1)[1].strip()
                        skill_items = [s.strip() for s in re.split(r',|;', skills_part)]
                        skills.extend([s for s in skill_items if s])
                    elif line.strip() and not any(keyword in line.lower() for keyword in ["skills", "kỹ năng"]):
                        skills.append(line.strip())

        return skills

    @staticmethod
    def _parse_languages(text: str) -> List[Dict[str, Any]]:
        languages = []

        languages_section = CVParser._extract_section(
            text,
            ["languages", "ngôn ngữ", "language skills"]
        )

        if languages_section:
            language_entries = re.findall(r"(?:^|\n)(?:[-•*]\s*|\d+\.\s*)?([A-Za-zÀ-ỹ\s]+)(?:\s*[:-]\s*([^-•*\n]+))?",
                                          languages_section)

            for entry in language_entries:
                language = entry[0].strip()
                proficiency = entry[1].strip() if len(entry) > 1 and entry[1].strip() else None

                if language and language.lower() not in ["languages", "ngôn ngữ", "language skills"]:
                    language_info = {"language": language}
                    if proficiency:
                        language_info["proficiency"] = proficiency
                    languages.append(language_info)

        return languages

    @staticmethod
    def _parse_certifications(text: str) -> List[Dict[str, Any]]:
        certifications = []

        certifications_section = CVParser._extract_section(
            text,
            ["certifications", "certificates", "chứng chỉ", "qualifications"]
        )

        if certifications_section:
            cert_entries = re.findall(r"(?:^|\n)(?:[-•*]\s*|\d+\.\s*)([^-•*\n]+)(?:\s*[:-]\s*([^-•*\n]+))?",
                                      certifications_section)

            for entry in cert_entries:
                cert_name = entry[0].strip()
                detail = entry[1].strip() if len(entry) > 1 and entry[1].strip() else None

                if cert_name and cert_name.lower() not in ["certifications", "certificates", "chứng chỉ"]:
                    cert_info = {"name": cert_name}
                    if detail:
                        cert_info["detail"] = detail

                        date_match = re.search(r"(\d{2}\/\d{4}|\d{4})", detail)
                        if date_match:
                            cert_info["date"] = date_match.group(0)

                    certifications.append(cert_info)

        return certifications

    @staticmethod
    def _extract_section(text: str, section_keywords: List[str]) -> Optional[str]:
        patterns = [
            fr"(?i)(?:^|\n)(?:{keyword})(?::|\.|\n)(.*?)(?:\n(?:{keyword})|$)"
            for keyword in section_keywords
        ]

        common_sections = [
            "personal information", "contact", "objective", "summary", "education",
            "experience", "work experience", "employment", "skills", "languages",
            "certifications", "projects", "achievements", "references",
            "thông tin cá nhân", "liên hệ", "mục tiêu", "tóm tắt", "học vấn",
            "kinh nghiệm", "kỹ năng", "ngôn ngữ", "chứng chỉ", "dự án", "thành tựu"
        ]
        common_section_pattern = "|".join(common_sections)

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()

                next_section_match = re.search(fr"(?i)(?:\n)({common_section_pattern})(?::|\.|\n)", content)
                if next_section_match:
                    content = content[:next_section_match.start()].strip()

                return content

        return None