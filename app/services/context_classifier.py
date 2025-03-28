import re
from typing import Dict, Set, List, Optional


class ContextClassifier:

    def __init__(self):
        self.recruitment_keywords = {
            'cv', 'hồ sơ', 'ứng tuyển', 'resume', 'tuyển dụng', 'phỏng vấn', 'interview',
            'việc làm', 'công việc', 'job', 'career', 'nghề nghiệp', 'vị trí', 'position',
            'recruit', 'hiring', 'skill', 'kỹ năng', 'kinh nghiệm', 'experience'
        }

        self.knowledge_keywords = {
            'chính sách', 'policy', 'quy định', 'regulation', 'rule', 'hướng dẫn', 'guide',
            'nội quy', 'tài liệu', 'document', 'thông tin', 'information', 'kiến thức', 'knowledge'
        }

        self.web_search_keywords = {
            'tìm', 'search', 'kiếm', 'find', 'tra cứu', 'lookup', 'web', 'internet',
            'online', 'thị trường', 'market', 'xu hướng', 'trend', 'mới nhất', 'latest',
            'cập nhật', 'update', 'tin tức', 'news', 'báo cáo', 'report'
        }

        self.search_indicators = {
            'tìm giúp tôi', 'tìm kiếm', 'tra cứu', 'tìm hiểu', 'research',
            'find out', 'look up', 'search for', 'trên mạng', 'online'
        }

        self.recency_indicators = {
            'hiện tại', 'gần đây', 'mới nhất', 'xu hướng', 'trending',
            'current', 'recent', 'latest', 'newest', 'now', 'trend'
        }

    def classify_question(self, question: str) -> Dict[str, float]:
        question = question.lower()
        words = set(re.findall(r'\w+', question))

        recruitment_score = self._calculate_score(words, self.recruitment_keywords)
        knowledge_score = self._calculate_score(words, self.knowledge_keywords)
        web_search_score = self._calculate_score(words, self.web_search_keywords)

        has_search_indicator = any(indicator in question for indicator in self.search_indicators)
        has_recency_indicator = any(indicator in question for indicator in self.recency_indicators)

        if has_search_indicator:
            web_search_score += 0.3
        if has_recency_indicator:
            web_search_score += 0.2

        total = recruitment_score + knowledge_score + web_search_score
        if total == 0:
            return {
                "recruitment": 0.1,
                "knowledge": 0.3,
                "web_search": 0.1,
                "needs_external_info": has_search_indicator or has_recency_indicator
            }

        return {
            "recruitment": recruitment_score / total,
            "knowledge": knowledge_score / total,
            "web_search": web_search_score / total,
            "needs_external_info": has_search_indicator or has_recency_indicator or web_search_score > 0.4
        }

    def _calculate_score(self, words: Set[str], keywords: Set[str]) -> float:
        matches = words.intersection(keywords)
        return len(matches) / max(1, len(words)) * 0.5

    def needs_web_search(self, question: str) -> bool:

        classification = self.classify_question(question)
        return classification["needs_external_info"] or classification["web_search"] > 0.3

    def should_use_knowledge_base(self, question: str) -> bool:
        classification = self.classify_question(question)
        return classification["knowledge"] > 0.25


# Singleton instance
context_classifier = ContextClassifier()