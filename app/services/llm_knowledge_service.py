import asyncio
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.knowledge_repository import KnowledgeRepository
from app.services.llm_service import llm_service


class LLMKnowledgeService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.knowledge_repo = KnowledgeRepository(db)

    async def get_relevant_knowledge(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        search_task = asyncio.create_task(self.knowledge_repo.search_knowledge(query))

        if hasattr(self, 'current_category') and self.current_category:
            category_task = asyncio.create_task(self.knowledge_repo.get_by_category(self.current_category))
            search_docs, category_docs = await asyncio.gather(search_task, category_task)
        else:
            search_docs = await search_task
            category_docs = []

        unique_docs = {}
        for doc in search_docs + category_docs:
            if str(doc.id) not in unique_docs:
                unique_docs[str(doc.id)] = doc

        knowledge_docs = list(unique_docs.values())[:max_results]

        results = []
        for doc in knowledge_docs:
            results.append({
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content,
                "description": doc.description,
                "category": doc.category,
                "created_at": doc.created_at.isoformat()
            })

        return results

    async def answer_with_knowledge(
            self,
            question: str,
            category: Optional[str] = None,
            use_knowledge: bool = True
    ) -> Dict[str, Any]:
        language = llm_service.detect_language(question)

        if not use_knowledge:
            response = await llm_service.generate_response(
                prompt=question,
                temperature=0.7
            )
            return {
                "answer": response,
                "sources": []
            }

        knowledge_docs = []
        if category:
            category_docs = await self.knowledge_repo.get_by_category(category)
            knowledge_docs.extend(category_docs)

        search_docs = await self.knowledge_repo.search_knowledge(question)
        knowledge_docs.extend(search_docs)

        unique_ids = set()
        unique_docs = []
        for doc in knowledge_docs:
            if doc.id not in unique_ids:
                unique_ids.add(doc.id)
                unique_docs.append(doc)

        knowledge_docs = unique_docs[:5]

        context = ""
        sources = []
        for i, doc in enumerate(knowledge_docs, 1):
            if language == "en":
                context += f"\n[Document {i}] {doc.title}\n{doc.content}\n"
            else:
                context += f"\n[Tài liệu {i}] {doc.title}\n{doc.content}\n"
            sources.append({
                "id": str(doc.id),
                "title": doc.title,
                "category": doc.category
            })

        if language == "en":
            prompt = f"""Please answer the question based on the information in the following documents. 
            If the information in the documents is not sufficient to answer, please indicate that you don't have enough information and answer based on your knowledge.

            Documents:
            {context}

            Question: {question}

            Answer:"""
        else:
            prompt = f"""Hãy trả lời câu hỏi dựa trên thông tin trong các tài liệu sau. 
            Nếu thông tin trong tài liệu không đủ để trả lời, hãy cho biết bạn không có đủ thông tin và trả lời dựa trên kiến thức của bạn.

            Tài liệu:
            {context}

            Câu hỏi: {question}

            Trả lời:"""

        response = await llm_service.generate_response(
            prompt=prompt,
            temperature=0.3
        )

        return {
            "answer": response,
            "sources": sources
        }

    async def generate_knowledge_summary(self, category: Optional[str] = None) -> str:
        if category:
            knowledge_docs = await self.knowledge_repo.get_by_category(category)
        else:
            knowledge_docs = await self.knowledge_repo.get_multi(is_active=True, limit=20)

        docs_info = ""
        for i, doc in enumerate(knowledge_docs, 1):
            docs_info += f"\n[Document {i}] {doc.title}: {doc.description or 'Không có mô tả'}\n"
            if doc.content:
                preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                docs_info += f"Preview: {preview}\n"

        prompt = f"""Hãy tạo một tóm tắt về các tài liệu sau đây trong knowledge base:
        {docs_info}
        
        Tóm tắt nên bao gồm:
        1. Tổng quan về số lượng và thể loại tài liệu
        2. Các chủ đề chính được đề cập
        3. Thông tin quan trọng nhất từ các tài liệu
        
        Tóm tắt:"""

        response = await llm_service.generate_response(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024
        )

        return response
