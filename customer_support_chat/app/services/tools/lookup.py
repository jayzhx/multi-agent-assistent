from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

settings = get_settings()
faq_vectordb = VectorDB(table_name="faq", collection_name="faq_collection")


def normalize_text(text: str) -> str:
    """保留中文、英文和数字，便于做轻量关键词匹配。"""
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text).lower()


def calculate_keyword_score(query: str, candidate: str) -> float:
    """在向量检索结果上叠加轻量关键词分数，提高 FAQ 精确命中率。"""
    query_text = normalize_text(query)
    candidate_text = normalize_text(candidate)

    if not query_text or not candidate_text:
        return 0.0

    score = 0.0
    if query_text in candidate_text:
        score += 5.0

    query_chars = set(query_text)
    if query_chars:
        score += len(query_chars & set(candidate_text)) / len(query_chars)

    query_bigrams = {
        query_text[index:index + 2]
        for index in range(len(query_text) - 1)
        if len(query_text[index:index + 2]) == 2
    }
    if query_bigrams:
        matched_bigrams = sum(1 for bigram in query_bigrams if bigram in candidate_text)
        score += (matched_bigrams / len(query_bigrams)) * 3

    return score

@tool
def search_faq(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索 FAQ 条目。"""
    candidate_limit = max(limit * 20, 100)
    search_results = faq_vectordb.search(query, limit=candidate_limit)

    faq_entries = []
    for result in search_results:
        payload = result.payload
        content = payload.get("content", "")
        
        # 优先使用写库时保存的问答元数据，无法获取时再回退到文本解析
        question = payload.get("question", "常规 FAQ 信息")
        answer = payload.get("answer", content)
        category = payload.get("category", "常见问题")
        
        # 查找带编号的问题格式（例如 “1. 我可以……”）
        question_match = re.search(r'^\d+\. (.+?)(?=\n|$)', content, re.MULTILINE)
        if not payload.get("question") and question_match:
            question = question_match.group(1).strip()
            # 提取答案（问题之后的全部内容）
            answer_start = content.find(question) + len(question)
            answer = content[answer_start:].strip()
        elif not payload.get("question") and content.startswith('##'):
            # 处理章节标题
            lines = content.split('\n', 1)
            question = lines[0].replace('##', '').strip()
            answer = lines[1] if len(lines) > 1 else "详情请查看该章节内容。"

        candidate_text = "\n".join(part for part in [category, question, answer, content] if part)
        keyword_score = calculate_keyword_score(query, candidate_text)
        
        faq_entries.append({
            "question": question,
            "answer": answer,
            "category": category,
            "chunk": content,
            "similarity": result.score,
            "_keyword_score": keyword_score,
        })

    faq_entries.sort(key=lambda entry: (entry["_keyword_score"], entry["similarity"]), reverse=True)

    ranked_entries = []
    for entry in faq_entries[:limit]:
        entry.pop("_keyword_score", None)
        ranked_entries.append(entry)

    return ranked_entries

@tool
def lookup_policy(query: str) -> str:
    """查询公司政策，确认某些操作是否被允许。
    在改签航班或执行其他写入类操作前，应先调用此工具。"""
    faq_results = search_faq.invoke({"query": query, "limit": 2})
    if not faq_results:
        return "抱歉，我没有找到相关的政策信息。请联系客服获取帮助。"
    
    policy_info = "\n\n".join([f"问：{entry['question']}\n答：{entry['answer']}" for entry in faq_results])
    return f"以下是相关的政策信息：\n\n{policy_info}"
