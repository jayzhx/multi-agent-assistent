from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter # 导入审批适配器
import sqlite3
from typing import Optional, List, Dict

settings = get_settings()
db = settings.SQLITE_DB_PATH
excursions_vectordb = VectorDB(table_name="trip_recommendations", collection_name="excursions_collection")

@tool
def search_trip_recommendations(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索行程推荐。"""
    search_results = excursions_vectordb.search(query, limit=limit)

    recommendations = []
    for result in search_results:
        payload = result.payload
        recommendations.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "keywords": payload["keywords"],
            "details": payload["details"],
            "booked": payload["booked"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return recommendations

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_excursion(recommendation_id: int, approval_result=None) -> str:
    """根据出游项目 ID 进行预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"出游项目 {recommendation_id} 预订成功。"
    else:
        conn.close()
        return f"未找到 ID 为 {recommendation_id} 的出游项目。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_excursion(recommendation_id: int, details: str, approval_result=None) -> str:
    """根据出游项目 ID 更新详情。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"出游项目 {recommendation_id} 更新成功。"
    else:
        conn.close()
        return f"未找到 ID 为 {recommendation_id} 的出游项目。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_excursion(recommendation_id: int, approval_result=None) -> str:
    """根据出游项目 ID 取消预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"出游项目 {recommendation_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到 ID 为 {recommendation_id} 的出游项目。"
