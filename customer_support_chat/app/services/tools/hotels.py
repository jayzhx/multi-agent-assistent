from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter # 导入审批适配器
import sqlite3
from typing import Optional, Union, List, Dict
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH
hotels_vectordb = VectorDB(table_name="hotels", collection_name="hotels_collection")

@tool
def search_hotels(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索酒店。"""
    search_results = hotels_vectordb.search(query, limit=limit)

    hotels = []
    for result in search_results:
        payload = result.payload
        hotels.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "checkin_date": payload["checkin_date"],
            "checkout_date": payload["checkout_date"],
            "booked": payload["booked"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return hotels

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_hotel(hotel_id: int, approval_result=None) -> str:
    """根据酒店 ID 进行预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 预订成功。"
    else:
        conn.close()
        return f"未找到 ID 为 {hotel_id} 的酒店。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
    approval_result=None
) -> str:
    """根据酒店 ID 更新入住和退房日期，并标记为已预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 更新时始终将酒店标记为已预订
    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?",
            (checkin_date.strftime('%Y-%m-%d'), hotel_id),
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date.strftime('%Y-%m-%d'), hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 已成功更新并标记为已预订。"
    else:
        conn.close()
        return f"未找到 ID 为 {hotel_id} 的酒店。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_hotel(hotel_id: int, approval_result=None) -> str:
    """根据酒店 ID 取消预订。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到 ID 为 {hotel_id} 的酒店。"
