from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter # 导入审批适配器
import sqlite3
from typing import Optional, Union, List, Dict
from datetime import datetime, date
import pytz

settings = get_settings()
db = settings.SQLITE_DB_PATH
flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")


@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> List[Dict]:
    """获取用户的全部机票信息，以及对应的航班信息和座位分配。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def search_flights(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """根据自然语言查询搜索航班。"""
    search_results = flights_vectordb.search(query, limit=limit)

    flights = []
    for result in search_results:
        payload = result.payload
        flights.append({
            "flight_id": payload["flight_id"],
            "flight_no": payload["flight_no"],
            "departure_airport": payload["departure_airport"],
            "arrival_airport": payload["arrival_airport"],
            "scheduled_departure": payload["scheduled_departure"],
            "scheduled_arrival": payload["scheduled_arrival"],
            "status": payload["status"],
            "aircraft_code": payload["aircraft_code"],
            "actual_departure": payload["actual_departure"],
            "actual_arrival": payload["actual_arrival"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return flights

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig, approval_result=None
) -> str:
    """将用户机票改签到新的有效航班。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。
    
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 检查机票是否存在且属于该乘客
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        # 理论上如果用户意图已确认，这里不应触发，但仍保留这层校验。
        return f"未找到属于乘客 {passenger_id} 的机票 {ticket_no}。"

    # 更新 ticket_flights 表中的航班信息
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    conn.close()
    if cursor.rowcount > 0:
        return f"机票 {ticket_no} 已成功改签到航班 {new_flight_id}。"
    else:
        return f"机票 {ticket_no} 更新失败。"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_ticket(ticket_no: str, *, config: RunnableConfig, approval_result=None) -> str:
    """取消用户机票，并将其从数据库中移除。"""
    # 如果审批被拒绝，将不会执行下面的函数体。
    # 如果审批通过，approval_result 中会包含审批详情。

    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客 ID。")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 检查机票是否存在且属于该乘客
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        # 理论上如果用户意图已确认，这里不应触发，但仍保留这层校验。
        return f"未找到属于乘客 {passenger_id} 的机票 {ticket_no}。"

    # 从 ticket_flights 中删除
    cursor.execute(
        "DELETE FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,),
    )
    # 从 tickets 中删除
    cursor.execute(
        "DELETE FROM tickets WHERE ticket_no = ?",
        (ticket_no,),
    )
    conn.commit()

    conn.close()
    return f"机票 {ticket_no} 已成功取消。"
