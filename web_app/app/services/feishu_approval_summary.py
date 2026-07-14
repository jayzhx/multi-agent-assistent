from datetime import date, datetime, timedelta
from typing import Any, Dict

from customer_support_chat.app.core.database import get_connection
from customer_support_chat.app.services.order_service import (
    get_product,
    resolve_service_dates,
)


TOOL_LABELS = {
    "book_flight": "航班预订",
    "book_hotel": "酒店预订",
    "book_car_rental": "租车预订",
    "book_excursion": "行程预订",
    "update_ticket_to_new_flight": "航班改签",
    "update_hotel": "酒店日期修改",
    "update_car_rental": "租车日期修改",
    "update_excursion": "行程修改",
    "cancel_ticket": "取消航班订单",
    "cancel_hotel": "取消酒店订单",
    "cancel_car_rental": "取消租车订单",
    "cancel_excursion": "取消行程订单",
    "retry_order_booking": "重试供应商下单",
}

ARGUMENT_LABELS = {
    "order_id": "订单ID",
    "new_flight_id": "新航班产品ID",
    "checkin_date": "新入住日期",
    "checkout_date": "新退房日期",
    "start_date": "新开始日期",
    "end_date": "新结束日期",
    "visit_date": "新出行日期",
    "participant_count": "参与人数",
}

ORDER_TYPE_LABELS = {
    "flight": "航班",
    "hotel": "酒店",
    "car": "租车",
    "trip": "行程",
}

STATUS_LABELS = {
    "pending": "待处理",
    "processing": "处理中",
    "confirmed": "已确认",
    "cancelled": "已取消",
    "failed": "失败",
}


def format_amount(amount_minor: int, currency: str) -> str:
    amount = amount_minor / 100
    if currency == "CNY":
        return f"人民币 {amount:.2f} 元"
    return f"{currency} {amount:.2f}"


def format_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "未指定")


def normalize_date(value: Any, fallback: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    return fallback


def join_summary(parts: list[str]) -> str:
    return "｜".join(part for part in parts if part)


def summarize_booking(tool_name: str, args: Dict[str, Any]) -> str:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            if tool_name == "book_flight":
                product_id = int(args["flight_id"])
                product = get_product(cursor, "flight", product_id)
                cursor.execute(
                    """
                    SELECT flight_no, departure_airport, arrival_airport,
                           scheduled_departure, scheduled_arrival
                    FROM flights WHERE flight_id = %s
                    """,
                    (product_id,),
                )
                flight = cursor.fetchone()
                if flight is None:
                    raise ValueError("航班产品不存在。")
                return join_summary([
                    "审批事项：航班预订",
                    f"航班：{flight[0]}",
                    f"航线：{flight[1]} 至 {flight[2]}",
                    f"起飞：{format_datetime(flight[3])}",
                    f"到达：{format_datetime(flight[4])}",
                    "舱位：经济舱",
                    f"预计金额：{format_amount(product['unit_amount_minor'], product['currency'])}",
                    f"航班产品ID：{product_id}",
                ])

            if tool_name == "book_hotel":
                product_id = int(args["hotel_id"])
                product = get_product(cursor, "hotel", product_id)
                cursor.execute(
                    """
                    SELECT h.name, h.location, h.checkin_date, h.checkout_date,
                           COALESCE(hp.room_type, '标准房')
                    FROM hotels h
                    LEFT JOIN hotel_products hp ON hp.product_id = %s
                    WHERE h.id = %s
                    """,
                    (product["id"], product_id),
                )
                hotel = cursor.fetchone()
                if hotel is None:
                    raise ValueError("酒店产品不存在。")
                checkin_date, checkout_date = resolve_service_dates(
                    args.get("checkin_date"),
                    args.get("checkout_date"),
                    hotel[2],
                    hotel[3],
                )
                nights = max((checkout_date - checkin_date).days, 1)
                return join_summary([
                    "审批事项：酒店预订",
                    f"酒店：{hotel[0]}",
                    f"地点：{hotel[1]}",
                    f"房型：{hotel[4]}",
                    f"入住：{checkin_date.isoformat()}",
                    f"退房：{checkout_date.isoformat()}",
                    f"住宿：{nights}晚",
                    f"预计金额：{format_amount(product['unit_amount_minor'] * nights, product['currency'])}",
                    f"酒店产品ID：{product_id}",
                ])

            if tool_name == "book_car_rental":
                product_id = int(args["rental_id"])
                product = get_product(cursor, "car", product_id)
                cursor.execute(
                    """
                    SELECT c.name, c.location, c.start_date, c.end_date,
                           cp.vehicle_class, cp.brand, cp.model
                    FROM car_rentals c
                    LEFT JOIN car_products cp ON cp.product_id = %s
                    WHERE c.id = %s
                    """,
                    (product["id"], product_id),
                )
                rental = cursor.fetchone()
                if rental is None:
                    raise ValueError("租车产品不存在。")
                start_date, end_date = resolve_service_dates(
                    args.get("start_date"),
                    args.get("end_date"),
                    rental[2],
                    rental[3],
                )
                days = max((end_date - start_date).days, 1)
                vehicle = " ".join(
                    str(value) for value in (rental[5], rental[6], rental[4]) if value
                ) or rental[0]
                return join_summary([
                    "审批事项：租车预订",
                    f"车辆：{vehicle}",
                    f"取还地点：{rental[1]}",
                    f"取车：{start_date.isoformat()}",
                    f"还车：{end_date.isoformat()}",
                    f"租期：{days}天",
                    f"预计金额：{format_amount(product['unit_amount_minor'] * days, product['currency'])}",
                    f"车辆产品ID：{product_id}",
                ])

            product_id = int(args["recommendation_id"])
            product = get_product(cursor, "trip", product_id)
            cursor.execute(
                """
                SELECT t.name, t.location, t.details, tp.duration_minutes
                FROM trip_recommendations t
                LEFT JOIN trip_products tp ON tp.product_id = %s
                WHERE t.id = %s
                """,
                (product["id"], product_id),
            )
            trip = cursor.fetchone()
            if trip is None:
                raise ValueError("行程产品不存在。")
            visit_date = normalize_date(
                args.get("visit_date"),
                date.today() + timedelta(days=1),
            )
            participant_count = max(int(args.get("participant_count", 1)), 1)
            duration = f"{trip[3]}分钟" if trip[3] else "以产品说明为准"
            return join_summary([
                "审批事项：行程预订",
                f"行程：{trip[0]}",
                f"地点：{trip[1]}",
                f"日期：{visit_date.isoformat()}",
                f"人数：{participant_count}人",
                f"时长：{duration}",
                f"预计金额：{format_amount(product['unit_amount_minor'] * participant_count, product['currency'])}",
                f"行程产品ID：{product_id}",
            ])


def summarize_order_operation(tool_name: str, args: Dict[str, Any]) -> str:
    order_id = args.get("order_id")
    if order_id is None:
        return summarize_fallback(tool_name, args)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT order_no, order_type, total_amount_minor, currency,
                       status, service_start_at, service_end_at
                FROM orders WHERE id = %s
                """,
                (order_id,),
            )
            order = cursor.fetchone()
    if order is None:
        return summarize_fallback(tool_name, args)

    changes = [
        f"{ARGUMENT_LABELS.get(key, key)}：{format_datetime(value)}"
        for key, value in args.items()
        if key != "order_id"
    ]
    return join_summary([
        f"审批事项：{TOOL_LABELS.get(tool_name, tool_name)}",
        f"订单号：{order[0]}",
        f"订单类型：{ORDER_TYPE_LABELS.get(order[1], order[1])}",
        f"当前状态：{STATUS_LABELS.get(order[4], order[4])}",
        f"服务时间：{format_datetime(order[5])} 至 {format_datetime(order[6])}",
        f"订单金额：{format_amount(order[2], order[3])}",
        *changes,
    ])


def summarize_fallback(tool_name: str, args: Dict[str, Any]) -> str:
    details = [
        f"{ARGUMENT_LABELS.get(key, key)}：{format_datetime(value)}"
        for key, value in args.items()
    ]
    return join_summary([
        f"审批事项：{TOOL_LABELS.get(tool_name, tool_name)}",
        *details,
    ])


def summarize_action(action: Dict[str, Any]) -> str:
    summaries = []
    for tool_call in action.get("tool_calls") or []:
        tool_name = str(tool_call.get("name") or "未知操作")
        args = tool_call.get("args") or {}
        if tool_name in {
            "book_flight",
            "book_hotel",
            "book_car_rental",
            "book_excursion",
        }:
            summaries.append(summarize_booking(tool_name, args))
        elif tool_name in TOOL_LABELS:
            summaries.append(summarize_order_operation(tool_name, args))
        else:
            summaries.append(summarize_fallback(tool_name, args))
    return "；".join(summaries) or "审批事项：旅行订单操作"
