import asyncio
import json
import unittest
import uuid
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


# 导入图和 API 时屏蔽 Qdrant 健康检查，避免测试依赖向量服务。
with patch("qdrant_client.QdrantClient.get_collections") as get_collections:
    get_collections.return_value = SimpleNamespace(collections=[])
    from customer_support_chat.app.core.database import get_connection
    from customer_support_chat.app.services import chat_service
    from customer_support_chat.app.services.order_service import (
        cancel_order,
        create_car_order,
        create_flight_order,
        create_hotel_order,
        create_trip_order,
        get_order_detail,
        list_user_orders,
        resolve_service_dates,
        update_car_order,
        update_flight_order,
        update_hotel_order,
        update_trip_order,
        SupplierBookingError,
    )
    from web_app.app.core.auth_manager import create_auth_token, create_user
    from web_app.app.core.user_data_manager import (
        attach_external_approval,
        set_pending_action,
        sync_session_identity,
    )
    from web_app.app.main import app
    from web_app.app.services.feishu_event_service import apply_feishu_status
    from web_app.app.services.feishu_long_connection import build_event_handler
    from customer_support_chat.app.services.tools.cars import cars_vectordb, search_car_rentals
    from customer_support_chat.app.services.tools.excursions import excursions_vectordb, search_trip_recommendations
    from customer_support_chat.app.services.tools.hotels import hotels_vectordb, search_hotels


def tearDownModule():
    from lark_oapi.ws import client as lark_ws_client

    if not lark_ws_client.loop.is_running() and not lark_ws_client.loop.is_closed():
        lark_ws_client.loop.close()


class OrderFlowRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.feishu_patch = patch(
            "web_app.app.main.create_approval_instance",
            new=AsyncMock(return_value=None),
        )
        self.feishu_patch.start()
        self.addCleanup(self.feishu_patch.stop)
        suffix = uuid.uuid4().hex[:10]
        self.users = [
            create_user(f"order-flow-{suffix}-a", "test-password"),
            create_user(f"order-flow-{suffix}-b", "test-password"),
        ]
        self.pending_action = {
            "tool_calls": [
                {
                    "id": "call-book-flight",
                    "name": "book_flight",
                    "args": {"flight_id": 1},
                }
            ]
        }
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT booked, owner_passenger_id, checkin_date, checkout_date FROM hotels WHERE id = 1")
                self.hotel_state = cursor.fetchone()
                cursor.execute("SELECT booked, owner_passenger_id, start_date, end_date FROM car_rentals WHERE id = 1")
                self.car_state = cursor.fetchone()
                cursor.execute("SELECT booked FROM trip_recommendations WHERE id = 1")
                self.trip_state = cursor.fetchone()

    def tearDown(self):
        user_ids = [user["id"] for user in self.users]
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM orders WHERE user_id = ANY(%s)", (user_ids,))
                cursor.execute(
                    "UPDATE hotels SET booked = %s, owner_passenger_id = %s, checkin_date = %s, checkout_date = %s WHERE id = 1",
                    self.hotel_state,
                )
                cursor.execute(
                    "UPDATE car_rentals SET booked = %s, owner_passenger_id = %s, start_date = %s, end_date = %s WHERE id = 1",
                    self.car_state,
                )
                cursor.execute(
                    "UPDATE trip_recommendations SET booked = %s WHERE id = 1",
                    self.trip_state,
                )
                cursor.execute("DELETE FROM user_sessions WHERE user_id = ANY(%s)", (user_ids,))
                cursor.execute("DELETE FROM users WHERE id = ANY(%s)", (user_ids,))

    def build_session(self, user):
        session_id = f"test-order-flow-{user['id']}"
        return {
            "session_id": session_id,
            "config": {
                "thread_id": session_id,
                "passenger_id": user["passenger_id"],
                "user_id": user["id"],
            },
        }

    async def process_decision(self, user, decision):
        with patch.object(chat_service, "approvals_enabled", return_value=True), patch.object(
            chat_service, "claim_pending_action", return_value=self.pending_action
        ), patch.object(chat_service, "add_operation_log"), patch.object(
            chat_service, "resolve_pending_action"
        ):
            return await chat_service.process_user_decision(
                self.build_session(user),
                decision,
            )

    async def test_reject_does_not_create_order(self):
        result = await self.process_decision(self.users[0], "reject")

        self.assertEqual(result, "操作已被用户取消。")
        self.assertEqual(list_user_orders(self.users[0]["id"]), [])

    def test_stale_demo_dates_default_to_future(self):
        today = date.today()
        start_date, end_date = resolve_service_dates(
            None,
            None,
            today - timedelta(days=2),
            today - timedelta(days=1),
        )

        self.assertEqual(start_date, today + timedelta(days=1))
        self.assertEqual(end_date, today + timedelta(days=2))

    async def test_approve_creates_flight_order(self):
        result = await self.process_decision(self.users[0], "approve")
        orders = list_user_orders(self.users[0]["id"])
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT response_payload FROM supplier_booking_attempts WHERE order_id = %s",
                    (orders[0]["id"],),
                )
                response_payload = cursor.fetchone()[0]

        self.assertIn("航班订单创建成功", result)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["order_type"], "flight")
        self.assertEqual(orders[0]["status"], "confirmed")
        self.assertTrue(orders[0]["supplier_confirmation_no"].startswith("DB-FLIGHT-"))
        self.assertEqual(response_payload["provider"], "postgresql")

    async def test_repeated_approval_keeps_one_order(self):
        await self.process_decision(self.users[0], "approve")
        result = await self.process_decision(self.users[0], "approve")

        self.assertIn("订单已存在", result)
        self.assertEqual(len(list_user_orders(self.users[0]["id"])), 1)

    async def test_orders_endpoint_only_returns_current_user_orders(self):
        create_flight_order(1, self.users[0]["passenger_id"])
        create_flight_order(2, self.users[1]["passenger_id"])

        with TestClient(app) as client:
            client.cookies.set("auth_token", create_auth_token(self.users[0]["id"]))
            response = client.get("/orders")

        self.assertEqual(response.status_code, 200)
        orders = response.json()["orders"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(
            orders[0]["order_no"],
            list_user_orders(self.users[0]["id"])[0]["order_no"],
        )

    async def approve_tool(self, tool_name, args):
        self.pending_action = {
            "tool_calls": [{
                "id": f"call-{tool_name}",
                "name": tool_name,
                "args": args,
            }]
        }
        return await self.process_decision(self.users[0], "approve")

    async def test_approve_creates_hotel_order(self):
        result = await self.approve_tool("book_hotel", {"hotel_id": 1})
        order = list_user_orders(self.users[0]["id"])[0]

        self.assertIn("酒店订单创建成功", result)
        self.assertEqual(order["order_type"], "hotel")
        self.assertIsNotNone(get_order_detail(self.users[0]["id"], order["id"])["detail"])

    async def test_feishu_long_connection_approval_creates_hotel_order(self):
        session_data = self.build_session(self.users[0])
        sync_session_identity(
            session_data["session_id"],
            self.users[0],
            session_data["config"],
        )
        action_id = set_pending_action(
            session_data["session_id"],
            {
                "tool_calls": [{
                    "id": "call-feishu-book-hotel",
                    "name": "book_hotel",
                    "args": {"hotel_id": 1},
                }],
            },
        )
        attach_external_approval(
            action_id,
            provider="feishu",
            instance_code="feishu-instance-hotel",
            external_status="PENDING",
        )
        event_handler = build_event_handler(apply_feishu_status)
        payload = json.dumps({
            "uuid": "event-hotel-approved",
            "event": {
                "type": "approval_instance",
                "instance_code": "feishu-instance-hotel",
                "status": "APPROVED",
            },
        }).encode("utf-8")

        approval_state = None
        with patch.object(chat_service, "approvals_enabled", return_value=True):
            event_handler._do_without_validation(payload)
            for _ in range(30):
                with get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT status, external_status FROM pending_actions WHERE id = %s",
                            (action_id,),
                        )
                        approval_state = cursor.fetchone()
                if list_user_orders(self.users[0]["id"]) and approval_state[0] == "approved":
                    break
                await asyncio.sleep(0.1)

        orders = list_user_orders(self.users[0]["id"])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["order_type"], "hotel")
        self.assertEqual(orders[0]["status"], "confirmed")
        self.assertEqual(approval_state, ("approved", "APPROVED"))

    async def test_approve_creates_car_order(self):
        result = await self.approve_tool("book_car_rental", {"rental_id": 1})
        order = list_user_orders(self.users[0]["id"])[0]

        self.assertIn("租车订单创建成功", result)
        self.assertEqual(order["order_type"], "car")
        self.assertIsNotNone(get_order_detail(self.users[0]["id"], order["id"])["detail"])

    async def test_approve_creates_trip_order(self):
        result = await self.approve_tool("book_excursion", {"recommendation_id": 1})
        order = list_user_orders(self.users[0]["id"])[0]

        self.assertIn("行程订单创建成功", result)
        self.assertEqual(order["order_type"], "trip")
        self.assertIsNotNone(get_order_detail(self.users[0]["id"], order["id"])["detail"])

    async def test_booked_inventory_rejects_another_user(self):
        first_passenger = self.users[0]["passenger_id"]
        second_passenger = self.users[1]["passenger_id"]

        create_hotel_order(1, first_passenger)
        with self.assertRaisesRegex(ValueError, "酒店产品已被预订"):
            create_hotel_order(1, second_passenger)

        create_car_order(1, first_passenger)
        with self.assertRaisesRegex(ValueError, "租车产品已被预订"):
            create_car_order(1, second_passenger)

        create_trip_order(1, first_passenger)
        with self.assertRaisesRegex(ValueError, "行程产品已被预订"):
            create_trip_order(1, second_passenger)

        self.assertEqual(list_user_orders(self.users[1]["id"]), [])

    async def test_cancel_updates_order_and_status_history(self):
        create_hotel_order(1, self.users[0]["passenger_id"])
        order = list_user_orders(self.users[0]["id"])[0]

        result = cancel_order(order["id"], self.users[0]["passenger_id"])
        detail = get_order_detail(self.users[0]["id"], order["id"])

        self.assertIn("已成功取消", result)
        self.assertEqual(detail["status"], "cancelled")
        self.assertEqual(detail["status_history"][-1]["to_status"], "cancelled")

    async def test_update_all_order_types_changes_formal_order(self):
        passenger_id = self.users[0]["passenger_id"]
        start_date = date.today() + timedelta(days=10)
        end_date = start_date + timedelta(days=3)

        create_hotel_order(1, passenger_id)
        hotel_order = list_user_orders(self.users[0]["id"])[0]
        update_hotel_order(hotel_order["id"], passenger_id, start_date, end_date)
        self.assertEqual(
            get_order_detail(self.users[0]["id"], hotel_order["id"])["detail"]["checkin_date"],
            start_date.isoformat(),
        )

        create_car_order(1, passenger_id)
        car_order = next(order for order in list_user_orders(self.users[0]["id"]) if order["order_type"] == "car")
        update_car_order(car_order["id"], passenger_id, start_date, end_date)
        self.assertEqual(
            get_order_detail(self.users[0]["id"], car_order["id"])["total_amount_minor"],
            29900 * 3,
        )

        create_trip_order(1, passenger_id)
        trip_order = next(order for order in list_user_orders(self.users[0]["id"]) if order["order_type"] == "trip")
        update_trip_order(trip_order["id"], passenger_id, start_date, 3)
        self.assertEqual(
            get_order_detail(self.users[0]["id"], trip_order["id"])["detail"]["participant_count"],
            3,
        )

        create_flight_order(1, passenger_id)
        flight_order = next(order for order in list_user_orders(self.users[0]["id"]) if order["order_type"] == "flight")
        update_flight_order(flight_order["id"], passenger_id, 2)
        self.assertEqual(
            get_order_detail(self.users[0]["id"], flight_order["id"])["segments"][0]["flight_no"],
            "CZ3156",
        )

    async def test_order_detail_and_cancel_request_api_flow(self):
        create_hotel_order(1, self.users[0]["passenger_id"])
        order = list_user_orders(self.users[0]["id"])[0]

        with TestClient(app) as client:
            client.cookies.set("auth_token", create_auth_token(self.users[0]["id"]))
            detail_response = client.get(f"/orders/{order['id']}")
            request_response = client.post(f"/orders/{order['id']}/cancel-request")
            pending_response = client.get("/pending-action")
            approval_response = client.post("/approve-action")
            cancelled_response = client.get(f"/orders/{order['id']}")

        self.assertEqual(detail_response.status_code, 200)
        self.assertTrue(detail_response.json()["order"]["detail"]["hotel_name"])
        self.assertEqual(request_response.status_code, 200)
        self.assertEqual(
            pending_response.json()["pending_action"]["tool_calls"][0]["name"],
            "cancel_hotel",
        )
        self.assertIn(
            "审批事项：取消酒店订单",
            pending_response.json()["pending_action"]["approval_summary"],
        )
        self.assertIn(
            order["order_no"],
            pending_response.json()["pending_action"]["approval_summary"],
        )
        self.assertIn("已成功取消", approval_response.json()["response"])
        self.assertEqual(cancelled_response.json()["order"]["status"], "cancelled")

    async def test_vector_search_uses_postgresql_dynamic_status(self):
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE hotels SET booked = TRUE WHERE id = 1")
                cursor.execute("UPDATE car_rentals SET booked = TRUE WHERE id = 1")
                cursor.execute("UPDATE trip_recommendations SET booked = TRUE WHERE id = 1")

        stale_payloads = {
            "hotel": {"id": 1, "content": "酒店固定介绍", "booked": False},
            "car": {"id": 1, "content": "车辆固定介绍", "booked": False},
            "trip": {"id": 1, "content": "行程固定介绍", "booked": False},
        }
        with patch.object(
            hotels_vectordb,
            "search",
            return_value=[SimpleNamespace(payload=stale_payloads["hotel"], score=0.9)],
        ), patch.object(
            cars_vectordb,
            "search",
            return_value=[SimpleNamespace(payload=stale_payloads["car"], score=0.9)],
        ), patch.object(
            excursions_vectordb,
            "search",
            return_value=[SimpleNamespace(payload=stale_payloads["trip"], score=0.9)],
        ):
            hotel = search_hotels.invoke({"query": "深圳酒店", "limit": 1})[0]
            car = search_car_rentals.invoke({"query": "深圳租车", "limit": 1})[0]
            trip = search_trip_recommendations.invoke({"query": "深圳行程", "limit": 1})[0]

        self.assertTrue(hotel["booked"])
        self.assertTrue(car["booked"])
        self.assertTrue(trip["booked"])

    async def test_supplier_failure_marks_order_failed_and_records_attempt(self):
        failing_gateway = SimpleNamespace(
            book=Mock(side_effect=RuntimeError("供应商暂时不可用")),
        )
        with patch(
            "customer_support_chat.app.services.order_service.get_supplier_gateway",
            return_value=failing_gateway,
        ):
            with self.assertRaises(SupplierBookingError):
                create_flight_order(1, self.users[0]["passenger_id"])

        order = list_user_orders(self.users[0]["id"])[0]
        detail = get_order_detail(self.users[0]["id"], order["id"])
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT status, error_message FROM supplier_booking_attempts WHERE order_id = %s",
                    (order["id"],),
                )
                attempt = cursor.fetchone()

        self.assertEqual(order["status"], "failed")
        self.assertEqual(detail["status_history"][-1]["to_status"], "failed")
        self.assertEqual(attempt[0], "failed")
        self.assertIn("供应商暂时不可用", attempt[1])

        with TestClient(app) as client:
            client.cookies.set("auth_token", create_auth_token(self.users[0]["id"]))
            retry_request = client.post(f"/orders/{order['id']}/retry-request")
            retry_approval = client.post("/approve-action")
            retried = client.get(f"/orders/{order['id']}").json()["order"]

        self.assertEqual(retry_request.status_code, 200)
        self.assertEqual(
            retry_request.json()["pending_action"]["tool_calls"][0]["name"],
            "retry_order_booking",
        )
        self.assertEqual(retry_approval.status_code, 200)
        self.assertEqual(retried["status"], "confirmed")
        self.assertTrue(retried["supplier_confirmation_no"].startswith("DB-FLIGHT-"))
        self.assertEqual(
            [attempt["status"] for attempt in retried["supplier_attempts"]],
            ["failed", "succeeded"],
        )

    async def test_supplier_cancel_failure_keeps_order_confirmed(self):
        passenger_id = self.users[0]["passenger_id"]
        create_hotel_order(1, passenger_id)
        order = list_user_orders(self.users[0]["id"])[0]
        failing_gateway = SimpleNamespace(
            cancel=Mock(side_effect=RuntimeError("供应商取消失败")),
        )

        with patch(
            "customer_support_chat.app.services.order_service.get_supplier_gateway",
            return_value=failing_gateway,
        ):
            with self.assertRaises(SupplierBookingError):
                cancel_order(order["id"], passenger_id)

        detail = get_order_detail(self.users[0]["id"], order["id"])
        self.assertEqual(detail["status"], "confirmed")
        self.assertEqual(detail["supplier_attempts"][-1]["status"], "failed")

    async def test_supplier_update_failure_keeps_order_detail_unchanged(self):
        passenger_id = self.users[0]["passenger_id"]
        create_hotel_order(1, passenger_id)
        order = list_user_orders(self.users[0]["id"])[0]
        before = get_order_detail(self.users[0]["id"], order["id"])
        failing_gateway = SimpleNamespace(
            update=Mock(side_effect=RuntimeError("供应商改期失败")),
        )

        with patch(
            "customer_support_chat.app.services.order_service.get_supplier_gateway",
            return_value=failing_gateway,
        ):
            with self.assertRaises(SupplierBookingError):
                update_hotel_order(
                    order["id"],
                    passenger_id,
                    date.today() + timedelta(days=30),
                    date.today() + timedelta(days=32),
                )

        after = get_order_detail(self.users[0]["id"], order["id"])
        self.assertEqual(after["status"], "confirmed")
        self.assertEqual(after["detail"]["checkin_date"], before["detail"]["checkin_date"])
        self.assertEqual(after["supplier_attempts"][-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
