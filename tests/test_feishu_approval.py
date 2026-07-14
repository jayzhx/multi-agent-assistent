import asyncio
import unittest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from web_app.app.main import app
from web_app.app import main as web_main
from web_app.app.services import feishu_approval_service
from web_app.app.services import feishu_long_connection


def tearDownModule():
    from lark_oapi.ws import client as lark_ws_client

    if not lark_ws_client.loop.is_running() and not lark_ws_client.loop.is_closed():
        lark_ws_client.loop.close()


class FeishuApprovalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_instance_persists_external_code(self):
        settings = SimpleNamespace(
            FEISHU_APPROVAL_CODE="approval-code",
            FEISHU_APPLICANT_USER_ID="applicant-id",
            FEISHU_FORM_FIELD_ID="widget-id",
        )
        with patch.object(
            feishu_approval_service,
            "is_feishu_approval_enabled",
            return_value=True,
        ), patch.object(
            feishu_approval_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            feishu_approval_service,
            "request_feishu",
            new=AsyncMock(return_value={"instance_code": "instance-1"}),
        ) as request_feishu, patch.object(
            feishu_approval_service,
            "attach_external_approval",
        ) as attach_external_approval:
            instance_code = await feishu_approval_service.create_approval_instance(
                10,
                {"tool_calls": [{"name": "book_hotel", "args": {"hotel_id": 1}}]},
            )

        self.assertEqual(instance_code, "instance-1")
        request_feishu.assert_awaited_once()
        form = json.loads(request_feishu.await_args.kwargs["json_body"]["form"])
        self.assertIn("审批事项：酒店预订", form[0]["value"])
        self.assertIn("酒店：", form[0]["value"])
        self.assertIn("预计金额：", form[0]["value"])
        self.assertNotIn("book_hotel", form[0]["value"])
        self.assertNotIn("{", form[0]["value"])
        attach_external_approval.assert_called_once_with(
            10,
            provider="feishu",
            instance_code="instance-1",
            external_status="PENDING",
        )

    async def test_disabled_feishu_does_not_call_api(self):
        with patch.object(
            feishu_approval_service,
            "is_feishu_approval_enabled",
            return_value=False,
        ), patch.object(
            feishu_approval_service,
            "request_feishu",
            new=AsyncMock(),
        ) as request_feishu:
            result = await feishu_approval_service.create_approval_instance(1, {})

        self.assertIsNone(result)
        request_feishu.assert_not_awaited()

    def test_four_booking_summaries_are_human_readable(self):
        cases = [
            ("book_flight", {"flight_id": 1}, "航班预订"),
            ("book_hotel", {"hotel_id": 1}, "酒店预订"),
            ("book_car_rental", {"rental_id": 1}, "租车预订"),
            ("book_excursion", {"recommendation_id": 1}, "行程预订"),
        ]

        for tool_name, args, label in cases:
            with self.subTest(tool_name=tool_name):
                summary = feishu_approval_service.summarize_action({
                    "tool_calls": [{"name": tool_name, "args": args}],
                })
                self.assertIn(f"审批事项：{label}", summary)
                self.assertIn("预计金额：", summary)
                self.assertNotIn(tool_name, summary)
                self.assertNotIn("{", summary)

    async def test_cancel_and_approve_follow_official_v4_contract(self):
        settings = SimpleNamespace(FEISHU_APPROVAL_CODE="approval-code")
        with patch.object(
            feishu_approval_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            feishu_approval_service,
            "request_feishu",
            new=AsyncMock(return_value={}),
        ) as request_feishu:
            await feishu_approval_service.cancel_approval_instance("instance-1", "user-1")
            await feishu_approval_service.approve_task("instance-1", "task-1", "user-2")

        self.assertEqual(
            request_feishu.await_args_list[0].args,
            ("POST", "/open-apis/approval/v4/instances/cancel"),
        )
        self.assertEqual(
            request_feishu.await_args_list[0].kwargs["json_body"],
            {
                "approval_code": "approval-code",
                "instance_code": "instance-1",
                "user_id": "user-1",
            },
        )
        self.assertEqual(
            request_feishu.await_args_list[1].args,
            ("POST", "/open-apis/approval/v4/tasks/approve"),
        )
        self.assertEqual(request_feishu.await_args_list[1].kwargs["json_body"]["task_id"], "task-1")
        self.assertEqual(
            request_feishu.await_args_list[1].kwargs["params"],
            {"user_id_type": "user_id"},
        )

    async def test_approve_task_supports_open_id(self):
        settings = SimpleNamespace(FEISHU_APPROVAL_CODE="approval-code")
        with patch.object(
            feishu_approval_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            feishu_approval_service,
            "request_feishu",
            new=AsyncMock(return_value={}),
        ) as request_feishu:
            await feishu_approval_service.approve_task(
                "instance-1",
                "task-1",
                "open-id-1",
                user_id_type="open_id",
            )

        self.assertEqual(
            request_feishu.await_args.kwargs["params"],
            {"user_id_type": "open_id"},
        )

    async def test_subscribe_approval_events_uses_definition_code(self):
        settings = SimpleNamespace(FEISHU_APPROVAL_CODE="approval-code")
        with patch.object(
            feishu_approval_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            feishu_approval_service,
            "request_feishu",
            new=AsyncMock(return_value={}),
        ) as request_feishu:
            await feishu_approval_service.subscribe_approval_events()

        request_feishu.assert_awaited_once_with(
            "POST",
            "/open-apis/approval/v4/approvals/approval-code/subscribe",
            accepted_error_codes={1390007},
        )

    async def test_config_status_uses_live_readiness_without_exposing_secrets(self):
        readiness = {
            "enabled": False,
            "missing": ["FEISHU_APPROVAL_CODE"],
            "token_valid": True,
            "contact_access": False,
            "approval_definition_access": None,
            "required_scopes": ["approval:instance"],
        }
        with patch.object(
            web_main,
            "check_feishu_readiness",
            new=AsyncMock(return_value=readiness),
        ):
            response = await web_main.get_feishu_config_status({"id": 1})

        data = json.loads(response.body)
        self.assertEqual(data, readiness)
        self.assertNotIn("app_secret", data)


class FeishuCallbackTests(unittest.TestCase):
    def test_url_verification_challenge(self):
        with patch.object(
            web_main.settings,
            "FEISHU_VERIFICATION_TOKEN",
            "test-token",
        ), TestClient(app) as client:
            response = client.post("/feishu/events", json={
                "token": "test-token",
                "challenge": "challenge-value",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"challenge": "challenge-value"})

    def test_approval_event_is_forwarded(self):
        with patch(
            "web_app.app.main.apply_feishu_status",
            new=AsyncMock(return_value="订单创建成功"),
        ) as apply_status, patch.object(
            web_main.settings,
            "FEISHU_VERIFICATION_TOKEN",
            "test-token",
        ), TestClient(app) as client:
            response = client.post("/feishu/events", json={
                "token": "test-token",
                "event": {
                    "instance_code": "instance-1",
                    "status": "APPROVED",
                }
            })

        self.assertEqual(response.status_code, 200)
        apply_status.assert_awaited_once_with("instance-1", "APPROVED")


class FeishuLongConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_dispatches_approval_instance_event(self):
        status_handler = AsyncMock(return_value="订单创建成功")
        event_handler = feishu_long_connection.build_event_handler(status_handler)
        payload = json.dumps({
            "uuid": "event-1",
            "event": {
                "type": "approval_instance",
                "instance_code": "instance-1",
                "status": "APPROVED",
            },
        }).encode("utf-8")

        event_handler._do_without_validation(payload)
        await asyncio.sleep(0)

        status_handler.assert_awaited_once_with("instance-1", "APPROVED")

    async def test_incomplete_event_is_ignored(self):
        status_handler = AsyncMock()
        event_handler = feishu_long_connection.build_event_handler(status_handler)
        payload = json.dumps({
            "uuid": "event-2",
            "event": {
                "type": "approval_instance",
                "instance_code": "instance-2",
            },
        }).encode("utf-8")

        event_handler._do_without_validation(payload)
        await asyncio.sleep(0)

        status_handler.assert_not_awaited()

    async def test_approval_task_event_does_not_trigger_order(self):
        status_handler = AsyncMock()
        with patch.object(
            feishu_long_connection,
            "attach_external_approval_task",
        ) as attach_task:
            event_handler = feishu_long_connection.build_event_handler(status_handler)
            payload = json.dumps({
                "uuid": "event-task-pending",
                "event": {
                    "type": "approval_task",
                    "instance_code": "instance-2",
                    "task_id": "task-1",
                    "open_id": "open-id-1",
                    "status": "PENDING",
                },
            }).encode("utf-8")

            event_handler._do_without_validation(payload)
            await asyncio.sleep(0)

        status_handler.assert_not_awaited()
        attach_task.assert_called_once_with(
            "instance-2",
            "task-1",
            "open-id-1",
            "open_id",
        )

    async def test_approval_cc_event_does_not_trigger_order(self):
        status_handler = AsyncMock()
        event_handler = feishu_long_connection.build_event_handler(status_handler)
        payload = json.dumps({
            "uuid": "event-cc-created",
            "event": {"type": "approval_cc", "instance_code": "instance-2"},
        }).encode("utf-8")

        event_handler._do_without_validation(payload)
        await asyncio.sleep(0)

        status_handler.assert_not_awaited()

    async def test_approval_task_without_approver_still_persists_task(self):
        status_handler = AsyncMock()
        with patch.object(
            feishu_long_connection,
            "attach_external_approval_task",
        ) as attach_task:
            event_handler = feishu_long_connection.build_event_handler(status_handler)
            payload = json.dumps({
                "uuid": "event-auto-task",
                "event": {
                    "type": "approval_task",
                    "instance_code": "instance-auto",
                    "task_id": "task-auto",
                    "status": "PENDING",
                },
            }).encode("utf-8")

            event_handler._do_without_validation(payload)
            await asyncio.sleep(0)

        attach_task.assert_called_once_with(
            "instance-auto",
            "task-auto",
            None,
            None,
        )
        status_handler.assert_not_awaited()

    async def test_approval_definition_event_does_not_trigger_order(self):
        status_handler = AsyncMock()
        event_handler = feishu_long_connection.build_event_handler(status_handler)
        payload = json.dumps({
            "schema": "2.0",
            "header": {
                "event_id": "event-definition-created",
                "event_type": "approval.approval.created_v4",
            },
            "event": {
                "object": {
                    "approval_code": "approval-code",
                },
            },
        }).encode("utf-8")

        event_handler._do_without_validation(payload)
        await asyncio.sleep(0)

        status_handler.assert_not_awaited()

    async def test_application_audit_event_does_not_trigger_order(self):
        status_handler = AsyncMock()
        event_handler = feishu_long_connection.build_event_handler(status_handler)
        payload = json.dumps({
            "schema": "2.0",
            "header": {
                "event_id": "event-app-audit",
                "event_type": "application.application.app_version.audit_v6",
            },
            "event": {},
        }).encode("utf-8")

        event_handler._do_without_validation(payload)
        await asyncio.sleep(0)

        status_handler.assert_not_awaited()



if __name__ == "__main__":
    unittest.main()
