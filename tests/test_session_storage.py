import unittest
import uuid

from customer_support_chat.app.core.database import get_connection
from web_app.app.core.auth_manager import create_user
from web_app.app.core.user_data_manager import (
    add_operation_log,
    claim_pending_action,
    get_operation_log,
    get_pending_action,
    get_user_session,
    resolve_pending_action,
    set_pending_action,
    sync_session_identity,
    update_user_chat_history,
)


class PostgreSQLSessionStorageTests(unittest.TestCase):
    def setUp(self):
        suffix = uuid.uuid4().hex[:10]
        self.user = create_user(f"session-flow-{suffix}", "test-password")
        self.session_id = f"test-session-{self.user['id']}"
        self.config = {
            "thread_id": self.session_id,
            "passenger_id": self.user["passenger_id"],
            "user_id": self.user["id"],
        }

    def tearDown(self):
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM user_sessions WHERE session_id = %s", (self.session_id,))
                cursor.execute("DELETE FROM users WHERE id = %s", (self.user["id"],))

    def test_chat_history_and_logs_are_stored_in_postgresql(self):
        sync_session_identity(self.session_id, self.user, self.config)
        update_user_chat_history(self.session_id, "查询订单", "当前有一张订单")
        add_operation_log(self.session_id, {
            "type": "tool_result",
            "title": "订单查询",
            "content": "返回一张订单",
        })

        session = get_user_session(self.session_id)

        self.assertEqual(session["chat_history"][0]["user_message"], "查询订单")
        self.assertEqual(session["config"]["user_id"], self.user["id"])
        self.assertEqual(get_operation_log(self.session_id)[0]["title"], "订单查询")

    def test_pending_action_is_claimed_and_resolved_once(self):
        sync_session_identity(self.session_id, self.user, self.config)
        action = {"tool_calls": [{"id": "call-1", "name": "book_hotel", "args": {"hotel_id": 1}}]}
        set_pending_action(self.session_id, action)

        self.assertEqual(get_pending_action(self.session_id), action)
        self.assertEqual(claim_pending_action(self.session_id), action)
        self.assertIsNone(claim_pending_action(self.session_id))

        resolve_pending_action(self.session_id, "approve")
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT status, decision FROM pending_actions WHERE session_id = %s",
                    (self.session_id,),
                )
                status, decision = cursor.fetchone()
        self.assertEqual(status, "approved")
        self.assertEqual(decision, "approve")


if __name__ == "__main__":
    unittest.main()
