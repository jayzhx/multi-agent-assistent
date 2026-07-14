from customer_support_chat.app.services.chat_service import process_user_decision
from web_app.app.core.user_data_manager import (
    get_action_by_external_instance,
    get_user_session,
    update_external_approval_status,
)


class FeishuEventMappingError(LookupError):
    pass


async def apply_feishu_status(instance_code: str, external_status: str) -> str:
    """将飞书审批终态映射为本地审批决策。"""
    normalized_status = external_status.upper()
    action = get_action_by_external_instance(instance_code)
    if action is None:
        raise FeishuEventMappingError("未找到对应的本地审批动作。")

    update_external_approval_status(instance_code, normalized_status)
    decisions = {
        "APPROVED": "approve",
        "REJECTED": "reject",
    }
    decision = decisions.get(normalized_status)
    if decision is None:
        return f"飞书审批状态已同步为 {normalized_status}。"

    session = get_user_session(action["session_id"])
    return await process_user_decision(
        {
            "session_id": action["session_id"],
            "config": session.get("config") or {},
        },
        decision,
    )
