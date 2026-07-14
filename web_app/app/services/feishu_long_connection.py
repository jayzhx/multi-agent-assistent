import asyncio
import logging
from typing import Awaitable, Callable

import lark_oapi as lark
from lark_oapi.event.custom import CustomizedEvent

from customer_support_chat.app.core.settings import get_settings
from web_app.app.core.user_data_manager import attach_external_approval_task
from web_app.app.services.feishu_approval_service import subscribe_approval_events
from web_app.app.services.feishu_event_service import apply_feishu_status


logger = logging.getLogger(__name__)
StatusHandler = Callable[[str, str], Awaitable[str]]


def handle_approval_instance_event(
    event: CustomizedEvent,
    status_handler: StatusHandler,
) -> None:
    """快速确认飞书事件，并在 SDK 事件循环中异步处理订单审批。"""
    event_data = event.event or {}
    instance_code = event_data.get("instance_code")
    external_status = event_data.get("status")
    if not instance_code or not external_status:
        logger.warning("忽略缺少 instance_code 或 status 的飞书审批事件。")
        return

    task = asyncio.get_running_loop().create_task(
        status_handler(str(instance_code), str(external_status))
    )
    task.add_done_callback(log_status_result)


def log_status_result(task: asyncio.Task[str]) -> None:
    try:
        logger.info("飞书审批事件处理完成：%s", task.result())
    except Exception:
        logger.exception("飞书审批事件处理失败。")


def ignore_non_order_event(event: object) -> None:
    """审批模板和应用版本事件与订单无关，只需正常确认。"""
    logger.info("已忽略与订单无关的飞书系统事件。")


def handle_approval_task_event(event: CustomizedEvent) -> None:
    """保存当前审批任务，但不触发订单状态变更。"""
    event_data = event.event or {}
    instance_code = event_data.get("instance_code") or event_data.get("approval_instance_code")
    task_id = event_data.get("task_id")
    approver_id = event_data.get("user_id") or event_data.get("open_id")
    approver_id_type = (
        "user_id" if event_data.get("user_id")
        else "open_id" if event_data.get("open_id")
        else None
    )
    if not instance_code or not task_id:
        logger.warning(
            "忽略字段不完整的飞书任务事件：instance=%s task=%s keys=%s",
            bool(instance_code),
            bool(task_id),
            sorted(event_data.keys()),
        )
        return
    attach_external_approval_task(
        str(instance_code),
        str(task_id),
        str(approver_id) if approver_id else None,
        approver_id_type,
    )
    logger.info("已保存飞书审批任务信息。")


def build_event_handler(
    status_handler: StatusHandler = apply_feishu_status,
) -> lark.EventDispatcherHandler:
    return (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_customized_event(
            "approval.approval.created_v4",
            ignore_non_order_event,
        )
        .register_p2_approval_approval_updated_v4(
            ignore_non_order_event,
        )
        .register_p2_application_application_app_version_audit_v6(
            ignore_non_order_event,
        )
        .register_p1_customized_event(
            "approval_instance",
            lambda event: handle_approval_instance_event(event, status_handler),
        )
        .register_p1_customized_event(
            "approval_task",
            handle_approval_task_event,
        )
        .register_p1_customized_event(
            "approval_cc",
            ignore_non_order_event,
        )
        .register_p1_customized_event(
            "approval",
            ignore_non_order_event,
        )
        .build()
    )


def build_ws_client(
    status_handler: StatusHandler = apply_feishu_status,
) -> lark.ws.Client:
    settings = get_settings()
    if not settings.FEISHU_APP_ID or not settings.FEISHU_APP_SECRET:
        raise RuntimeError("请先配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET。")

    log_level = getattr(
        lark.LogLevel,
        settings.FEISHU_SDK_LOG_LEVEL.upper(),
        lark.LogLevel.WARNING,
    )
    return lark.ws.Client(
        settings.FEISHU_APP_ID,
        settings.FEISHU_APP_SECRET,
        log_level=log_level,
        event_handler=build_event_handler(status_handler),
        domain=settings.FEISHU_API_BASE_URL,
        auto_reconnect=True,
    )


def run_long_connection() -> None:
    asyncio.run(subscribe_approval_events())
    logger.info("已确认当前审批定义的飞书事件订阅。")
    logger.info("正在启动飞书审批事件长连接。")
    build_ws_client().start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_long_connection()
