import json
from typing import Any, Dict, Optional

import httpx

from customer_support_chat.app.core.settings import get_settings
from web_app.app.core.user_data_manager import attach_external_approval
from web_app.app.services.feishu_approval_summary import summarize_action


class FeishuApprovalError(RuntimeError):
    pass


def get_missing_configuration() -> list[str]:
    settings = get_settings()
    required = {
        "FEISHU_APP_ID": settings.FEISHU_APP_ID,
        "FEISHU_APP_SECRET": settings.FEISHU_APP_SECRET,
        "FEISHU_APPROVAL_CODE": settings.FEISHU_APPROVAL_CODE,
        "FEISHU_APPLICANT_USER_ID": settings.FEISHU_APPLICANT_USER_ID,
        "FEISHU_FORM_FIELD_ID": settings.FEISHU_FORM_FIELD_ID,
    }
    return [name for name, value in required.items() if not value]


def is_feishu_approval_enabled() -> bool:
    settings = get_settings()
    return settings.FEISHU_ENABLED and not get_missing_configuration()


async def check_feishu_readiness() -> Dict[str, Any]:
    """实时检查飞书令牌、通讯录权限和审批定义访问状态。"""
    settings = get_settings()
    result: Dict[str, Any] = {
        "enabled": is_feishu_approval_enabled(),
        "missing": get_missing_configuration(),
        "token_valid": False,
        "contact_access": False,
        "approval_definition_access": None,
        "required_scopes": [
            "approval:instance",
            "approval:task",
            "contact:contact.base:readonly",
        ],
    }
    if not settings.FEISHU_APP_ID or not settings.FEISHU_APP_SECRET:
        return result

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            f"{settings.FEISHU_API_BASE_URL}/open-apis/auth/v3/tenant_access_token/internal",
            json={
                "app_id": settings.FEISHU_APP_ID,
                "app_secret": settings.FEISHU_APP_SECRET,
            },
        )
        token_data = token_response.json()
        token = token_data.get("tenant_access_token")
        result["token_valid"] = token_response.status_code == 200 and token_data.get("code") == 0 and bool(token)
        if not token:
            return result

        headers = {"Authorization": f"Bearer {token}"}
        contact_response = await client.get(
            f"{settings.FEISHU_API_BASE_URL}/open-apis/contact/v3/users/find_by_department",
            headers=headers,
            params={
                "department_id": "0",
                "page_size": 1,
                "user_id_type": "user_id",
                "department_id_type": "department_id",
            },
        )
        contact_data = contact_response.json()
        result["contact_access"] = contact_response.status_code == 200 and contact_data.get("code") == 0

        if settings.FEISHU_APPROVAL_CODE:
            approval_response = await client.get(
                f"{settings.FEISHU_API_BASE_URL}/open-apis/approval/v4/approvals/{settings.FEISHU_APPROVAL_CODE}",
                headers=headers,
            )
            approval_data = approval_response.json()
            result["approval_definition_access"] = (
                approval_response.status_code == 200 and approval_data.get("code") == 0
            )
    return result


async def request_feishu(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    accepted_error_codes: Optional[set[int]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=20) as client:
        token_response = await client.post(
            f"{settings.FEISHU_API_BASE_URL}/open-apis/auth/v3/tenant_access_token/internal",
            json={
                "app_id": settings.FEISHU_APP_ID,
                "app_secret": settings.FEISHU_APP_SECRET,
            },
        )
        token_data = token_response.json()
        if token_response.status_code != 200 or token_data.get("code") != 0:
            raise FeishuApprovalError(token_data.get("msg", "获取飞书访问令牌失败。"))

        response = await client.request(
            method,
            f"{settings.FEISHU_API_BASE_URL}{path}",
            headers={
                "Authorization": f"Bearer {token_data['tenant_access_token']}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=json_body,
            params=params,
        )
        data = response.json()
        error_code = data.get("code")
        accepted_error = error_code in (accepted_error_codes or set())
        if not accepted_error and (response.status_code >= 400 or error_code != 0):
            raise FeishuApprovalError(data.get("msg", "飞书审批接口调用失败。"))
        return data.get("data") or {}


async def subscribe_approval_events() -> Dict[str, Any]:
    """订阅当前审批定义的事件，重复订阅按幂等成功处理。"""
    settings = get_settings()
    return await request_feishu(
        "POST",
        f"/open-apis/approval/v4/approvals/{settings.FEISHU_APPROVAL_CODE}/subscribe",
        accepted_error_codes={1390007},
    )


async def create_approval_instance(
    action_id: int,
    action: Dict[str, Any],
) -> Optional[str]:
    if not is_feishu_approval_enabled():
        return None
    settings = get_settings()
    form = [{
        "id": settings.FEISHU_FORM_FIELD_ID,
        "type": "input",
        "value": summarize_action(action),
    }]
    body: Dict[str, Any] = {
        "approval_code": settings.FEISHU_APPROVAL_CODE,
        "user_id": settings.FEISHU_APPLICANT_USER_ID,
        "form": json.dumps(form, ensure_ascii=False),
        "uuid": f"travel-action-{action_id}",
    }
    data = await request_feishu(
        "POST",
        "/open-apis/approval/v4/instances",
        json_body=body,
    )
    instance_code = data.get("instance_code")
    if not instance_code:
        raise FeishuApprovalError("飞书没有返回审批实例 Code。")
    attach_external_approval(
        action_id,
        provider="feishu",
        instance_code=instance_code,
        external_status="PENDING",
    )
    return instance_code


async def get_approval_instance(instance_code: str) -> Dict[str, Any]:
    return await request_feishu(
        "GET",
        f"/open-apis/approval/v4/instances/{instance_code}",
    )


async def cancel_approval_instance(instance_code: str, user_id: str) -> Dict[str, Any]:
    settings = get_settings()
    return await request_feishu(
        "POST",
        "/open-apis/approval/v4/instances/cancel",
        json_body={
            "approval_code": settings.FEISHU_APPROVAL_CODE,
            "instance_code": instance_code,
            "user_id": user_id,
        },
        params={"user_id_type": "user_id"},
    )


async def approve_task(
    instance_code: str,
    task_id: str,
    user_id: str,
    *,
    user_id_type: str = "user_id",
) -> Dict[str, Any]:
    settings = get_settings()
    return await request_feishu(
        "POST",
        "/open-apis/approval/v4/tasks/approve",
        json_body={
            "approval_code": settings.FEISHU_APPROVAL_CODE,
            "instance_code": instance_code,
            "user_id": user_id,
            "task_id": task_id,
            "comment": "旅行订单审批通过",
        },
        params={"user_id_type": user_id_type},
    )
