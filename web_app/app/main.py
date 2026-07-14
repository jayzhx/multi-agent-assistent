from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import date, datetime
from pydantic import BaseModel
import os
import sys
import uuid

# Add the customer_support_chat directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from customer_support_chat.app.services.chat_service import (
    process_user_decision,
    process_user_message,
)
from customer_support_chat.app.services.order_service import (
    get_order_detail,
    query_user_orders,
)
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.humanloop_manager import approvals_enabled
from .core.user_data_manager import (
    get_user_session,
    sync_session_identity,
    update_user_chat_history,
    get_pending_action,
    get_operation_log,
    set_pending_action,
)
from .core.auth_manager import (
    initialize_auth_schema,
    create_user,
    authenticate_user,
    create_auth_token,
    get_user_from_token,
)
from .services.feishu_approval_service import (
    FeishuApprovalError,
    check_feishu_readiness,
    create_approval_instance,
    get_approval_instance,
)
from .services.feishu_approval_summary import summarize_action
from .services.feishu_event_service import (
    FeishuEventMappingError,
    apply_feishu_status,
)

# Load environment variables
load_dotenv()
settings = get_settings()

app = FastAPI()
initialize_auth_schema()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class ApprovalDecision(BaseModel):
    decision: str

class AuthPayload(BaseModel):
    username: str
    password: str


class RegisterPayload(AuthPayload):
    passenger_id: str | None = None


class OrderUpdatePayload(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    new_flight_id: int | None = None
    participant_count: int = 1


def build_public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "passenger_id": user["passenger_id"],
        "created_at": user["created_at"],
    }


def apply_auth_cookie(response: JSONResponse, user_id: int):
    auth_token = create_auth_token(user_id)
    response.set_cookie(
        key="auth_token",
        value=auth_token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )


def clear_auth_cookie(response: JSONResponse):
    response.delete_cookie("auth_token")
    response.delete_cookie("session_id")


def build_public_pending_action(action: dict) -> dict:
    """为审批界面补充可读摘要，同时保留原始工具参数供执行。"""
    return {**action, "approval_summary": summarize_action(action)}


async def create_pending_approval(session_id: str, action: dict) -> dict:
    """保存待审批动作，并在配置完整时同步创建飞书实例。"""
    action_id = set_pending_action(session_id, action)
    instance_code = None
    try:
        instance_code = await create_approval_instance(action_id, action)
    except FeishuApprovalError as error:
        from .core.user_data_manager import add_operation_log
        add_operation_log(session_id, {
            "type": "error",
            "title": "飞书审批创建失败",
            "content": str(error),
        })
    return {
        "pending_action": build_public_pending_action(action),
        "action_id": action_id,
        "feishu_instance_code": instance_code,
    }


def get_current_user(request: Request) -> dict:
    auth_token = request.cookies.get("auth_token")
    current_user = get_user_from_token(auth_token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="请先登录后再继续操作。")
    return current_user


def get_session_data(current_user: dict = Depends(get_current_user)):
    """Get or create session data for the current user."""
    session_id = f"user-{current_user['id']}"
    config = {
        "thread_id": session_id,
        "passenger_id": current_user["passenger_id"],
        "user_id": current_user["id"],
    }
    user_profile = build_public_user(current_user)

    # 将登录用户身份同步到当前会话，避免仍然使用匿名默认乘机人。
    session_data = sync_session_identity(session_id, user_profile, config)

    return {
        "session_id": session_id,
        "config": config,
        "user": user_profile,
        "user_data": session_data
    }


@app.post("/auth/register")
async def register(register_payload: RegisterPayload):
    """Create a user and start a logged-in session."""
    try:
        current_user = create_user(
            register_payload.username,
            register_payload.password,
            register_payload.passenger_id,
        )
    except ValueError as error:
        return JSONResponse(content={"error": str(error)}, status_code=400)

    response = JSONResponse(content={"user": build_public_user(current_user)})
    apply_auth_cookie(response, current_user["id"])
    response.set_cookie(key="session_id", value=f"user-{current_user['id']}", samesite="lax")
    return response


@app.post("/auth/login")
async def login(auth_payload: AuthPayload):
    """Authenticate user and return login state."""
    current_user = authenticate_user(auth_payload.username, auth_payload.password)
    if current_user is None:
        return JSONResponse(content={"error": "用户名或密码错误。"}, status_code=401)

    response = JSONResponse(content={"user": build_public_user(current_user)})
    apply_auth_cookie(response, current_user["id"])
    response.set_cookie(key="session_id", value=f"user-{current_user['id']}", samesite="lax")
    return response


@app.post("/auth/logout")
async def logout():
    """Clear current login state."""
    response = JSONResponse(content={"success": True})
    clear_auth_cookie(response)
    return response


@app.get("/auth/me")
async def get_current_user_endpoint(request: Request):
    """Return current authenticated user info."""
    current_user = get_user_from_token(request.cookies.get("auth_token"))
    if current_user is None:
        return JSONResponse(content={"authenticated": False})

    response = JSONResponse(content={
        "authenticated": True,
        "user": build_public_user(current_user),
    })
    response.set_cookie(key="session_id", value=f"user-{current_user['id']}", samesite="lax")
    return response

@app.get("/")
async def get_chat_page():
    """返回当前 API 服务状态。"""
    return JSONResponse(content={
        "service": "multi-agent-assistent-api",
        "approvals_enabled": settings.ENABLE_APPROVALS,
    })

@app.get("/session")
async def get_session_endpoint(session_data: dict = Depends(get_session_data)):
    """Return current session data for standalone frontends."""
    response = JSONResponse(content={
        "session_id": session_data["session_id"],
        "user": session_data["user"],
        "chat_history": session_data["user_data"].get("chat_history", []),
        "approvals_enabled": settings.ENABLE_APPROVALS,
    })
    response.set_cookie(key="session_id", value=session_data["session_id"])
    return response

@app.post("/chat")
async def chat(chat_message: ChatMessage, session_data: dict = Depends(get_session_data)):
    """Process a chat message and return the AI response."""
    try:
        # Process the user message
        ai_response = await process_user_message(session_data, chat_message.message)

        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], chat_message.message, ai_response)

        # Return the AI response
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing chat message: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)

# HITL (Human-in-the-Loop) endpoints

@app.get("/pending-action")
async def get_pending_action_endpoint(session_data: dict = Depends(get_session_data)):
    """Check if there is a pending action requiring user approval."""
    if not approvals_enabled():
        return JSONResponse(content={"enabled": False, "pending_action": None, "message": "当前未启用审批功能。"})

    try:
        pending_action = get_pending_action(session_data["session_id"])
        if pending_action:
            return JSONResponse(content={"pending_action": build_public_pending_action(pending_action)})
        else:
            return JSONResponse(content={"pending_action": None})
    except Exception as e:
        print(f"Error checking pending action: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


@app.post("/approve-action")
async def approve_action(request: Request, session_data: dict = Depends(get_session_data)):
    """Approve a pending action."""
    if not approvals_enabled():
        return JSONResponse(content={"enabled": False, "response": "当前版本未启用审批功能。"})

    try:
        # Process the user's approval decision
        from customer_support_chat.app.services.chat_service import process_user_decision
        ai_response = await process_user_decision(session_data, "approve")

        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], "[User approved action]", ai_response)

        # Return the AI response
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing approval: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


@app.post("/reject-action")
async def reject_action(request: Request, session_data: dict = Depends(get_session_data)):
    """Reject a pending action."""
    if not approvals_enabled():
        return JSONResponse(content={"enabled": False, "response": "当前版本未启用审批功能。"})

    try:
        # Process the user's rejection decision
        from customer_support_chat.app.services.chat_service import process_user_decision
        ai_response = await process_user_decision(session_data, "reject")

        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], "[User rejected action]", ai_response)

        # Return the AI response
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing rejection: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)

@app.get("/operation-log")
async def get_operation_log_endpoint(session_data: dict = Depends(get_session_data)):
    """Get the operation log for the current session."""
    try:
        # Get only the most recent 20 log entries to reduce data transfer
        operation_log = get_operation_log(session_data["session_id"], limit=20)
        return JSONResponse(content={"operation_log": operation_log})
    except Exception as e:
        print(f"Error retrieving operation log: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


@app.get("/orders")
async def get_orders_endpoint(
    status: str | None = None,
    order_type: str | None = None,
    page: int = 1,
    page_size: int = 10,
    current_user: dict = Depends(get_current_user),
):
    """返回当前登录用户筛选和分页后的订单列表。"""
    result = query_user_orders(
        current_user["id"],
        status=status,
        order_type=order_type,
        page=page,
        page_size=page_size,
    )
    return JSONResponse(content={"orders": result["items"], "pagination": {
        "total": result["total"],
        "page": result["page"],
        "page_size": result["page_size"],
    }})


@app.get("/orders/{order_id}")
async def get_order_detail_endpoint(
    order_id: int,
    current_user: dict = Depends(get_current_user),
):
    """返回当前登录用户的订单详情。"""
    try:
        order = get_order_detail(current_user["id"], order_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return JSONResponse(content={"order": order})


@app.post("/orders/{order_id}/cancel-request")
async def request_order_cancellation(
    order_id: int,
    session_data: dict = Depends(get_session_data),
):
    """为订单取消创建待审批动作。"""
    try:
        order = get_order_detail(session_data["user"]["id"], order_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    tool_names = {
        "flight": "cancel_ticket",
        "hotel": "cancel_hotel",
        "car": "cancel_car_rental",
        "trip": "cancel_excursion",
    }
    action = {
        "tool_calls": [{
            "id": f"order-cancel-{uuid.uuid4().hex}",
            "name": tool_names[order["order_type"]],
            "args": {"order_id": order_id},
        }],
        "timestamp": datetime.now().timestamp(),
    }
    return JSONResponse(content=await create_pending_approval(session_data["session_id"], action))


@app.post("/orders/{order_id}/update-request")
async def request_order_update(
    order_id: int,
    payload: OrderUpdatePayload,
    session_data: dict = Depends(get_session_data),
):
    """为订单改期或改签创建待审批动作。"""
    try:
        order = get_order_detail(session_data["user"]["id"], order_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    order_type = order["order_type"]
    if order_type == "flight":
        if payload.new_flight_id is None:
            raise HTTPException(status_code=400, detail="航班改签需要 new_flight_id。")
        tool_name = "update_ticket_to_new_flight"
        args = {"order_id": order_id, "new_flight_id": payload.new_flight_id}
    elif order_type == "hotel":
        if payload.start_date is None or payload.end_date is None:
            raise HTTPException(status_code=400, detail="酒店修改需要入住和退房日期。")
        tool_name = "update_hotel"
        args = {
            "order_id": order_id,
            "checkin_date": payload.start_date.isoformat(),
            "checkout_date": payload.end_date.isoformat(),
        }
    elif order_type == "car":
        if payload.start_date is None or payload.end_date is None:
            raise HTTPException(status_code=400, detail="租车修改需要取车和还车日期。")
        tool_name = "update_car_rental"
        args = {
            "order_id": order_id,
            "start_date": payload.start_date.isoformat(),
            "end_date": payload.end_date.isoformat(),
        }
    else:
        if payload.start_date is None:
            raise HTTPException(status_code=400, detail="行程修改需要出行日期。")
        tool_name = "update_excursion"
        args = {
            "order_id": order_id,
            "visit_date": payload.start_date.isoformat(),
            "participant_count": payload.participant_count,
        }

    action = {
        "tool_calls": [{
            "id": f"order-update-{uuid.uuid4().hex}",
            "name": tool_name,
            "args": args,
        }],
        "timestamp": datetime.now().timestamp(),
    }
    return JSONResponse(content=await create_pending_approval(session_data["session_id"], action))


@app.post("/orders/{order_id}/retry-request")
async def request_order_retry(
    order_id: int,
    session_data: dict = Depends(get_session_data),
):
    """为失败订单创建供应商下单重试审批。"""
    try:
        order = get_order_detail(session_data["user"]["id"], order_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if order["status"] != "failed":
        raise HTTPException(status_code=400, detail="只有失败订单可以重试。")
    action = {
        "tool_calls": [{
            "id": f"order-retry-{uuid.uuid4().hex}",
            "name": "retry_order_booking",
            "args": {"order_id": order_id},
        }],
        "timestamp": datetime.now().timestamp(),
    }
    return JSONResponse(content=await create_pending_approval(session_data["session_id"], action))


@app.get("/feishu/config-status")
async def get_feishu_config_status(current_user: dict = Depends(get_current_user)):
    """返回飞书审批配置状态，不暴露任何密钥。"""
    return JSONResponse(content=await check_feishu_readiness())


@app.post("/feishu/events")
async def receive_feishu_event(request: Request):
    """接收飞书审批回调和 URL 校验请求。"""
    payload = await request.json()
    token = payload.get("token") or (payload.get("header") or {}).get("token")
    if settings.FEISHU_VERIFICATION_TOKEN and token != settings.FEISHU_VERIFICATION_TOKEN:
        raise HTTPException(status_code=403, detail="飞书回调校验失败。")
    if payload.get("challenge"):
        return JSONResponse(content={"challenge": payload["challenge"]})

    event = payload.get("event") or {}
    instance_code = event.get("instance_code") or event.get("approval_instance_code")
    external_status = event.get("status")
    if not instance_code or not external_status:
        return JSONResponse(content={"success": True, "ignored": True})
    try:
        result = await apply_feishu_status(instance_code, external_status)
    except FeishuEventMappingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return JSONResponse(content={"success": True, "response": result})


@app.post("/feishu/approvals/{instance_code}/sync")
async def sync_feishu_approval(
    instance_code: str,
    current_user: dict = Depends(get_current_user),
):
    """主动查询飞书审批状态，便于本地开发环境完成闭环。"""
    try:
        instance = await get_approval_instance(instance_code)
    except FeishuApprovalError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    external_status = instance.get("status")
    if not external_status:
        raise HTTPException(status_code=502, detail="飞书没有返回审批状态。")
    try:
        result = await apply_feishu_status(instance_code, external_status)
    except FeishuEventMappingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return JSONResponse(content={"status": external_status, "response": result})
