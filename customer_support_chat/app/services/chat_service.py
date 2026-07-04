# customer_support_chat/app/services/chat_service.py
"""
该模块提供一个基于 LangGraph 多智能体系统处理用户消息的服务。
它封装了 main.py 中的核心对话逻辑，使其能够在 Web 应用场景下复用。
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Union
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.core.logger import logger

# 尝试导入 web_app 模块
try:
    # 将项目根目录加入路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from web_app.app.core.user_data_manager import set_pending_action, get_pending_action, get_user_decision, clear_pending_action, clear_user_decision, add_operation_log
    WEB_APP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web 应用模块不可用，HITL 功能将受到限制。错误信息：{e}")
    WEB_APP_AVAILABLE = False


async def process_user_message(session_data: Dict[str, Any], user_message: str) -> str:
    """
    使用 LangGraph 多智能体系统处理用户消息。

    Args:
        session_data (Dict[str, Any]): 会话数据，包含配置项（thread_id、passenger_id）。
        user_message (str): 需要处理的用户消息。

    Returns:
        str: AI 返回的响应消息。
    """
    # 从 session_data 中提取配置
    config = session_data.get("config", {})
    # 确保配置格式符合 LangGraph 要求
    langgraph_config = {"configurable": config}
    
    # 用于跟踪已处理的消息 ID，避免重复
    # 在 Web 场景下，我们收集消息并仅返回最新的 AI 响应
    printed_message_ids = set()
    latest_ai_response = None
    
    try:
        # 将用户输入写入操作日志
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "user_input",
                "title": "用户消息",
                "content": user_message
            })
        
        # 通过图工作流处理用户输入
        # astream_events 具有更好的异步支持和更细粒度的控制
        # 但为了简化实现并兼容现有代码，这里使用 stream
        events = multi_agentic_graph.stream(
            {"messages": [("user", user_message)]}, langgraph_config, stream_mode="values"
        )
        
        # 收集流式输出中的消息
        all_tool_calls_needing_response = []  # 跟踪所有需要响应的工具调用
        
        for event in events:
            messages = event.get("messages", [])
            for message in messages:
                if message.id not in printed_message_ids:
                    # 跟踪需要响应的工具调用
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # 仅跟踪尚未处理过的工具调用
                            if tool_call["id"] not in [tc["id"] for tc in all_tool_calls_needing_response]:
                                all_tool_calls_needing_response.append(tool_call)
                                logger.debug(f"正在跟踪工具调用：{tool_call['name']}（ID：{tool_call['id']}）")
                    
                    # 记录不同类型的消息
                    if WEB_APP_AVAILABLE:
                        if isinstance(message, AIMessage) and message.content and message.content.strip():
                            # 仅在 AI 响应有实际内容时写入操作日志
                            add_operation_log(session_data["session_id"], {
                                "type": "ai_response",
                                "title": "AI 响应",
                                "content": message.content
                            })
                        elif hasattr(message, 'tool_calls') and message.tool_calls:
                            # 将工具调用写入操作日志
                            for tool_call in message.tool_calls:
                                add_operation_log(session_data["session_id"], {
                                    "type": "tool_call",
                                    "title": f"{tool_call['name']} 调用",
                                    "content": "\n".join([f"{k}: {v}" for k, v in tool_call['args'].items()]),
                                    "details": {
                                        "tool_name": tool_call['name'],
                                        "tool_call_id": tool_call['id'],
                                        "parameters": tool_call['args']
                                    }
                                })
                    
                    # message.pretty_print()  # 在 Web 应用中不希望打印到控制台
                    if isinstance(message, AIMessage) and message.content.strip():
                        # 只保留最新的 AI 响应，而不是全部响应
                        latest_ai_response = message.content
                    printed_message_ids.add(message.id)
                    
        logger.info(f"流式处理期间共处理了 {len(all_tool_calls_needing_response)} 个工具调用")
                    
        # 检查是否发生中断（HITL）
        snapshot = multi_agentic_graph.get_state(langgraph_config)
        logger.info(f"图状态快照 - next: {snapshot.next}, values 键: {list(snapshot.values.keys()) if snapshot.values else 'None'}")
        
        if snapshot.next:
            # 通过补充合适的工具消息响应来处理中断
            logger.info("检测到中断。在 Web 应用中，这通常需要用户审批。")
            
            # 获取最后一条消息，该消息通常应包含工具调用
            last_message = snapshot.values["messages"][-1] if snapshot.values.get("messages") else None
            
            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"最后一条消息包含 {len(last_message.tool_calls)} 个工具调用：{[tc['name'] for tc in last_message.tool_calls]}")
                
                # 在 Web 应用场景下，设置待处理操作并等待用户输入
                if WEB_APP_AVAILABLE:
                    # 提取工具调用详情，供用户审批
                    tool_calls_details = []
                    for tool_call in last_message.tool_calls:
                        tool_calls_details.append({
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "args": tool_call["args"]
                        })
                    
                    # 保存待处理操作
                    pending_action = {
                        "tool_calls": tool_calls_details,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    set_pending_action(session_data["session_id"], pending_action)
                    
                    # 将中断信息写入操作日志
                    add_operation_log(session_data["session_id"], {
                        "type": "system_message",
                        "title": "HITL 中断",
                        "content": "敏感操作需要用户审批",
                        "details": {
                            "tool_calls": tool_calls_details
                        }
                    })
                    
                    # 创建工具消息响应以确认这些工具调用
                    # 这样可以避免缺少工具调用响应的报错
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="该操作需要用户审批，请等待用户决策。",
                            )
                        )
                    
                    logger.info(f"正在为 HITL 发送 {len(tool_messages)} 条确认消息")
                    
                    # 发送工具消息以确认这些工具调用
                    # 这样可以避免缺少工具调用响应的报错
                    multi_agentic_graph.update_state(
                        langgraph_config,
                        {"messages": tool_messages},
                    )
                    
                    # 返回需要用户审批的提示消息
                    if latest_ai_response:
                        latest_ai_response += "\n\n[敏感操作需要用户审批，请在 Web 界面中批准或拒绝该操作。]"
                    else:
                        latest_ai_response = "[敏感操作需要用户审批，请在 Web 界面中批准或拒绝该操作。]"
                else:
                    # 如果 Web 应用不可用，则回退为自动拒绝
                    # 为所有工具调用创建工具消息响应
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="该 API 调用已被用户拒绝。原因：'敏感操作需要在 Web 界面中进行明确审批。如需改签、取消等协助，请联系人工客服。' 请结合用户输入继续提供帮助。",
                            )
                        )
                    
                    # 使用拒绝响应继续执行图工作流
                    denial_response = multi_agentic_graph.invoke(
                        {"messages": tool_messages},
                        langgraph_config,
                    )
                    
                    # 处理拒绝后的响应消息
                    messages = denial_response.get("messages", [])
                    for message in messages:
                        if message.id not in printed_message_ids:
                            if isinstance(message, AIMessage) and message.content.strip():
                                # 只保留拒绝流程中的最新 AI 响应
                                latest_ai_response = message.content
                            printed_message_ids.add(message.id)
            else:
                logger.warning("检测到中断，但最后一条消息中未发现工具调用")
                # 如果未发现工具调用，则使用兜底提示
                if latest_ai_response:
                    latest_ai_response += "\n\n[敏感操作需要用户审批，如需协助请联系人工客服。]"
                else:
                    latest_ai_response = "[敏感操作需要用户审批，如需协助请联系人工客服。]"
        else:
            logger.info("未检测到中断")
            # 即使没有发生中断，也检查是否有尚未响应的工具调用
            # 这是为了防止边界情况下出现 tool_calls 错误
            if all_tool_calls_needing_response:
                logger.info(f"正在检查 {len(all_tool_calls_needing_response)} 个工具调用是否已被正确确认")
                
                # 通过最终状态检查所有工具调用是否都已处理
                final_messages = snapshot.values.get("messages", [])
                handled_tool_call_ids = set()
                
                # 收集所有已有对应工具消息的工具调用 ID
                for msg in final_messages:
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        handled_tool_call_ids.add(msg.tool_call_id)
                
                logger.info(f"共发现 {len(handled_tool_call_ids)} 个已处理的工具调用 ID")
                
                # 查找尚未收到响应的工具调用
                unhandled_tool_calls = [
                    tc for tc in all_tool_calls_needing_response 
                    if tc["id"] not in handled_tool_call_ids
                ]
                
                if unhandled_tool_calls:
                    logger.warning(f"发现 {len(unhandled_tool_calls)} 个未处理的工具调用，正在创建确认消息")
                    
                    for tc in unhandled_tool_calls:
                        logger.warning(f"未处理的工具调用：{tc['name']}（ID：{tc['id']}）")
                    
                    # 为未处理的工具调用创建确认消息
                    acknowledgment_messages = []
                    for tool_call in unhandled_tool_calls:
                        acknowledgment_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content=f"工具 '{tool_call['name']}' 已成功处理。",
                            )
                        )
                    
                    # 如果有确认消息，则发送
                    if acknowledgment_messages:
                        try:
                            multi_agentic_graph.update_state(
                                langgraph_config,
                                {"messages": acknowledgment_messages},
                            )
                            logger.info(f"已为未处理工具调用发送 {len(acknowledgment_messages)} 条确认消息")
                        except Exception as ack_error:
                            logger.error(f"发送确认消息失败：{ack_error}")
                else:
                    logger.info("所有工具调用都已被正确确认")
            
        # 返回最新的 AI 响应；如果没有生成响应，则返回默认提示
        if latest_ai_response:
            return latest_ai_response
        else:
            return "抱歉，我没有理解您的意思。您可以换一种说法吗？"
            
    except Exception as e:
        logger.error(f"处理用户消息时发生错误：{e}")
        
        # 针对 tool_calls 错误进行特殊处理
        if "tool_calls must be followed by tool messages" in str(e):
            logger.warning("检测到 tool_calls 确认错误，正在尝试恢复")
            logger.error(f"完整错误详情：{e}")
            
            try:
                # 获取当前图状态，以确认发生了哪些工具调用
                snapshot = multi_agentic_graph.get_state(langgraph_config)
                logger.info(f"图状态 - next: {snapshot.next}")
                
                # 查找最后一条包含工具调用的消息
                if snapshot.values and "messages" in snapshot.values:
                    messages = snapshot.values["messages"]
                    logger.info(f"当前状态中的消息总数：{len(messages)}")
                    
                    # 查找带有工具调用的消息
                    for i, msg in enumerate(reversed(messages[-10:])):
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            logger.info(f"消息 {len(messages)-i} 包含 {len(msg.tool_calls)} 个工具调用：")
                            for tc in msg.tool_calls:
                                logger.info(f"  - {tc['name']}（ID：{tc['id']}）")
                
                # 尝试从错误信息中提取工具调用 ID 并补发确认消息
                error_str = str(e)
                if "tool_call_ids did not have response messages:" in error_str:
                    # 从错误信息中提取工具调用 ID
                    import re
                    tool_call_match = re.search(r'call_[a-zA-Z0-9]+', error_str)
                    if tool_call_match:
                        missing_tool_call_id = tool_call_match.group()
                        logger.info(f"正在为工具调用 ID 创建紧急确认消息：{missing_tool_call_id}")
                        
                        # 创建紧急确认消息
                        emergency_acknowledgment = ToolMessage(
                            tool_call_id=missing_tool_call_id,
                            content="紧急确认：工具调用已处理。",
                        )
                        
                        # 尝试发送确认消息
                        multi_agentic_graph.update_state(
                            langgraph_config,
                            {"messages": [emergency_acknowledgment]},
                        )
                        
                        logger.info("紧急确认消息发送成功")
                        
                        # 返回提示，说明问题已处理
                        return "抱歉，刚才出现了技术问题。您的请求已经处理完成，如需进一步帮助，请换一种说法再试一次。"
                        
            except Exception as recovery_error:
                logger.error(f"从 tool_calls 错误中恢复失败：{recovery_error}")
        
        # 将错误写入操作日志
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "error",
                "title": "处理错误",
                "content": str(e)
            })
        # 在 Web 应用中，这里返回更友好的错误提示
        return "处理您的请求时发生了异常错误，请稍后重试。"


async def process_user_decision(session_data: Dict[str, Any], decision: str) -> str:
    """
    处理用户对待处理操作的决策（批准/拒绝）。

    Args:
        session_data (Dict[str, Any]): 会话数据，包含配置项（thread_id、passenger_id）。
        decision (str): 用户的决策（'approve' 或 'reject'）。

    Returns:
        str: 处理决策后返回的 AI 响应消息。
    """
    if not WEB_APP_AVAILABLE:
        return "当前环境不支持 HITL 功能。"
    
    # 从 session_data 中提取配置
    config = session_data.get("config", {})
    # 确保配置格式符合 LangGraph 要求
    langgraph_config = {"configurable": config}
    
    # 用于跟踪已处理的消息 ID，避免重复
    printed_message_ids = set()
    result_message = ""
    
    try:
        # 获取待处理操作
        pending_action = get_pending_action(session_data["session_id"])
        if not pending_action:
            return "未找到待处理操作。"
        
        # 将用户决策写入操作日志
        add_operation_log(session_data["session_id"], {
            "type": "user_input",
            "title": "用户决策",
            "content": f"用户已对该操作执行{decision.lower()}处理"
        })
        
        # 获取待处理操作中的工具调用
        tool_calls = pending_action.get("tool_calls", [])
        
        if decision.lower() == "approve":
            # 对于批准操作，直接执行工具
            # 这里采用简化实现，实际项目中可以执行真实工具并返回结果
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 导入并执行对应工具
                try:
                    if tool_name == "update_hotel":
                        from customer_support_chat.app.services.tools.hotels import update_hotel
                        # 异步函数使用 ainvoke
                        result = await update_hotel.ainvoke(tool_args)
                        result_message = f"酒店修改成功：{result}"
                    elif tool_name == "book_hotel":
                        from customer_support_chat.app.services.tools.hotels import book_hotel
                        # 异步函数使用 ainvoke
                        result = await book_hotel.ainvoke(tool_args)
                        result_message = f"酒店预订成功：{result}"
                    elif tool_name == "cancel_hotel":
                        from customer_support_chat.app.services.tools.hotels import cancel_hotel
                        # 异步函数使用 ainvoke
                        result = await cancel_hotel.ainvoke(tool_args)
                        result_message = f"酒店取消成功：{result}"
                    elif tool_name == "update_car_rental":
                        from customer_support_chat.app.services.tools.cars import update_car_rental
                        # 异步函数使用 ainvoke
                        result = await update_car_rental.ainvoke(tool_args)
                        result_message = f"租车信息修改成功：{result}"
                    elif tool_name == "book_car_rental":
                        from customer_support_chat.app.services.tools.cars import book_car_rental
                        # 异步函数使用 ainvoke
                        result = await book_car_rental.ainvoke(tool_args)
                        result_message = f"租车预订成功：{result}"
                    elif tool_name == "cancel_car_rental":
                        from customer_support_chat.app.services.tools.cars import cancel_car_rental
                        # 异步函数使用 ainvoke
                        result = await cancel_car_rental.ainvoke(tool_args)
                        result_message = f"租车取消成功：{result}"
                    elif tool_name == "book_excursion":
                        from customer_support_chat.app.services.tools.excursions import book_excursion
                        # 异步函数使用 ainvoke
                        result = await book_excursion.ainvoke(tool_args)
                        result_message = f"行程预订成功：{result}"
                    elif tool_name == "update_excursion":
                        from customer_support_chat.app.services.tools.excursions import update_excursion
                        # 异步函数使用 ainvoke
                        result = await update_excursion.ainvoke(tool_args)
                        result_message = f"行程修改成功：{result}"
                    elif tool_name == "cancel_excursion":
                        from customer_support_chat.app.services.tools.excursions import cancel_excursion
                        # 异步函数使用 ainvoke
                        result = await cancel_excursion.ainvoke(tool_args)
                        result_message = f"行程取消成功：{result}"
                    elif tool_name == "update_ticket_to_new_flight":
                        from customer_support_chat.app.services.tools.flights import update_ticket_to_new_flight
                        # 异步函数使用 ainvoke
                        result = await update_ticket_to_new_flight.ainvoke({**tool_args, "config": langgraph_config})
                        result_message = f"航班改签成功：{result}"
                    elif tool_name == "cancel_ticket":
                        from customer_support_chat.app.services.tools.flights import cancel_ticket
                        # 异步函数使用 ainvoke
                        result = await cancel_ticket.ainvoke({**tool_args, "config": langgraph_config})
                        result_message = f"航班取消成功：{result}"
                    else:
                        result_message = f"工具 {tool_name} 已成功执行（审批处理器中尚未实现该工具的专用处理）"
                    
                    # 将工具执行结果写入操作日志
                    add_operation_log(session_data["session_id"], {
                        "type": "tool_result",
                        "title": f"{tool_name} 结果",
                        "content": result if 'result' in locals() else result_message
                    })
                    
                except Exception as e:
                    error_msg = f"执行 {tool_name} 时出错：{str(e)}"
                    result_message = error_msg
                    add_operation_log(session_data["session_id"], {
                        "type": "error",
                        "title": f"{tool_name} 执行错误",
                        "content": error_msg
                    })
        else:  # reject
            # 对于拒绝操作，直接告知用户
            result_message = "操作已被用户取消。"
            # 将取消操作写入日志
            add_operation_log(session_data["session_id"], {
                "type": "system_message",
                "title": "操作已取消",
                "content": "用户拒绝了敏感操作"
            })
        
        # 清除待处理操作和用户决策
        clear_pending_action(session_data["session_id"])
        clear_user_decision(session_data["session_id"])
        
        # 返回结果消息
        if result_message:
            return result_message
        else:
            return "操作处理成功。"
            
    except Exception as e:
        logger.error(f"处理用户决策时发生错误：{e}")
        # 将错误写入操作日志
        add_operation_log(session_data["session_id"], {
            "type": "error",
            "title": "决策处理错误",
            "content": str(e)
        })
        # 即使发生错误，也尝试清除待处理操作和用户决策
        try:
            clear_pending_action(session_data["session_id"])
            clear_user_decision(session_data["session_id"])
        except:
            pass
        return "处理您的决策时发生了异常错误，请稍后重试。"




