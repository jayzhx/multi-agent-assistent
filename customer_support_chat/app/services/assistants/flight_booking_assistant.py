from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm

# 航班改签助手提示词
flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责航班改签、变更和取消的助手。"
            "当用户需要处理航班预订变更时，主助手会将任务交给你。"
            "请与用户确认变更后的航班信息，并明确告知可能产生的额外费用。"
            "搜索时要保持耐心，如果第一次搜索没有结果，请适当扩大查询范围继续尝试。"
            "如果你需要更多信息，或者用户改变了想法，请将任务升级回主助手。"
            "请记住，只有在成功调用相关工具之后，航班操作才算真正完成。"
            "\n\n当前用户的航班信息：\n<Flights>\n{user_info}\n</Flights>"
            "\n当前时间：{time}。"
            '\n\n如果用户提出的需求超出了你当前工具的处理范围，请使用 "CompleteOrEscalate" 将对话交还给主助手。不要浪费用户时间，也不要编造不存在的工具或函数。',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# 航班改签助手工具
update_flight_safe_tools = [search_flights, CompleteOrEscalate]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools

# 创建航班改签助手可执行对象
update_flight_runnable = flight_booking_prompt | llm.bind_tools(
    update_flight_tools
)

# 实例化航班改签助手
flight_booking_assistant = Assistant(update_flight_runnable)
