from typing import Optional
from langchain_core.runnables import Runnable, RunnableConfig
from customer_support_chat.app.core.state import State
from pydantic import BaseModel
from customer_support_chat.app.core.settings import get_settings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

settings = get_settings()

# 初始化语言模型（供各个助手共享）
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None,
    temperature=1,
    max_tokens=settings.MAX_TOKENS,  # 限制 token 数量以控制成本
)

# 定义所有助手的基本壳
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: Optional[RunnableConfig] = None):
        while True:
            result = self.runnable.invoke(state, config)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "请给出实际有效的回复。")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# 定义 CompleteOrEscalate 工具
@tool
def CompleteOrEscalate(reason: str) -> str:
    """用于将当前任务标记为已完成，或将控制权升级回主助手的工具。

    Args:
        reason: 完成或升级的原因

    Returns:
        确认操作结果的消息
    """
    return f"任务已完成或已升级给主助手。原因：{reason}"
