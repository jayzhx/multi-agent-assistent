from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.woocommerce import (
    search_products,
    search_orders,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm, CompleteOrEscalate
from customer_support_chat.app.core.logger import logger
from pydantic import BaseModel, Field

# 定义 WooCommerce 任务委派工具
class ToWooCommerceProducts(BaseModel):
    """将任务转交给专门处理商品搜索的助手。"""
    query: str = Field(description="商品搜索关键词，例如商品名称或分类。")

class ToWooCommerceOrders(BaseModel):
    """将任务转交给专门处理订单搜索的助手。"""
    search_type: str = Field(description="搜索类型，必须是 'email'、'name' 或 'id' 之一。")
    search_value: str = Field(description="搜索值。按邮箱搜索时填写客户邮箱，按姓名搜索时填写客户全名，按订单号搜索时填写订单 ID。")

# WooCommerce 助手提示词
woocommerce_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责 WooCommerce 商品与订单查询的助手。"
            "你的主要职责是使用现有工具查询商品和订单信息。"
            "当用户提出商品搜索需求时，请直接调用 search_products 工具。"
            "当用户提出订单搜索需求时，你必须先要求用户提供邮箱地址或姓名中的至少一种身份校验信息，然后才能查询订单。"
            "在没有有效校验信息（邮箱或姓名）的情况下，绝不能直接查询订单。"
            "如果用户只说“查找订单”之类的话，而没有提供校验信息，请礼貌地让其补充邮箱地址或姓名。"
            "如果用户提供邮箱，请使用 search_orders 工具，并传入 search_type='email' 和对应邮箱。"
            "如果用户提供姓名，请使用 search_orders 工具，并传入 search_type='name' 和对应姓名。"
            "如果用户提供订单号，请使用 search_orders 工具，并传入 search_type='id' 和对应订单号。"
            "如果查询没有结果，请明确告知未找到匹配内容，并建议用户换一种查询方式。"
            "如果工具调用因为超时或连接异常失败，请告知用户服务器可能繁忙，建议稍后重试。"
            "请始终基于工具结果，用清晰、简洁的方式向用户反馈信息。"
            "如果用户需求超出商品或订单查询范围，请使用 CompleteOrEscalate 工具将控制权交还给主助手。"
            "当前时间：{time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# WooCommerce 助手工具
woocommerce_assistant_tools = [
    search_products,
    search_orders,
    CompleteOrEscalate,
]

# 创建 WooCommerce 助手可执行对象
woocommerce_assistant_runnable = woocommerce_assistant_prompt | llm.bind_tools(woocommerce_assistant_tools)

# 实例化 WooCommerce 助手
woocommerce_assistant = Assistant(woocommerce_assistant_runnable)
