from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.blog import search_blog_posts
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm, CompleteOrEscalate
from pydantic import BaseModel, Field

# 定义博客搜索任务委派工具
class ToBlogSearch(BaseModel):
    """将任务转交给专门处理博客文章搜索的助手。"""
    keyword: str = Field(description="用于搜索博客文章的关键词。")

# 博客搜索助手提示词
blog_search_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名专门负责博客文章搜索的助手。"
            "你的主要职责是根据用户提供的关键词，调用 search_blog_posts 工具查找相关文章。"
            "请用清晰、易读的方式展示搜索结果，包含标题、摘要和链接。"
            "如果用户的需求与博客搜索无关，请使用 CompleteOrEscalate 工具将控制权交还给主助手。"
            "当前时间：{time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# 博客搜索助手工具
blog_search_assistant_tools = [
    search_blog_posts,
    CompleteOrEscalate,
]

# 创建博客搜索助手可执行对象
blog_search_assistant_runnable = blog_search_assistant_prompt | llm.bind_tools(blog_search_assistant_tools)

# 实例化博客搜索助手
blog_search_assistant = Assistant(blog_search_assistant_runnable)
