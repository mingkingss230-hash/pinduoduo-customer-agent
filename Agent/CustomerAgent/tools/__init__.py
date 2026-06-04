"""
工具模块

导入所有工具以触发 @agent_tool 装饰器注册。
"""
from Agent.CustomerAgent.tools import move_conversation
from Agent.CustomerAgent.tools import search_knowledge
from Agent.CustomerAgent.tools import send_product_card

__all__ = [
    "move_conversation",
    "search_knowledge",
    "send_product_card",
]
