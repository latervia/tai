"""对话 Agent — 处理通用对话和简单问答"""
from langchain_core.language_models import BaseChatModel

from app.domain.agent.base import BaseAgent


class ChatAgent(BaseAgent):
    """通用对话 Agent"""

    # 类级元数据 — 无需实例化即可被 Supervisor 查询
    AGENT_NAME = "chat_agent"
    AGENT_DESCRIPTION = "通用对话 Agent — 处理闲聊、简单问答、无需专业检索的请求"

    def __init__(self, model: BaseChatModel):
        super().__init__(model)

    @property
    def name(self) -> str:
        return self.AGENT_NAME

    @property
    def description(self) -> str:
        return self.AGENT_DESCRIPTION

    @property
    def system_prompt(self) -> str:
        return (
            "你是一个友好、专业的 AI 助手。\n"
            "核心规则：\n"
            "1. 用简洁清晰的中文回复\n"
            "2. 不确定的答案直接说明，不要编造\n"
            "3. 保持对话自然流畅"
        )
