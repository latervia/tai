import inspect

from app.domain.agentx.workers.base import BaseAgent


class ChatWorker(BaseAgent):

    @property
    def name(self) -> str:
        return "chat_worker"

    @property
    def description(self) -> str:
        return "通用对话 Agent — 处理闲聊、简单问答、无需专业检索的请求"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc(
            """
            你是一个友好、专业的 AI 助手。
            核心规则：
            1. 用简洁清晰的中文回复
            2. 不确定的答案直接说明，不要编造
            3. 保持对话自然流畅
            """
        )
