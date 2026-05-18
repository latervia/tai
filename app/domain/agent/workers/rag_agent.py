"""RAG Agent — 知识检索增强生成"""
from langchain_core.language_models import BaseChatModel

from app.domain.agent.base import BaseAgent


class RAGAgent(BaseAgent):
    """知识检索 Agent"""

    AGENT_NAME = "rag_agent"
    AGENT_DESCRIPTION = "知识检索 Agent — 负责从知识库检索文档并基于检索结果回答"

    def __init__(self, model: BaseChatModel, search_tool=None):
        super().__init__(model)
        self._search_tool = search_tool

    @property
    def name(self) -> str:
        return self.AGENT_NAME

    @property
    def description(self) -> str:
        return self.AGENT_DESCRIPTION

    @property
    def system_prompt(self) -> str:
        return (
            "你是一个知识检索助手。\n"
            "核心规则：\n"
            "1. 先使用 search 工具检索相关文档\n"
            "2. 基于检索到的文档内容回答用户问题\n"
            "3. 如果没有检索到相关内容，明确告知用户\n"
            "4. 引用文档内容时注明来源"
        )

    @property
    def tools(self) -> list:
        tools = []
        if self._search_tool:
            tools.append(self._search_tool)
        return tools
