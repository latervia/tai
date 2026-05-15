"""BaseAgent — 所有 Agent 的基类，封装 LLM 调用 + 工具使用循环"""
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

from app.agent.state import MultiAgentState
from app.agent.tools.registry import ToolRegistry
from app.agent.types import AgentThought, ActionResult, ThoughtType, AgentStatus
from app.core.logger import logger


class BaseAgent(ABC):
    """Agent 基类

    每个 Agent 是一个可被 LangGraph 当作 node 使用的可调用对象。
    子类需实现 system_prompt 来定义角色，以及可选的 _post_process 钩子。
    """

    # ── 子类必须定义 ──────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 唯一标识"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """自然语言描述，供 Supervisor 路由时参考"""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """该 Agent 的系统提示词"""
        ...

    # ── 子类可选覆盖 ──────────────────────────────────────

    @property
    def tools(self) -> list:
        """该 Agent 可用的 LangChain tool 列表"""
        return []

    @property
    def max_tool_rounds(self) -> int:
        """工具调用最大轮数，防止无限循环"""
        return 5

    @property
    def temperature(self) -> float:
        return 0.7

    # ── 构造 ──────────────────────────────────────────────

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.status = AgentStatus.IDLE
        # 绑定工具的 LLM（延迟初始化，因为 tools 是 property）
        self._bound_llm: Optional[BaseChatModel] = None

    def _get_llm(self) -> BaseChatModel:
        """获取绑定了工具的 LLM 实例"""
        if self._bound_llm is None:
            llm = self.model
            if self.tools:
                llm = llm.bind_tools(self.tools)
            self._bound_llm = llm
        return self._bound_llm

    # ── LangGraph node 接口 ───────────────────────────────

    async def __call__(self, state: MultiAgentState) -> dict:
        """LangGraph 节点入口 — 接收 State，返回部分 State 更新"""
        self.status = AgentStatus.THINKING
        logger.info(f"[{self.name}] 开始执行")

        try:
            # 构建消息列表：system prompt + 历史消息
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

            # 工具调用循环
            for _ in range(self.max_tool_rounds):
                llm = self._get_llm()
                response: AIMessage = await llm.ainvoke(messages)

                # 没有工具调用 → 直接返回回复
                if not response.tool_calls:
                    self.status = AgentStatus.DONE
                    return self._build_response(response)

                # 执行工具调用
                self.status = AgentStatus.ACTING
                tool_results = []
                for tc in response.tool_calls:
                    result = await self._execute_tool(tc["name"], tc["args"])
                    tool_results.append(result)

                # 将 LLM 响应和工具结果追加到消息列表，继续循环
                messages.append(response)
                for tr in tool_results:
                    messages.append(ToolMessage(
                        content=str(tr.output) if tr.success else f"Error: {tr.error}",
                        tool_call_id=tr.tool_name or "",
                    ))

            # 达到最大轮数，强制生成最终回复
            self.status = AgentStatus.DONE
            messages.append(SystemMessage(content="已达到工具调用上限，请基于已有信息直接回答用户。"))
            response = await self.model.ainvoke(messages)
            return self._build_response(response)

        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"[{self.name}] 执行失败: {e}")
            return {
                "messages": [AIMessage(content=f"抱歉，处理您的请求时出错了：{e}")],
                "agent_outputs": {self.name: {"error": str(e)}},
            }

    # ── 内部方法 ──────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> ActionResult:
        """执行单个工具调用"""
        import time
        start = time.time()

        tool_def = ToolRegistry.get(tool_name)
        if tool_def is None:
            # 尝试 LangChain 绑定的工具（如 search）
            return ActionResult(
                success=False,
                error=f"工具 '{tool_name}' 未在注册表中找到",
                tool_name=tool_name,
            )

        try:
            output = tool_def.fn(**tool_args)
            if hasattr(output, "__await__"):
                output = await output
            return ActionResult(
                success=True,
                output=output,
                tool_name=tool_name,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                tool_name=tool_name,
                duration_ms=(time.time() - start) * 1000,
            )

    def _build_response(self, response: AIMessage) -> dict:
        """构建 State 更新"""
        return {
            "messages": [response],
            "agent_outputs": {
                self.name: {"status": self.status.value, "content": response.content}
            },
        }

    def should_handle(self, user_message: str) -> bool:
        """简单的关键词匹配 — 子类可覆盖为 LLM 判断"""
        return True
