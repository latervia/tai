import time
from abc import ABC, abstractmethod
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from app.domain.agentx.human.risk_level import RiskLevel
from app.domain.agentx.states.state import State
from app.domain.agentx.tools.tool_registry import ToolRegistry
from app.domain.agentx.tools.tool_result import ToolResult
from app.shared import logger


class AgentStatus(str, Enum):
    IDLE = "idle"  # 空闲，未开始执行任务
    THINKING = "thinking"  # 正在思考
    ACTING = "acting"  # 正在执行操作
    DONE = "done"  # 已完成任务
    FAILED = "failed"  # 执行任务失败


"""
BaseAgent 定义了多智能体系统中各 Agent 的基础结构和执行流程
"""


class BaseAgent(ABC):

    # 子类必须定义
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    # 子类可选覆盖
    @property
    def tools(self) -> list:
        return []

    @property
    def max_tool_rounds(self) -> int:
        return 5

    def _extra_messages(self, state: State) -> list:
        """子类可覆盖此方法，向 LLM 消息列表中注入额外上下文（如研究大纲）"""
        return []

    # 构造
    def __init__(self, model: BaseChatModel):
        self.model = model  # 具体类型由子类定义
        self.status = AgentStatus.IDLE  # 当前状态
        self._bound_llm = None  # 绑定工具后的 LLM 实例, 延迟初始化

    def _get_llm(self) -> BaseChatModel:
        """获取绑定工具后的 LLM 实例，支持延迟绑定"""
        if self._bound_llm is None:
            llm = self.model
            if self.tools:
                llm = llm.bind_tools(self.tools)
            self._bound_llm = llm
        return self._bound_llm

    # LangGraph node 接口
    async def __call__(self, state: State):
        """执行 Agent 任务的入口，接受多智能体状态作为输入"""
        self.status = AgentStatus.THINKING
        session_id = state.get("session_id", "unknown")
        logger.info(f"[{self.name}] 开始执行")

        try:
            messages = [SystemMessage(content=self.system_prompt)] + self._extra_messages(state) + state["messages"]

            for _ in range(self.max_tool_rounds):  # 工具调用轮数控制
                llm = self._get_llm()
                response: AIMessage = await llm.ainvoke(messages)

                if not response.tool_calls:
                    self.status = AgentStatus.DONE
                    return self._build_response(response)

                # 执行工具调用（含审批检查）
                self.status = AgentStatus.ACTING
                tool_results = []
                needs_approval = False

                for tc in response.tool_calls:
                    result = await self._execute_tool(session_id, tc["name"], tc["args"])
                    tool_results.append(result)
                    if result.needs_approval:
                        needs_approval = True

                # 有需要审批的操作 → 挂起，返回审批请求给上层
                if needs_approval:
                    pending = [
                        {"tool": r.tool_name, "args": r.output}
                        for r in tool_results if r.needs_approval
                    ]
                    logger.info(f"[{self.name}] 需要审批: {pending}")
                    return {
                        "pending_approval": pending,
                        "current_agent": self.name,
                    }

                # 追加 LLM 响应和工具结果到消息列表
                messages.append(response)
                for tr in tool_results:
                    messages.append(ToolMessage(
                        content=str(tr.output) if tr.success else f"Error: {tr.error}",
                        tool_call_id=tr.tool_name or "",
                    ))

            # 达到最大轮数 → 强制生成最终回复
            self.status = AgentStatus.DONE
            messages.append(SystemMessage(content="已达到工具调用上限，请基于已有信息直接回答用户。"))
            response = await self.model.ainvoke(messages)
            return self._build_response(response)

        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"[{self.name}] 执行失败: {e}")
            from app.deps import get_trace_collector
            get_trace_collector().error(session_id=session_id, agent=self.name, error=str(e))
            return {
                "messages": [AIMessage(content=f"抱歉，处理您的请求时出错了：{e}")],
                "agent_outputs": {self.name: {"error": str(e)}},
            }

    async def _execute_tool(self, session_id: str, tool_name: str, tool_args: dict) -> ToolResult:
        start = time.time()

        tool_def = ToolRegistry.get(tool_name)
        if tool_def is None:
            return ToolResult(success=False, error=f"工具 '{tool_name}' 未注册", tool_name=tool_name)

        # 审批检查：risk_level >= HIGH 需要人类确认
        if tool_def.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            from app.deps import get_approval_manager
            am = get_approval_manager()
            pending = am.list_pending()
            # 检查是否已有该工具的待审批请求
            already_pending = any(
                p.tool_name == tool_name and p.status == "pending"
                for p in pending
            )
            if not already_pending:
                # 创建审批请求，不执行
                import uuid
                rid = str(uuid.uuid4())
                am.request(rid, tool_name, tool_args, f"Agent {self.name} 请求执行 {tool_name}", tool_def.risk_level)
                return ToolResult(
                    success=False,
                    needs_approval=True,
                    output={"request_id": rid, "tool": tool_name, "args": tool_args},
                    tool_name=tool_name,
                )

        # 执行工具
        try:
            output = tool_def.fn(**tool_args)
            if hasattr(output, "__await__"):
                output = await output
            duration = (time.time() - start) * 1000

            # tracing
            from app.deps import get_trace_collector
            get_trace_collector().tool_call(
                session_id=session_id, agent=self.name,
                tool_name=tool_name, tool_args=tool_args,
                success=True, duration_ms=duration,
            )

            return ToolResult(success=True, output=output, tool_name=tool_name, duration_ms=duration)
        except Exception as e:
            duration = (time.time() - start) * 1000

            from app.deps import get_trace_collector
            get_trace_collector().tool_call(
                session_id=session_id, agent=self.name,
                tool_name=tool_name, tool_args=tool_args,
                success=False, duration_ms=duration, error=str(e),
            )

            return ToolResult(success=False, error=str(e), tool_name=tool_name, duration_ms=duration)

    def _build_response(self, response: AIMessage) -> dict:
        return {
            "messages": [response],
            "agent_outputs": {
                self.name: {"status": self.status.value, "content": response.content}
            },
        }
