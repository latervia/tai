import inspect
import json

from langchain_core.messages import AIMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.tools.tool_registry import ToolRegistry
from app.domain.agentx.tools.tool_result import ToolResult
from app.domain.agentx.tools.web_search import web_search
from app.domain.agentx.workers.base import BaseAgent

# ---------------------------------------------------------------
# 确保搜索工具已注册（幂等）
# ---------------------------------------------------------------
if "web_search" not in ToolRegistry.list_all():
    ToolRegistry.register(
        "web_search",
        web_search,
        description="搜索互联网获取资料，传入 query 关键词，返回包含 title / url / snippet 的搜索结果列表。",
        permissions=["search"],
    )
    ToolRegistry.grant("search_worker", ["search"])

# LangChain bind_tools 所需的 schema
_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": "搜索互联网获取资料。传入 query 关键词，返回 title / url / snippet 列表。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，中英文均可。每个查询应具体、有针对性。",
            },
        },
        "required": ["query"],
    },
}


class SearchWorker(BaseAgent):

    def __init__(self, model):
        super().__init__(model)
        self._collected: list[dict] = []

    # ---------- Agent 元信息 ----------

    @property
    def name(self) -> str:
        return "search_worker"

    @property
    def description(self) -> str:
        return "搜索专家 — 根据研究大纲执行搜索，收集各章节所需的参考资料"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位搜索专家。你的任务是根据研究大纲，为每个章节执行搜索，收集资料。

            规则：
            1. 你会收到一份 JSON 格式的研究大纲，每个章节包含 search_queries
            2. 调用 web_search 工具执行搜索 —— 每轮可以批量调用多个查询
            3. 优先搜索大纲中明确列出的 search_queries
            4. 如果某个章节的 search_queries 不足，根据你的知识补充 1-2 个查询
            5. 搜索完成后，在最终回复中列出"按章节整理的搜索结果摘要"

            输出格式：
            {
              "summary": "搜索执行概况（一句话）",
              "by_section": [
                {
                  "section_heading": "章节标题",
                  "queries_run": ["查询1", "查询2"],
                  "result_count": 3
                }
              ]
            }
            """)

    @property
    def tools(self) -> list:
        return [_SEARCH_TOOL_SCHEMA]

    @property
    def max_tool_rounds(self) -> int:
        return 12

    # ---------- 三个扩展点 ----------

    def _extra_messages(self, state: State) -> list:
        """注入研究大纲"""
        plan = state.get("research_plan")
        if not plan:
            return []
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
        return [SystemMessage(
            content=f"以下是你需要执行搜索的研究大纲：\n\n```json\n{plan_text}\n```\n\n"
                    f"请依次为每个章节的 search_queries 调用 web_search 工具。"
        )]

    async def _execute_tool(self, session_id: str, tool_name: str, tool_args: dict) -> ToolResult:
        """执行工具并收集搜索结果"""
        result = await super()._execute_tool(session_id, tool_name, tool_args)
        if tool_name == "web_search" and result.success and isinstance(result.output, list):
            self._collected.extend(result.output)
        return result

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        result["search_results"] = self._deduplicate(self._collected)
        self._collected = []  # 清空，为下一次调用做准备
        return result

    # 工具方法
    @staticmethod
    def _deduplicate(results: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for r in results:
            key = r.get("url", "")
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
