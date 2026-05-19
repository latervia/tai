import inspect
import json

from langchain_core.messages import AIMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.tools.tool_registry import ToolRegistry
from app.domain.agentx.tools.tool_result import ToolResult
from app.domain.agentx.tools.web_fetch import web_fetch
from app.domain.agentx.workers.base import BaseAgent


# ---------------------------------------------------------------
# 注册（幂等）
# ---------------------------------------------------------------
if "web_fetch" not in ToolRegistry.list_all():
    ToolRegistry.register(
        "web_fetch",
        web_fetch,
        description="抓取指定 URL 的网页正文，返回 title / url / text。用于阅读搜索结果中的页面内容。",
        permissions=["read"],
    )
    ToolRegistry.grant("read_worker", ["read"])

_FETCH_TOOL_SCHEMA = {
    "name": "web_fetch",
    "description": "抓取指定 URL 的网页正文，返回 title / url / text。每次调用传入一个 URL。",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页完整 URL",
            },
        },
        "required": ["url"],
    },
}


class ReadWorker(BaseAgent):

    def __init__(self, model):
        super().__init__(model)
        self._collected: list[dict] = []

    # ---------- 元信息 ----------

    @property
    def name(self) -> str:
        return "read_worker"

    @property
    def description(self) -> str:
        return "浏览专家 — 抓取搜索结果中的网页，获取全文内容供萃取阶段使用"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位浏览专家。你的任务是根据搜索结果列表，抓取最有价值的网页全文。

            规则：
            1. 你会收到一份搜索结果列表（JSON），每项包含 title / url / snippet
            2. 调用 web_fetch 工具抓取网页正文 —— 每次传入一个 URL
            3. 优先抓取：标题与主题高度相关、snippet 信息量大的页面
            4. 跳过明显低质量的来源（内容农场、论坛水帖等）
            5. 每个 URL 只抓取一次
            6. 抓取完成后，在最终回复中列出"已抓取页面清单"

            输出格式：
            {
              "summary": "抓取概况（一句话）",
              "fetched": [
                {"title": "页面标题", "url": "https://...", "reason": "为什么选这个页面"}
              ]
            }
            """)

    @property
    def tools(self) -> list:
        return [_FETCH_TOOL_SCHEMA]

    @property
    def max_tool_rounds(self) -> int:
        return 15  # 每个 URL 一次调用，可能较多

    # ---------- 扩展点 ----------

    def _extra_messages(self, state: State) -> list:
        results = state.get("search_results")
        if not results:
            return []
        # 只传 title/url/snippet 给 LLM，不需要原始结果里的其他字段
        brief = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
            for r in results
        ]
        return [SystemMessage(
            content=f"以下是待浏览的搜索结果列表：\n\n```json\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n```\n\n"
                    f"请挑选最有价值的页面，依次调用 web_fetch 抓取全文。"
        )]

    async def _execute_tool(self, session_id: str, tool_name: str, tool_args: dict) -> ToolResult:
        result = await super()._execute_tool(session_id, tool_name, tool_args)
        if tool_name == "web_fetch" and result.success and isinstance(result.output, dict):
            self._collected.append(result.output)
        return result

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        unique = self._deduplicate(self._collected)
        result["collected_sources"] = unique
        self._collected = []
        return result

    # ---------- 工具方法 ----------

    @staticmethod
    def _deduplicate(sources: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for s in sources:
            key = s.get("url", "")
            if key and key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
