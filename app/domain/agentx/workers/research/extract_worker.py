import inspect
import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.workers.base import BaseAgent


class ExtractWorker(BaseAgent):
    """萃取专家 — 从原始网页内容中提炼结构化素材，按章节归类。

    不依赖外部工具，纯 LLM 推理步骤。
    """

    def __init__(self, model):
        super().__init__(model)

    # ---------- 元信息 ----------

    @property
    def name(self) -> str:
        return "extract_worker"

    @property
    def description(self) -> str:
        return "萃取专家 — 从抓取的网页原文中提炼关键事实、引用和参考，按章节归类存储"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位信息萃取专家。你的任务是从一组网页原文中提取关键信息，并按研究大纲的章节结构归类。

            规则：
            1. 你会收到：研究大纲 + 已抓取的网页内容列表
            2. 逐篇阅读网页内容，提取与大纲章节相关的关键事实
            3. 每条关键事实注明来源 URL
            4. 保留有价值的原文引用（quote）
            5. 将事实归类到对应的章节 heading 下
            6. 不要编造信息 —— 只提取原文中存在的内容
            7. 严格按以下 JSON 格式输出，不要输出任何其他文字：

            {
              "summary": "萃取概况（一句话）",
              "facts_by_section": [
                {
                  "section_heading": "1. 章节标题（需与大纲一致）",
                  "key_points": ["关键事实1", "关键事实2"],
                  "quotes": [
                    {"text": "原文引用", "source_url": "https://..."}
                  ],
                  "references": ["https://..."]
                }
              ]
            }
            """)

    # ---------- 扩展点 ----------

    def _extra_messages(self, state: State) -> list:
        plan = state.get("research_plan", {})
        sources = state.get("collected_sources", [])

        if not sources:
            return [SystemMessage(content="没有待萃取的素材。")]

        parts: list[str] = []

        if plan:
            parts.append(f"## 研究大纲\n```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")

        # 来源列表（含正文）
        sources_text = json.dumps(
            [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("text", "")}
             for s in sources],
            ensure_ascii=False,
            indent=2,
        )
        parts.append(f"## 网页内容\n```json\n{sources_text}\n```")

        return [SystemMessage(content="\n\n".join(parts) + "\n\n请按上述规则提取结构化素材。")]

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        facts = self._parse_facts(response.content)
        if facts:
            result["extracted_facts"] = facts
        return result

    # ---------- 解析 ----------

    @staticmethod
    def _extract_json(text: str) -> str | None:
        m = re.search(r"```(?:json)?\s*\n?(.+?)\n?```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\{.+", text, re.DOTALL)
        if m:
            return m.group(0).strip()
        return None

    @classmethod
    def _parse_facts(cls, content: str) -> dict | None:
        try:
            json_str = cls._extract_json(content)
            if json_str is None:
                return None
            data = json.loads(json_str)
            if "facts_by_section" in data:
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None
