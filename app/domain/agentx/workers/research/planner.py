import inspect
import json
import re

from langchain_core.messages import AIMessage

from app.domain.agentx.workers.base import BaseAgent


class ResearchSection(dict):
    """研究报告的单个章节

    用 dict 子类而非 TypedDict，便于 LangGraph state 序列化。
    字段约定：
      heading: str          — 章节标题
      description: str      — 本节要覆盖的内容说明
      subsections: list     — 子章节（递归同构）
      search_queries: list  — 建议搜索关键词，供 Search Worker 使用
    """


class Planner(BaseAgent):

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return "研究规划师 — 分析用户的研究主题，制定结构化的长篇报告大纲"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位资深研究规划师。你的唯一任务是根据用户的研究主题，生成一份结构化的报告大纲。

            规则：
            1. 分析用户主题，拆解为逻辑清晰、层层递进的章节结构
            2. 每个章节包含：
               - heading: 章节标题
               - description: 本章应覆盖的核心内容（1-2 句话）
               - search_queries: 2-4 个可用于搜索引擎的具体查询词
            3. 嵌套深度不超过 3 层（章 → 节 → 小节）
            4. 总章节数控制在 5-12 个为宜
            5. 严格按以下 JSON 格式输出，不要输出任何其他文字：

            {
              "title": "报告标题",
              "sections": [
                {
                  "heading": "1. 章节标题",
                  "description": "本章覆盖的内容说明",
                  "search_queries": ["关键词1", "关键词2"],
                  "subsections": [
                    {
                      "heading": "1.1 小节标题",
                      "description": "小节内容说明",
                      "search_queries": ["子关键词1"]
                    }
                  ]
                }
              ]
            }
            """)

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        plan = self._parse_plan(response.content)
        if plan:
            result["research_plan"] = plan
        return result

    # 私有方法

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """从 LLM 输出中提取 JSON 字符串"""
        # 优先匹配 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*\n?(.+?)\n?```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 否则尝试匹配首个 { ... } 块
        m = re.search(r"\{.+", text, re.DOTALL)
        if m:
            return m.group(0).strip()
        return None

    @classmethod
    def _parse_plan(cls, content: str) -> dict | None:
        try:
            json_str = cls._extract_json(content)
            if json_str is None:
                return None
            plan = json.loads(json_str)
            if "title" in plan and "sections" in plan:
                return plan
            return None
        except (json.JSONDecodeError, TypeError):
            return None
