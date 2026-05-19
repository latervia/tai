import inspect
import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.workers.base import BaseAgent


class ReviewWorker(BaseAgent):
    """评审员 — 审查初稿质量，裁决通过或退回补充。

    不依赖外部工具，纯 LLM 推理步骤。
    产出结构化 verdict，供 LangGraph 条件边路由。
    """

    def __init__(self, model):
        super().__init__(model)

    # ---------- 元信息 ----------

    @property
    def name(self) -> str:
        return "review_worker"

    @property
    def description(self) -> str:
        return "评审员 — 审查研究报告初稿，裁决通过或退回补充搜索"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位严格的学术评审员。你的任务是审查研究报告初稿，给出通过或退回的裁决。

            评审维度：
            1. 完整性 — 大纲的所有章节是否都有实质内容（非 GAP 标注）？
            2. 事实性 — 关键论断是否有素材支撑（而非凭空编造）？
            3. 逻辑性 — 章节之间、段落之间的论证链条是否严密？
            4. 引用质量 — 来源标注是否充分、可追溯？

            裁决标准：
            - **pass**：四项维度均无重大问题，可直接交付
            - **revise**：存在重大缺口或事实性错误，需要补充搜索和重写

            严格按以下 JSON 格式输出，不要输出任何其他文字：

            {
              "verdict": "pass",
              "score": 8,
              "summary": "评审总结（1-2句话）",
              "issues": [
                {
                  "section": "受影响的章节标题",
                  "severity": "major",
                  "type": "completeness|factuality|logic|citation",
                  "description": "具体问题描述"
                }
              ],
              "strengths": ["做得好的方面1", "做得好的方面2"],
              "suggestions": ["改进建议1", "改进建议2"]
            }

            如果 verdict 为 "revise"，suggestions 中必须包含具体的补充搜索方向。
            """)

    # ---------- 扩展点 ----------

    def _extra_messages(self, state: State) -> list:
        draft = state.get("draft_report", {})
        plan = state.get("research_plan", {})
        facts = state.get("extracted_facts", {})

        parts: list[str] = []

        if plan:
            parts.append(f"## 研究大纲\n```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")

        parts.append(f"## 初稿\n```json\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n```")

        if facts:
            parts.append(f"## 萃取素材（用于事实核查）\n```json\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n```")

        return [SystemMessage(content="\n\n".join(parts) + "\n\n请按上述标准评审初稿。")]

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        review = self._parse_review(response.content)
        if review:
            result["review_result"] = review
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
    def _parse_review(cls, content: str) -> dict | None:
        try:
            json_str = cls._extract_json(content)
            if json_str is None:
                return None
            data = json.loads(json_str)
            if "verdict" in data and data["verdict"] in ("pass", "revise"):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None
