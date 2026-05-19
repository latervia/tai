import inspect
import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.workers.base import BaseAgent


class WriteWorker(BaseAgent):
    """撰稿人 — 根据研究大纲和萃取素材撰写长篇报告初稿。

    不依赖外部工具，纯 LLM 推理步骤。
    """

    def __init__(self, model):
        super().__init__(model)

    # ---------- 元信息 ----------

    @property
    def name(self) -> str:
        return "write_worker"

    @property
    def description(self) -> str:
        return "撰稿人 — 根据结构化素材撰写长篇研究报告初稿"

    @property
    def system_prompt(self) -> str:
        return inspect.cleandoc("""
            你是一位资深研究报告撰稿人。你的任务是根据研究大纲和萃取素材，撰写一份专业的长篇报告初稿。

            规则：
            1. 严格遵循大纲的章节结构，逐节撰写
            2. 使用萃取素材中的事实和引用作为论据
            3. 写作风格：专业、客观、逻辑严密
            4. 在每个引用事实后标注来源 URL，格式为 `[来源](URL)`
            5. 如果某个章节的素材不足，在该章节末尾标注 `<!-- GAP: 缺少关于 XXX 的数据 -->`
            6. 不要编造素材中不存在的事实
            7. 严格按以下 JSON 格式输出，不要输出任何其他文字：

            {
              "title": "报告标题",
              "content": "# 报告标题\\n\\n## 1. 章节标题\\n\\n正文段落，引用事实时标注来源...\\n\\n## 2. ...",
              "citations": [
                {"section": "1. 章节标题", "url": "https://..."}
              ],
              "gaps": [
                {"section": "3. 某章节", "missing": "缺少的具体信息描述"}
              ]
            }
            """)

    # ---------- 扩展点 ----------

    def _extra_messages(self, state: State) -> list:
        plan = state.get("research_plan", {})
        facts = state.get("extracted_facts", {})

        parts: list[str] = []

        if plan:
            parts.append(f"## 研究大纲\n```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")

        if facts:
            parts.append(f"## 萃取素材\n```json\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n```")
        else:
            parts.append("## 萃取素材\n（无素材，请根据大纲和你的知识撰写，并标注所有章节为 GAP）")

        return [SystemMessage(content="\n\n".join(parts))]

    def _build_response(self, response: AIMessage) -> dict:
        result = super()._build_response(response)
        draft = self._parse_draft(response.content)
        if draft:
            result["draft_report"] = draft
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
    def _parse_draft(cls, content: str) -> dict | None:
        try:
            json_str = cls._extract_json(content)
            if json_str is None:
                return None
            data = json.loads(json_str)
            if "content" in data:
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None
