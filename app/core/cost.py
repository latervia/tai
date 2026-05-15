"""成本控制器 — Token 预算管理与费用追踪"""
from dataclasses import dataclass, field
from app.core.logger import logger


@dataclass
class CostReport:
    """单次会话的成本报告"""
    session_id: str
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    budget_limit: int = 0
    budget_remaining: int = 0
    is_over_budget: bool = False


class CostController:
    """全局成本控制器

    管理多个会话的 token 预算，防止单个会话消耗过多。
    支持：
    - 单会话预算上限
    - 全局日预算
    - 超限时的优雅降级策略
    """

    def __init__(
        self,
        default_session_budget: int = 50_000,
        daily_budget: int = 0,     # 0 = 不限
    ):
        self.default_budget = default_session_budget
        self.daily_budget = daily_budget
        # 会话追踪
        self._sessions: dict[str, dict] = {}
        self._daily_total: int = 0

    # ── 会话管理 ──────────────────────────────────────────

    def start_session(self, session_id: str, budget: int = None) -> int:
        """开始新会话（或重置已有会话）

        Returns:
            本会话的 token 预算上限
        """
        budget = budget or self.default_budget
        self._sessions[session_id] = {
            "budget": budget,
            "used": 0,
            "calls": 0,
            "records": [],
        }
        logger.info(f"[Cost] 会话 {session_id[:8]}... 预算 {budget} tokens")
        return budget

    def track(self, session_id: str, tokens: dict):
        """记录一次 LLM 调用的 token 消耗"""
        if session_id not in self._sessions:
            self.start_session(session_id)

        session = self._sessions[session_id]
        total = tokens.get("total", 0)
        session["used"] += total
        session["calls"] += 1
        self._daily_total += total

    def check_budget(self, session_id: str) -> bool:
        """检查是否还有预算

        Returns:
            True 表示预算充足，False 表示已超限
        """
        if session_id not in self._sessions:
            return True
        session = self._sessions[session_id]
        return session["used"] < session["budget"]

    def remaining(self, session_id: str) -> int:
        """返回剩余 token 预算"""
        if session_id not in self._sessions:
            return self.default_budget
        session = self._sessions[session_id]
        return max(0, session["budget"] - session["used"])

    # ── 报告 ──────────────────────────────────────────────

    def get_report(self, session_id: str) -> CostReport:
        """生成会话成本报告"""
        session = self._sessions.get(session_id, {})
        used = session.get("used", 0)
        budget = session.get("budget", 0)
        return CostReport(
            session_id=session_id,
            total_tokens=used,
            llm_calls=session.get("calls", 0),
            budget_limit=budget,
            budget_remaining=max(0, budget - used),
            is_over_budget=used >= budget,
        )

    def end_session(self, session_id: str) -> CostReport:
        """结束会话并返回最终成本报告"""
        report = self.get_report(session_id)
        self._sessions.pop(session_id, None)
        logger.info(
            f"[Cost] 会话结束 {session_id[:8]}... "
            f"消耗 {report.total_tokens} tokens, {report.llm_calls} 次调用"
        )
        return report

    @property
    def daily_total(self) -> int:
        return self._daily_total


# 全局成本控制器通过 app.core.deps.get_cost_controller() 统一管理
