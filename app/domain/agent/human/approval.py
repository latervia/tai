"""审批管理器 — 高风险操作需人类确认后才能执行"""
from typing import Optional

from app.domain.agent.types import ApprovalRequest, RiskLevel


class ApprovalManager:
    """审批管理器

    高风险工具调用前创建 ApprovalRequest 并挂起。
    外部系统（如 UI / API）根据 request_id 审批后，流程继续。
    """

    def __init__(self):
        self._pending: dict[str, ApprovalRequest] = {}

    def request(
        self,
        request_id: str,
        tool_name: str,
        tool_args: dict,
        reason: str,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            tool_args=tool_args,
            reason=reason,
            risk_level=risk_level,
        )
        self._pending[request_id] = req
        return req

    def approve(self, request_id: str) -> bool:
        req = self._pending.get(request_id)
        if req and req.status == "pending":
            req.status = "approved"
            return True
        return False

    def reject(self, request_id: str, reason: str = "") -> bool:
        req = self._pending.get(request_id)
        if req and req.status == "pending":
            req.status = "rejected"
            req.reason = reason
            return True
        return False

    def get(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._pending.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        return [r for r in self._pending.values() if r.status == "pending"]

    def cleanup(self, request_id: str):
        self._pending.pop(request_id, None)
