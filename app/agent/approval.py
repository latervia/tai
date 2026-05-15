"""Human-in-the-Loop 审批机制 — 高风险操作需人类确认后才能执行"""
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """操作风险等级"""
    LOW = "low"          # 只读操作，无需审批
    MEDIUM = "medium"    # 数据修改，建议审批
    HIGH = "high"        # 数据删除/外部写入，必须审批
    CRITICAL = "critical"  # 系统级操作，强制审批 + 二次确认


@dataclass
class ApprovalRequest:
    """等待审批的操作"""
    tool_name: str
    tool_args: dict
    reason: str                     # 为什么需要这个操作
    risk_level: RiskLevel
    request_id: str                 # 唯一标识，用于审批回调
    status: str = "pending"         # pending | approved | rejected


class ApprovalManager:
    """审批管理器

    高风险工具调用前，Agent 创建一个 ApprovalRequest 并挂起。
    外部系统（如 UI）根据 request_id 审批后，流程继续。
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
        """创建审批请求"""
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
        """批准一个请求"""
        req = self._pending.get(request_id)
        if req and req.status == "pending":
            req.status = "approved"
            return True
        return False

    def reject(self, request_id: str, reason: str = "") -> bool:
        """拒绝一个请求"""
        req = self._pending.get(request_id)
        if req and req.status == "pending":
            req.status = "rejected"
            req.reason = reason
            return True
        return False

    def get(self, request_id: str) -> Optional[ApprovalRequest]:
        """查询审批请求状态"""
        return self._pending.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        """列出所有待审批的请求"""
        return [r for r in self._pending.values() if r.status == "pending"]

    def cleanup(self, request_id: str):
        """审批完成后清理"""
        self._pending.pop(request_id, None)


# 全局审批管理器实例
_approval_manager: Optional[ApprovalManager] = None


def get_approval_manager() -> ApprovalManager:
    """获取全局审批管理器（惰性初始化）"""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager
