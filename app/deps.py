"""统一依赖注入 — 所有全局单例的集中管理点

解决之前 5 处分散的 get_xxx() 惰性单例问题。
测试时可直接替换实例，无需逐个 reset。
"""
from typing import Optional


# ── 延迟导入以避免循环依赖 ──────────────────────────────

def _create_trace_collector():
    from app.shared.tracing import TraceCollector
    return TraceCollector()


def _create_cost_controller():
    from app.shared.cost import CostController
    return CostController()


def _create_approval_manager():
    from app.domain.agent.human.approval import ApprovalManager
    return ApprovalManager()


def _create_prompt_manager():
    from app.domain.agent.prompts.manager import PromptManager
    return PromptManager()


# ── 容器 ─────────────────────────────────────────────────

class Container:
    """全局依赖容器

    所有 get_xxx() 调用统一代理到这里，
    测试时可替换 _instance 或单独 override。
    """

    _instance: Optional["Container"] = None

    def __init__(self):
        self._trace = None
        self._cost = None
        self._approval = None
        self._prompt = None

    @property
    def trace(self):
        if self._trace is None:
            self._trace = _create_trace_collector()
        return self._trace

    @property
    def cost(self):
        if self._cost is None:
            self._cost = _create_cost_controller()
        return self._cost

    @property
    def approval(self):
        if self._approval is None:
            self._approval = _create_approval_manager()
        return self._approval

    @property
    def prompt(self):
        if self._prompt is None:
            self._prompt = _create_prompt_manager()
        return self._prompt

    def reset(self):
        """重置所有依赖（测试用）"""
        self._trace = None
        self._cost = None
        self._approval = None
        self._prompt = None


# ── 全局访问 ─────────────────────────────────────────────

def _container() -> Container:
    if Container._instance is None:
        Container._instance = Container()
    return Container._instance


def get_trace_collector():
    return _container().trace


def get_cost_controller():
    return _container().cost


def get_approval_manager():
    return _container().approval


def get_prompt_manager():
    return _container().prompt


def reset_all_deps():
    """一次性重置所有全局依赖（测试 setup 用）"""
    _container().reset()
    Container._instance = None
