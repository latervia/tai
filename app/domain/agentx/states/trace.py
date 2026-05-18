from enum import Enum
from typing import TypedDict, List


class TraceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


class TraceStep(TypedDict):
    node: str
    input: dict
    output: dict
    start_at: float
    end_at: float
    status: TraceStatus


class Trace(TypedDict):
    """单次请求的追踪事件"""
    trace_id: str
    steps: List[TraceStep]
