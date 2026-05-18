"""可观测性追踪 — 全链路追踪 Agent 执行过程"""
import time
import json
from typing import Optional, Any
from collections import defaultdict

from app.shared.logger import logger


class TraceCollector:
    """全链路追踪收集器

    记录每个请求的完整执行链路：
    - LLM 调用（模型、tokens、耗时）
    - 工具调用（工具名、参数、结果）
    - Agent 切换（Supervisor 的路由决策）
    - 异常事件

    数据可序列化为 JSON，供前端面板或 Postgres 存储使用。
    """

    def __init__(self):
        self._traces: dict[str, dict] = defaultdict(self._new_trace)

    @staticmethod
    def _new_trace() -> dict:
        return {
            "start_time": time.time(),
            "events": [],
            "summary": {},
        }

    # ── 事件记录 ──────────────────────────────────────────

    def start_request(self, session_id: str, message: str):
        """开始追踪一个请求"""
        trace = self._traces[session_id]
        trace["start_time"] = time.time()
        trace["events"].append({
            "type": "request_start",
            "timestamp": time.time(),
            "message_preview": message[:200],
        })

    def llm_call(
        self,
        session_id: str,
        *,
        agent: str,
        model: str,
        tokens: dict,
        duration_ms: float,
        input_preview: str = "",
    ):
        """记录一次 LLM 调用"""
        self._traces[session_id]["events"].append({
            "type": "llm_call",
            "timestamp": time.time(),
            "agent": agent,
            "model": model,
            "tokens": tokens,
            "duration_ms": duration_ms,
            "input_preview": input_preview[:200],
        })

    def tool_call(
        self,
        session_id: str,
        *,
        agent: str,
        tool_name: str,
        tool_args: dict,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ):
        """记录一次工具调用"""
        self._traces[session_id]["events"].append({
            "type": "tool_call",
            "timestamp": time.time(),
            "agent": agent,
            "tool_name": tool_name,
            "tool_args": str(tool_args)[:500],    # 截断长参数
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
        })

    def agent_switch(
        self,
        session_id: str,
        *,
        from_agent: str,
        to_agent: str,
        reason: str = "",
    ):
        """记录一次 Agent 路由切换"""
        self._traces[session_id]["events"].append({
            "type": "agent_switch",
            "timestamp": time.time(),
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
        })

    def error(
        self,
        session_id: str,
        *,
        agent: str,
        error: str,
    ):
        """记录异常"""
        self._traces[session_id]["events"].append({
            "type": "error",
            "timestamp": time.time(),
            "agent": agent,
            "error": error,
        })

    # ── 汇总 ──────────────────────────────────────────────

    def finish_request(self, session_id: str) -> dict:
        """结束追踪，返回汇总摘要"""
        trace = self._traces.pop(session_id, self._new_trace())
        events = trace["events"]
        duration = time.time() - trace["start_time"]

        # 汇总统计
        llm_calls = [e for e in events if e["type"] == "llm_call"]
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        errors = [e for e in events if e["type"] == "error"]

        total_tokens = sum(e.get("tokens", {}).get("total", 0) for e in llm_calls)

        summary = {
            "session_id": session_id,
            "duration_sec": round(duration, 2),
            "event_count": len(events),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "errors": len(errors),
            "total_tokens": total_tokens,
        }

        trace["summary"] = summary
        logger.info(
            f"[Trace] 请求完成: {session_id[:8]}... "
            f"{summary['llm_calls']} LLM calls, "
            f"{summary['total_tokens']} tokens, "
            f"{duration:.1f}s"
            + (f", {len(errors)} errors!" if errors else "")
        )

        return trace

    # ── 查询 ──────────────────────────────────────────────

    def get_trace(self, session_id: str) -> Optional[dict]:
        """获取正在进行的追踪数据（不终止）"""
        return self._traces.get(session_id)

    def to_json(self, session_id: str) -> str:
        """序列化为 JSON（供 API 返回）"""
        trace = self.get_trace(session_id)
        if trace is None:
            return "{}"
        return json.dumps(
            {
                "events": trace.get("events", []),
                "summary": trace.get("summary", {}),
            },
            ensure_ascii=False,
            default=str,     # 处理不可序列化的对象
        )
