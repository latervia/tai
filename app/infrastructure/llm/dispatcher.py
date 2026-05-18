"""模型调度器 — LLM 调用的统一入口，含限流、重试、追踪、成本控制"""
import time
import asyncio
from typing import Optional

from langchain_core.messages import BaseMessage

from app.shared.logger import logger
from app.infrastructure.llm.factory import ModelProvider, ModelFactory


class BudgetExceededError(Exception):
    def __init__(self, used: int, limit: int):
        self.used = used
        self.limit = limit
        super().__init__(f"Token 预算超限: 已用 {used}/{limit}")


class ModelDispatcher:
    """统一的 LLM 调用调度器

    职责：
    1. 主备模型切换
    2. 指数退避重试（仅 5xx / 429）
    3. Token 预算控制（接入 CostController）
    4. 调用追踪（接入 TraceCollector）
    """

    def __init__(
        self,
        primary_provider: ModelProvider,
        *,
        max_retries: int = 2,
        token_budget: Optional[int] = None,
        base_delay: float = 1.0,
    ):
        self.primary_llm = ModelFactory.get_model(primary_provider)
        self.fallback_llm = ModelFactory.get_model(ModelProvider.OLLAMA)
        self.max_retries = max_retries
        self.token_budget = token_budget
        self.base_delay = base_delay

        self._tokens_used: int = 0
        self._call_count: int = 0
        self._call_records: list[dict] = []

    # ── 公共接口 ──────────────────────────────────────────

    async def invoke(
        self, messages: list[BaseMessage], *,
        session_id: str = "", agent_name: str = "",
        **kwargs,
    ) -> dict:
        """调用 LLM，自动处理重试和 fallback"""
        return await self._invoke_with_retry(
            self.primary_llm, messages,
            session_id=session_id, agent_name=agent_name,
            **kwargs,
        )

    async def think(
        self, messages: list[BaseMessage], *,
        session_id: str = "", agent_name: str = "supervisor",
        **kwargs,
    ) -> dict:
        """轻量调用 — 用于路由判断等简单任务"""
        kwargs.setdefault("temperature", 0.1)
        return await self.invoke(
            messages, session_id=session_id, agent_name=agent_name, **kwargs,
        )

    # ── 内部实现 ──────────────────────────────────────────

    async def _invoke_with_retry(
        self, llm, messages, *, session_id: str, agent_name: str, **kwargs,
    ) -> dict:
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._do_invoke(llm, messages, session_id=session_id, agent_name=agent_name, **kwargs)
            except Exception as e:
                last_error = e
                if self._is_client_error(e):
                    logger.warning(f"LLM 客户端错误，不重试: {e}")
                    break
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"LLM 调用失败 (attempt {attempt+1})，{delay}s 后重试: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM 重试耗尽: {e}")

        if llm is not self.fallback_llm:
            logger.warning("切换到 fallback 模型")
            try:
                return await self._do_invoke(
                    self.fallback_llm, messages,
                    session_id=session_id, agent_name=agent_name, **kwargs,
                )
            except Exception as e:
                last_error = e

        raise last_error or RuntimeError("LLM 调用失败且无 fallback")

    async def _do_invoke(
        self, llm, messages, *, session_id: str, agent_name: str, **kwargs,
    ) -> dict:
        """单次 LLM 调用 + token 追踪 + 成本控制"""
        start = time.time()

        # 预算检查
        if self.token_budget and self._tokens_used >= self.token_budget:
            raise BudgetExceededError(self._tokens_used, self.token_budget)

        response = await llm.ainvoke(messages, **kwargs)

        duration_ms = (time.time() - start) * 1000
        tokens = self._extract_tokens(response)
        self._tokens_used += tokens.get("total", 0)
        self._call_count += 1

        model_name = getattr(llm, "model_name", "unknown")

        # → 接入全局 tracing
        from app.deps import get_trace_collector
        get_trace_collector().llm_call(
            session_id=session_id,
            agent=agent_name,
            model=model_name,
            tokens=tokens,
            duration_ms=duration_ms,
            input_preview=str(messages[-1].content)[:200] if messages else "",
        )

        # → 接入全局 cost
        from app.deps import get_cost_controller
        get_cost_controller().track(session_id, tokens)

        self._call_records.append({
            "call_no": self._call_count,
            "model": model_name,
            "tokens": tokens,
            "duration_ms": round(duration_ms, 1),
        })
        logger.info(
            f"[LLM #{self._call_count}] {model_name} "
            f"tokens={tokens} latency={round(duration_ms, 1)}ms"
        )

        return {
            "message": response,
            "tokens": tokens,
            "duration_ms": duration_ms,
        }

    # ── 辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _extract_tokens(response) -> dict:
        tokens = {"input": 0, "output": 0, "total": 0}
        meta = getattr(response, "usage_metadata", None) or {}
        if meta:
            tokens["input"] = meta.get("input_tokens", 0)
            tokens["output"] = meta.get("output_tokens", 0)
            tokens["total"] = tokens["input"] + tokens["output"]
        rm = getattr(response, "response_metadata", {}) or {}
        if not tokens["total"] and "token_usage" in rm:
            tu = rm["token_usage"]
            tokens["input"] = tu.get("prompt_tokens", 0)
            tokens["output"] = tu.get("completion_tokens", 0)
            tokens["total"] = tokens["input"] + tokens["output"]
        return tokens

    @staticmethod
    def _is_client_error(exception: Exception) -> bool:
        msg = str(exception).lower()
        for code in ["400", "401", "402", "403", "404"]:
            if code in msg:
                return True
        return any(kw in msg for kw in ["invalid request", "invalid api key", "permission", "quota"])

    # ── 查询接口 ──────────────────────────────────────────

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def call_count(self) -> int:
        return self._call_count

    def get_trace(self) -> list[dict]:
        return list(self._call_records)

    def reset_budget(self):
        self._tokens_used = 0
        self._call_count = 0
        self._call_records.clear()
