"""模型调度与成本控制测试 — 适配重构后结构"""
import pytest
from unittest.mock import AsyncMock, patch

from app.core.model.model_dispatcher import ModelDispatcher, BudgetExceededError
from app.core.model.model_factory import ModelProvider
from app.core.cost import CostController
from app.core import deps


@pytest.fixture(autouse=True)
def reset_state():
    deps.reset_all_deps()
    yield
    deps.reset_all_deps()


class TestModelDispatcher:
    def test_invoke_passes_session_to_trace(self):
        import asyncio
        from langchain_core.messages import AIMessage, HumanMessage

        async def _run():
            dispatcher = ModelDispatcher(ModelProvider.QWEN, max_retries=0)

            response = AIMessage(content="ok")
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = response
            mock_llm.model_name = "test-model"

            with patch.object(dispatcher, 'primary_llm', mock_llm):
                return await dispatcher.invoke(
                    [HumanMessage(content="hi")],
                    session_id="s1", agent_name="test_agent",
                )

        result = asyncio.run(_run())
        assert result["tokens"]["total"] == 15

    def test_extract_tokens(self):
        class R:
            usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            response_metadata = {}
        t = ModelDispatcher._extract_tokens(R())
        assert t["total"] == 150

    def test_extract_tokens_alt(self):
        class R:
            usage_metadata = {}
            response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        t = ModelDispatcher._extract_tokens(R())
        assert t["total"] == 30

    def test_client_error(self):
        assert ModelDispatcher._is_client_error(Exception("HTTP 400"))
        assert ModelDispatcher._is_client_error(Exception("quota exceeded"))
        assert not ModelDispatcher._is_client_error(Exception("timeout"))

    def test_reset_budget(self):
        d = ModelDispatcher(ModelProvider.QWEN)
        d._tokens_used = 500
        d.reset_budget()
        assert d.tokens_used == 0


class TestCostController:
    def test_session_lifecycle(self):
        cc = CostController(default_session_budget=1000)
        cc.start_session("s1")
        cc.track("s1", {"total": 400})
        assert cc.remaining("s1") == 600
        assert cc.check_budget("s1")

    def test_over_budget(self):
        cc = CostController(default_session_budget=100)
        cc.start_session("s1")
        cc.track("s1", {"total": 150})
        assert not cc.check_budget("s1")

    def test_report(self):
        cc = CostController()
        cc.start_session("s1")
        cc.track("s1", {"total": 123})
        r = cc.get_report("s1")
        assert r.total_tokens == 123
        assert r.llm_calls == 1

    def test_daily_total(self):
        cc = CostController()
        cc.track("s1", {"total": 10})
        cc.track("s2", {"total": 20})
        assert cc.daily_total == 30


class TestTraceCollector:
    def test_start_and_finish(self):
        tc = deps.get_trace_collector()
        tc.start_request("s1", "hello")
        assert tc.get_trace("s1") is not None
        trace = tc.finish_request("s1")
        assert "summary" in trace
        assert trace["summary"]["session_id"] == "s1"
