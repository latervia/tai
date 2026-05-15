# Multi-Agent 架构设计

## 一、核心设计原则

1. **渐进式复杂度**：从单 Agent 到多 Agent 应是"新增节点 + 新增路由"而非"重写整个图"。
2. **图即文档**：Agent 间的协作关系直接体现在 LangGraph 的图结构中，而非隐藏在消息传递里。
3. **状态驱动通信**：Agent 之间不直接"发消息"，而是通过共享 State 协作——LangGraph 的 State 就是天然的"黑板"。
4. **工具即能力边界**：每个 Agent 的能力边界由其可调用的 Tool 集合定义，而不是代码耦合。

## 二、分层架构（与现有代码的对齐）

```
┌──────────────────────────────────────────────────┐
│  API Layer (已有: app/api/routers/)              │
├──────────────────────────────────────────────────┤
│  Orchestrator (编排层)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Supervisor│ │ Router   │ │ Scheduler│          │
│  └──────────┘ └──────────┘ └──────────┘          │
├──────────────────────────────────────────────────┤
│  Agent 层 (app/agent/)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │BaseAgent │ │ChatAgent │ │RAGAgent  │ ...      │
│  └──────────┘ └──────────┘ └──────────┘          │
├──────────────────────────────────────────────────┤
│  工具层 (app/tools/)  — 已有 file_tools          │
│  记忆层 (app/agent/memory/)  — 待实现            │
│  模型层 (app/core/model/)  — 已有 ModelFactory   │
├──────────────────────────────────────────────────┤
│  基础设施 (app/core/)                            │
│  Redis / Milvus / Postgres / Minio               │
└──────────────────────────────────────────────────┘
```

### 2.1 模型抽象层（已有，需增强）

`ModelFactory` 和 `ModelDispatcher` 已具备基本的 provider 抽象和 fallback 能力。

**需补充的设计点：**
- **速率限制**：在 `ModelDispatcher` 中增加 per-provider 的 rate limiter，防止超额调用。
- **成本追踪**：每次调用记录 token 消耗，按 session 或 task 聚合，用于成本审计。
- **重试策略**：指数退避 + 仅重试 5xx/429，避免重复扣费。

### 2.2 BaseAgent（核心抽象——当前缺失）

当前 `nodes.py` 中的 `chat_node` 是一个裸函数，没有 Agent 抽象。应抽象出标准 Agent 接口：

```python
class BaseAgent:
    """Agent 基类 — 所有 Agent 继承自此"""

    # 身份
    name: str                          # 唯一标识，如 "rag_agent"
    description: str                   # 自然语言描述，供 Router 判断谁能处理

    # 能力
    tools: list[Callable]              # 该 Agent 可调用的工具
    model: BaseChatModel               # 绑定的 LLM

    # 核心方法
    async def think(self, state: State) -> AgentThought: ...
    async def act(self, thought: AgentThought) -> ActionResult: ...
    async def should_handle(self, state: State) -> bool: ...  # 意图匹配
```

**关键设计决策**: Agent 不直接输出最终回复，而是输出 `AgentThought`（思考结果），由 Orchestrator 决定下一步。这避免了"Agent 说了算"的黑盒问题。

### 2.3 工具层（需规范化）

当前 `app/tools/` 已有 `file_tools.py`，但缺少注册机制。

```python
# 工具注册表 — 统一管理，支持按名称查找、按 Agent 授权
class ToolRegistry:
    _tools: dict[str, ToolDef] = {}

    @classmethod
    def register(cls, name: str, fn: Callable, *, permissions: list[str] = None): ...
    @classmethod
    def get_for_agent(cls, agent_name: str) -> list[ToolDef]: ...  # 按 Agent 权限过滤
```

**安全设计**: 每个 Agent 声明自己需要的工具权限，敏感工具（如文件写入、数据库修改）需要人类确认后才执行。

### 2.4 记忆层（待实现）

```
Short-term:  State.messages (LangGraph 自动管理，已有)
             State.summary_memory (已有字段，需实现摘要逻辑)
Long-term:   Vector Store (Milvus 已有，需对接 Agent 记忆写入)
Working:     State.task_context (当前任务的目标、约束、中间产物)
```

**关键设计**: 不同类型记忆有不同的生命周期和读写策略：
- `messages`: LangGraph add_messages reducer 自动管理
- `summary_memory`: 当 messages 超过阈值时，由独立节点触发摘要并写入
- 长期记忆: 写入 Milvus，按 `(session_id, importance)` 过滤检索

## 三、编排模式（核心——基于 LangGraph DAG）

### 3.1 为什么选 DAG + Supervisor 而非纯消息传递

项目已使用 LangGraph，State 天然是共享的。用 DAG 编排意味着：
- **可预测**: 状态流转路径在图中可见，排查问题直接看图
- **可中断**: 任意节点间插入 Human-in-the-loop 或审批
- **可复用**: 每个 Agent 是独立节点，可被不同图复用

不选纯 Actor/消息模型的原因：异步消息在排查"Agent A 为什么不回复 Agent B"时极其痛苦——消息队列是不可见的图。

### 3.2 Supervisor 模式（推荐用于路由）

```
         ┌─────────────┐
         │  Supervisor  │  ← 分析意图，决定派发给谁
         └──────┬───────┘
         ┌──────┼──────┐
    ┌────▼──┐ ┌─▼───┐ ┌▼──────┐
    │ Chat  │ │ RAG │ │ Tool  │  ← 各司其职的 Agent
    └────┬──┘ └─┬───┘ └───┬───┘
         └──────┼──────┘
          ┌─────▼─────┐
          │ Finalize  │  ← 汇总结果，生成最终回复
          └───────────┘
```

Supervisor 本身也是一个轻量 LLM 调用：根据用户意图 + 各 Agent 的 `description` 字段判断路由目标。

**路由策略**（从简单到复杂）：
1. **意图分类**: 用 LLM 做 few-shot 分类，映射到 Agent
2. **关键词 + 规则**: 简单场景用规则过滤，减少一次 LLM 调用
3. **Agent 主动声明**: 每个 Agent 实现 `should_handle(state)`，Supervisor 逐个询问"你能处理这个吗？"

### 3.3 串行 vs 并行编排

```python
# 串行: A→B→C（依赖关系明确时使用）
graph.add_edge("agent_a", "agent_b")
graph.add_edge("agent_b", "agent_c")

# 并行: Supervisor → [A, B] 同时执行，然后 Merge
# LangGraph 的 Send API 可实现 fan-out → fan-in
```

**设计建议**: 默认串行，显式声明并行。并行增加了调试复杂度，只在性能确实需要时启用。

## 四、缺失的关键设计

### 4.1 错误处理与韧性

多 Agent 系统的故障模式与单 Agent 完全不同——一个 Agent 的失败可能级联。

```python
class AgentResult:
    status: Literal["success", "partial", "failed"]
    output: Any
    error: Optional[str]
    retry_count: int

# 编排层的错误策略
error_policy = {
    "retry": 2,                      # 每个 Agent 最多重试 2 次
    "fallback_agent": "chat_agent",   # 特定 Agent 失败后的兜底
    "timeout": 30,                    # 单个 Agent 执行超时
    "on_permanent_failure": "notify_human"  # 彻底失败时人工介入
}
```

**关键**: 不要在一个 Agent 内部吞掉所有异常然后返回空结果——让编排层感知失败并决策。

### 4.2 可观测性（从设计阶段就嵌入）

当前项目无 tracing。多 Agent 系统必须做到：

```
每个请求的完整链路：
  Request → Supervisor(意图=知识问答) → RAG_Agent(检索3条, 耗时1.2s) → Finalize(生成回答, 耗时0.8s)
           ├── LLM调用: 3次
           ├── Token消耗: 4500 input + 800 output
           └── 工具调用: search_kb × 1
```

**实现路径**：
1. 在 `ModelDispatcher.chat()` 中挂钩子，记录每次 LLM 调用的 input/output/tokens/latency
2. 在 Agent 的 think/act 边界记录 timeline
3. 结构化日志输出到 Postgres（已有连接），用 session_id 串联

不需要引入 LangSmith 等外部服务——先在 Postgres 里存结构化 trace，够用且可控。

### 4.3 State 设计（扩展现有 State）

当前 State 过于简单，仅有 messages 和 summary_memory。多 Agent 需要：

```python
class MultiAgentState(TypedDict):
    # 继承自现有 State
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    summary_memory: NotRequired[str]

    # 多 Agent 扩展
    current_agent: NotRequired[str]              # 当前正在执行的 Agent
    agent_outputs: NotRequired[dict[str, Any]]   # 各 Agent 的输出产物
    task_plan: NotRequired[list[TaskStep]]       # 当前任务的分步计划
    pending_approval: NotRequired[ApprovalRequest]  # 等待人类审批的操作
    trace: NotRequired[list[TraceEvent]]         # 可观测性事件
```

### 4.4 Human-in-the-Loop

高风险操作（数据删除、外部 API 写入、费用超过阈值）必须中断等待人类确认。

LangGraph 原生支持：在需要审批的节点后调用 `interrupt()`，状态持久化到 Redis，等待人类确认后继续。

```python
# 敏感工具标记
@tool(requires_approval=True, risk_level="high")
def delete_records(query: str): ...

# Graph 中的审批节点
def approval_node(state):
    if state.get("pending_approval"):
        return interrupt("需要人工审批")  # 暂停，等待外部确认
    return state
```

### 4.5 Prompt 管理

当前 `chat_node` 直接用 messages 调 LLM，没有 system prompt。多 Agent 场景下，每个 Agent 的 system prompt 需要版本化管理：

```
app/agent/prompts/
  supervisor.yaml      # Supervisor 的系统提示
  chat_agent.yaml      # 对话 Agent 的提示
  rag_agent.yaml       # RAG Agent 的提示
```

每个 prompt 文件包含 `version`, `template`, `variables`，支持热更新（从文件/RDB 加载，不重启服务）。

### 4.6 测试策略

多 Agent 测试分为三层：

| 层级 | 测试对象 | 策略 |
|------|---------|------|
| 单元 | 单个 Agent 的 think/act | Mock LLM 返回固定值，验证逻辑分支 |
| 集成 | 两个 Agent 的协作流 | 用真实 LLM 但固定 seed/temperature=0 |
| E2E | 完整对话流程 | 用评估集（eval set）跑，比较输出质量 |

**关键**: 不要 Mock 所有 LLM 调用。有些 bug 只在真实 LLM 输出下才会暴露——保留一组"金标准"测试用例用真实模型跑。

### 4.7 成本控制

```
预算策略:
  - 每次对话最高消耗 $0.5
  - 单个 Agent 调用限制 10 轮 tool-use 循环
  - Supervisor + 2 个 Worker Agent 的默认配置
  - 超过限制时 graceful degradation（用已收集的信息生成回答）
```

在 `ModelDispatcher` 中实现 token 计数器，达到阈值时抛出 `BudgetExceededError`，编排层据此做降级处理。

## 五、实施路径（从当前代码出发）

### Phase 1: 单 Agent 标准化（当前→完善）

1. 从 `chat_node` 提取出 `BaseAgent` 抽象
2. 实现 `ToolRegistry`，将 `app/tools/` 下的工具注册进去
3. 在 `ModelDispatcher` 中增加 tracing hook

### Phase 2: 引入 Supervisor

1. 扩展 `State` 为 `MultiAgentState`
2. 新增 `SupervisorNode`：分析意图 → 路由到对应 Agent
3. 实现 `FinalizeNode`：汇总结果 → 生成最终回复
4. 重新 compile graph：Supervisor → [ChatAgent | RAGAgent] → Finalize

### Phase 3: 多 Agent 协作

1. 实现 Agent 间的串行/并行编排
2. 增加 Human-in-the-loop 审批节点
3. 接入长期记忆（Milvus）
4. 增加评估集和自动化测试

### Phase 4: 生产就绪

1. 完整的可观测性面板（基于 Postgres trace 数据）
2. 成本追踪与预算告警
3. Prompt 版本管理与 A/B 测试
4. 性能优化（LLM 调用缓存、并行执行）

---

## 六、与业界框架的对比及选型理由

| 维度 | LangGraph (本项目选择) | AutoGen | CrewAI |
|------|----------------------|---------|--------|
| 编排可控性 | ⭐⭐⭐ 显式图，路径可见 | ⭐⭐ 对话驱动，隐式 | ⭐ 高层封装，黑盒 |
| 与现有代码兼容 | ⭐⭐⭐ 已在用 | ⭐ 需要替换 | ⭐ 需要替换 |
| 生产就绪度 | ⭐⭐⭐ 中断/恢复/流式 | ⭐⭐ | ⭐ |
| 学习曲线 | 中等 | 中等 | 低 |

本项目已基于 LangGraph 构建，继续深入 LangGraph 的 Supervisor/Subgraph 能力是最小迁移成本的选择。
