# TAI — Multi-Agent + RAG System

## 快速启动

```bash
# 启动依赖服务
docker-compose -f docker/docker-compose.yaml up -d

# 启动 FastAPI
uvicorn app.delivery.fastapi.main:app --reload
```

---

## 架构

### 分层原则

**依赖只能向下。** 上层可以 import 下层，反之禁止。

```
┌──────────────────────────────────────────────────┐
│  delivery/        ← FastAPI / CLI / gRPC         │  框架适配，可替换
├──────────────────────────────────────────────────┤
│  services/        ← 薄编排层                      │  连接 domain ↔ infrastructure
├──────────────────┬───────────────────────────────┤
│  domain/agent/   │  domain/rag/                  │  核心域（必选）
│                  │  （可选，可按需删掉）             │  业务域（可选）
├──────────────────┴───────────────────────────────┤
│  infrastructure/  ← DB / 存储 / 缓存 / LLM        │  基础设施实现，可替换
├──────────────────────────────────────────────────┤
│  shared/          ← config / logger / 追踪 / 成本  │  全局共享，零外部依赖
└──────────────────────────────────────────────────┘
```

### 各层职责

| 层 | 目录 | 依赖对象 | 核心约束 |
|---|---|---|---|
| **shared** | `app/shared/` | 无 | 不 import 任何业务代码 |
| **domain** | `app/domain/{agent,rag,...}` | shared | 不 import 框架、数据库、存储 |
| **infrastructure** | `app/infrastructure/` | shared | 不 import domain 或 services |
| **services** | `app/services/` | domain + infrastructure | 薄编排，不含业务逻辑 |
| **delivery** | `app/delivery/{fastapi,...}` | services | 框架专属代码集中于此 |

### 组合根

`app/deps.py` — 全局依赖容器，负责懒加载所有单例（TraceCollector、CostController、ApprovalManager、PromptManager），通过 `Container` 统一管理生命周期。测试时可通过 `reset_all_deps()` 一键重置。

---

## 目录结构

```
app/
├── shared/                         # 共享内核（零外部依赖）
│   ├── config.py                   #   全局配置（Pydantic Settings）
│   ├── logger.py                   #   日志
│   ├── cost.py                     #   Token 成本控制器
│   ├── tracing.py                  #   全链路追踪收集器
│   └── lifecycle.py                #   启动/关闭事件
│
├── domain/                         # 领域层（只依赖 shared）
│   ├── agent/                      #   Agent 核心域
│   │   ├── graph.py                #     GraphRuntime — 图运行时
│   │   ├── base.py                 #     BaseAgent — Agent 基类
│   │   ├── registry.py             #     AgentRegistry — 注册中心
│   │   ├── nodes.py                #     LangGraph 节点 + Supervisor
│   │   ├── state.py                #     MultiAgentState 图状态
│   │   ├── types.py                #     领域类型（AgentStatus, ToolDef,...）
│   │   ├── approval.py             #     审批管理器
│   │   ├── memory/                 #     长期记忆管理器
│   │   ├── prompts/                #     Prompt 管理器 + YAML 模板
│   │   ├── tools/                  #     工具注册 + 消息适配
│   │   └── workers/                #     具体 Agent 实现（chat, rag）
│   │
│   └── rag/                        #   RAG 域（可选）
│       ├── chunk/                  #     文本切分策略
│       ├── convert/                #     文档转换器
│       ├── embedding/              #     向量嵌入实现
│       └── pipeline/               #     RAG 处理管道
│
├── infrastructure/                 # 基础设施实现层
│   ├── llm/                        #   LLM 工厂 + 调度器（重试、fallback）
│   ├── storage/                    #   MinIO 对象存储
│   ├── database/                   #   PostgreSQL + SQLAlchemy ORM
│   ├── vectorstore/                #   Milvus 向量存储
│   └── cache/                      #   Redis 缓存
│
├── services/                       # 编排层
│   ├── chat_service.py             #   Chat 服务
│   └── rag_service.py              #   RAG 服务
│
├── delivery/                       # 交付层
│   └── fastapi/                    #   FastAPI 适配器
│       ├── main.py                 #     create_app() 工厂
│       ├── result.py               #     统一响应包装
│       ├── routers/                #     API 路由
│       └── schemas/                #     Pydantic 请求/响应模型
│
└── deps.py                         # 组合根（全局依赖容器）
```

> 顶层 `app/main.py`、`app/agent/`、`app/rag/`、`app/core/` 等为向后兼容 shim，
> 实际代码已全部迁移至上述新结构。

---

## 作为模板复制到新项目

### 必选部分（任何项目都保留）

```
app/shared/              # 全保留
app/deps.py              # 全保留
app/domain/agent/        # 全保留
app/infrastructure/llm/  # 全保留（LLM 调度）
app/services/            # chat_service.py 保留，rag_service.py 留空壳
app/delivery/fastapi/    # 保留 main.py + result.py + routers/chat.py
```

### 可选部分（按需保留或删除）

| 场景 | 操作 |
|---|---|
| 不需要 RAG | 删除 `app/domain/rag/`、`app/services/rag_service.py`、`app/delivery/fastapi/routers/rag.py` |
| 不需要 FastAPI | 删除 `app/delivery/fastapi/`，新增 `app/delivery/cli/` 或直接调 `domain/agent` |
| 不需要 Milvus | 删除 `app/infrastructure/vectorstore/` |
| 不需要 MinIO | 删除 `app/infrastructure/storage/` |
| 不需要 PostgreSQL | 删除 `app/infrastructure/database/`，domain 中的记忆管理器退化为内存模式 |
| 新增业务域 | 在 `app/domain/` 下新增子包，在 `services/` 新增编排，在 `delivery/` 新增路由 |

### 裁剪示例

```bash
# 最简项目：只要 Agent + LLM，不要 RAG、不要数据库
rm -rf app/domain/rag
rm -rf app/infrastructure/vectorstore app/infrastructure/storage app/infrastructure/database app/infrastructure/cache
rm app/services/rag_service.py
rm app/delivery/fastapi/routers/rag.py
```

---

## 开发规范

### Import 规则

```python
# ✅ domain 层：只能 import shared
from app.shared.logger import logger
from app.domain.agent.workers.base import BaseAgent

# ❌ domain 层：禁止 import infrastructure 或框架
# from app.infrastructure.llm import ModelDispatcher  ← 违规

# ✅ infrastructure 层：只能 import shared
from app.shared.config import settings

# ✅ delivery 层：可以 import services / domain / infrastructure
from app.services.chat_service import get_chat_service
```

### 已知待解耦问题

| 位置 | 问题 |
|---|---|
| `domain/agent/graph.py` | 直接依赖 `infrastructure/llm` (ModelDispatcher) |
| `domain/rag/convert/` | 直接依赖 `infrastructure/llm` + `storage` |
| `domain/rag/embedding/` | 直接依赖 `infrastructure/vectorstore` |
| `services/rag_service.py` | 依赖 `fastapi.Depends`, `fastapi.UploadFile` |

> 上述耦合通过引入 Protocol/ABC 接口即可消除。当前不影响功能，属于架构演进方向。

---

## 测试

```bash
pytest tests/ -v
```
