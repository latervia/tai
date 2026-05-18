# README

```bash
# 启动FastAPI
uvicorn app.main:app --reload
```

```bash
# 启动docker-compose
docker-compose -f docker/docker-compose.yaml up -d
```

```text
# 目标架构(依赖只能向下)
  ┌──────────────────────────────────────────┐
  │  delivery/          ← FastAPI / CLI      │  可替换
  ├──────────────────────────────────────────┤
  │  services/          ← 薄编排层             │  按领域组织
  ├──────────────────┬───────────────────────┤
  │  domain/agent/   │  domain/rag/          │  agent 必选
  │  （核心域）        │  (可选域)               │  rag  可选
  ├──────────────────┴───────────────────────┤
  │  infrastructure/   ← DB / 存储 / 缓存      │  可替换
  ├──────────────────────────────────────────┤
  │  shared/           ← config / logger     │  全局共享
  └──────────────────────────────────────────┘
```