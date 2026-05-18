"""Prompt 管理器 — 统一管理所有 Agent 的 System Prompt，支持热加载"""
import time
from pathlib import Path
from typing import Optional

import yaml

from app.shared.logger import logger


class PromptManager:
    """Prompt 版本管理与加载

    每个 Agent 的 system prompt 存储在 YAML 文件中：
      app/agent/prompts/<agent_name>.yaml

    YAML 格式:
      version: "1.0"
      description: "这个 prompt 的用途"
      template: |
        实际的 prompt 文本，支持 {variable} 占位符

    支持：
    - 按名称加载 prompt
    - 变量插值
    - 文件变更检测（热加载）
    - 回退到内置默认值
    """

    # 所有 Agent 的内置默认 prompt（文件缺失时的回退）
    DEFAULTS = {
        "chat_agent": (
            "你是一个友好、专业的 AI 助手。\n"
            "核心规则：\n"
            "1. 用简洁清晰的中文回复\n"
            "2. 不确定的答案直接说明，不要编造\n"
            "3. 保持对话自然流畅"
        ),
        "rag_agent": (
            "你是一个知识检索助手。\n"
            "核心规则：\n"
            "1. 先使用 search 工具检索相关文档\n"
            "2. 基于检索到的文档内容回答用户问题\n"
            "3. 如果没有检索到相关内容，明确告知用户\n"
            "4. 引用文档内容时注明来源"
        ),
        "supervisor": (
            "你是 Multi-Agent 系统的调度中心。\n"
            "可用 Agent:\n{agent_list}\n"
            "分析用户意图，输出 JSON: {{\"next_agent\": \"<name 或 null>\", \"reason\": \"...\"}}"
        ),
    }

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Args:
            prompts_dir: prompt 文件目录，默认为 app/agent/prompts/
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self._dir = prompts_dir
        self._cache: dict[str, tuple[float, str]] = {}  # name → (mtime, template)
        self._check_interval = 30  # 30 秒内不重复检查文件

    def get(self, name: str, **variables) -> str:
        """获取指定 Agent 的 prompt 模板并填入变量

        Args:
            name: Agent 名称（对应 YAML 文件名，不含 .yaml）
            **variables: 模板变量，如 agent_list="..."

        Returns:
            填充变量后的 prompt 字符串
        """
        template = self._load(name)
        if variables:
            try:
                return template.format(**variables)
            except KeyError as e:
                logger.warning(f"[PromptManager] 模板变量缺失: {e}，使用原始模板")
                return template
        return template

    def _load(self, name: str) -> str:
        """加载 prompt 模板（带文件变更检测）"""
        now = time.time()
        cached = self._cache.get(name)

        # 缓存未过期 → 直接返回
        if cached and (now - cached[0]) < self._check_interval:
            return cached[1]

        # 尝试从文件加载
        filepath = self._dir / f"{name}.yaml"
        if filepath.exists():
            try:
                mtime = filepath.stat().st_mtime
                # 文件未变化 → 返回缓存的模板
                if cached and mtime <= cached[0]:
                    self._cache[name] = (now, cached[1])  # 刷新检查时间
                    return cached[1]
                # 文件有变化 → 重新读取
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    template = data.get("template", "")
                    version = data.get("version", "unknown")
                    self._cache[name] = (mtime, template)
                    logger.info(f"[PromptManager] 加载 {name}.yaml v{version}")
                    return template
            except Exception as e:
                logger.error(f"[PromptManager] 读取 {filepath} 失败: {e}")

        # 文件不存在或读取失败 → 使用内置默认值
        default = self.DEFAULTS.get(name, "")
        if default:
            logger.warning(f"[PromptManager] {name} 使用内置默认 prompt")
        self._cache[name] = (now, default)
        return default

    def reload(self, name: Optional[str] = None):
        """强制重新加载 prompt（用于手动热更新）

        Args:
            name: 指定 Agent 名称，None 表示全部重新加载
        """
        if name:
            self._cache.pop(name, None)
            logger.info(f"[PromptManager] 强制重载: {name}")
        else:
            self._cache.clear()
            logger.info("[PromptManager] 全部 prompt 已强制重载")

    def list_versions(self) -> dict[str, str]:
        """列出所有 prompt 文件的版本信息"""
        versions = {}
        for f in self._dir.glob("*.yaml"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                    versions[f.stem] = data.get("version", "unknown")
            except Exception:
                versions[f.stem] = "error"
        return versions


# 全局单例通过 app.deps.get_prompt_manager() 统一管理，
# 此处不再维护独立的惰性初始化。
