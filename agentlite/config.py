"""
AgentLite — 配置管理

支持:
- 代码内字典配置
- 环境变量覆盖
- YAML/JSON 文件加载 (可选)
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """LLM Provider 配置"""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = ""        # 空则沿用 base_url
    embedding_api_key: str = ""          # 空则沿用 api_key
    # 额外 HTTP 参数
    timeout: int = 120
    max_retries: int = 3


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    short_term_max_tokens: int = 8000
    # 长期记忆
    persist_dir: str = "./agentlite_memory"
    # 情节记忆
    auto_summarize: bool = True
    summary_trigger_tokens: int = 6000


@dataclass
class ToolConfig:
    """工具系统配置"""
    allowed: List[str] = field(default_factory=lambda: [
        "read_file", "write_file", "list_dir",
        "shell_cmd", "python_exec", "search_files",
        "web_search", "web_fetch",
    ])
    dangerous_require_confirm: bool = True
    shell_timeout: int = 60
    shell_max_output: int = 10000


@dataclass
class RAGConfig:
    """RAG 系统配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    min_score: float = 0.3
    # 分隔符优先级
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", ". ", " "
    ])


@dataclass
class SubAgentConfig:
    """子 Agent 配置"""
    max_depth: int = 3
    default_timeout: int = 300
    max_concurrent: int = 5
    max_iterations: int = 15


@dataclass
class AgentConfig:
    """Agent 总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    subagent: SubAgentConfig = field(default_factory=SubAgentConfig)

    # Agent 行为
    system_prompt: str = (
        "You are a capable AI agent. You can use tools to accomplish tasks. "
        "Think step by step. When you have enough information, "
        "provide a clear and concise final answer."
    )
    max_iterations: int = 20
    verbose: bool = False

    # ── 工厂方法 ─────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """从环境变量构建配置"""
        cfg = cls()
        # LLM
        cfg.llm.api_key = os.environ.get("LLM_API_KEY", cfg.llm.api_key)
        cfg.llm.base_url = os.environ.get("LLM_BASE_URL", cfg.llm.base_url)
        cfg.llm.model = os.environ.get("LLM_MODEL", cfg.llm.model)
        cfg.llm.provider = os.environ.get("LLM_PROVIDER", cfg.llm.provider)
        cfg.llm.embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", cfg.llm.embedding_base_url)
        cfg.llm.embedding_api_key = os.environ.get("EMBEDDING_API_KEY", cfg.llm.embedding_api_key)
        cfg.llm.embedding_model = os.environ.get("EMBEDDING_MODEL", cfg.llm.embedding_model)
        # 通用
        cfg.verbose = os.environ.get("AGENT_VERBOSE", "").lower() == "true"
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        """从字典构建配置"""
        cfg = cls()
        if "llm" in d:
            for k, v in d["llm"].items():
                if hasattr(cfg.llm, k):
                    setattr(cfg.llm, k, v)
        if "memory" in d:
            for k, v in d["memory"].items():
                if hasattr(cfg.memory, k):
                    setattr(cfg.memory, k, v)
        if "tools" in d:
            for k, v in d["tools"].items():
                if hasattr(cfg.tools, k):
                    setattr(cfg.tools, k, v)
        if "rag" in d:
            for k, v in d["rag"].items():
                if hasattr(cfg.rag, k):
                    setattr(cfg.rag, k, v)
        if "subagent" in d:
            for k, v in d["subagent"].items():
                if hasattr(cfg.subagent, k):
                    setattr(cfg.subagent, k, v)
        if "system_prompt" in d:
            cfg.system_prompt = d["system_prompt"]
        if "max_iterations" in d:
            cfg.max_iterations = d["max_iterations"]
        if "verbose" in d:
            cfg.verbose = d["verbose"]
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "AgentConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "embedding_model": self.llm.embedding_model,
            },
            "memory": {
                "short_term_max_tokens": self.memory.short_term_max_tokens,
                "persist_dir": self.memory.persist_dir,
            },
            "tools": {
                "allowed": self.tools.allowed,
                "dangerous_require_confirm": self.tools.dangerous_require_confirm,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "top_k": self.rag.top_k,
            },
            "subagent": {
                "max_depth": self.subagent.max_depth,
                "default_timeout": self.subagent.default_timeout,
            },
            "system_prompt": self.system_prompt,
            "max_iterations": self.max_iterations,
        }
