"""
AgentLite — A minimalist, portable AI Agent framework.

Usage:
    from agentlite import Agent

    agent = Agent(llm_config={
        "model": "gpt-4o",
        "api_key": "sk-...",
    })

    # Basic task
    result = agent.run("List all Python files in the current directory")

    # With RAG
    agent.ingest("./docs/")
    result = agent.run("How do I configure the database?")
"""

import warnings
# 抑制 urllib3 v2 + LibreSSL 旧版警告（无害，仅视觉干扰）
# 不能用 from urllib3.exceptions import NotOpenSSLWarning，因为那会先触发警告
warnings.filterwarnings("ignore", message=".*OpenSSL 1\\.1\\.1\\+.*")

from .agent import Agent
from .config import AgentConfig, LLMConfig, MemoryConfig, ToolConfig, RAGConfig
from .llm import LLMInterface
from .memory import ShortTermMemory, LongTermMemory, VectorStore, KeyValueStore
from .tools import ToolRegistry, ToolDef, build_builtin_tools
from .planner import Planner
from .rag import RAGEngine, TextSplitter, DocumentLoader
from .subagent import SubAgentManager
from .types import (
    Message, ToolCall, ToolResult, Usage, ToolDef as ToolDefType,
    Plan, Step, StepStatus, AgentState,
    LLMResponse, Chunk, SearchResult, SubAgentResult,
)

__version__ = "0.1.2"
__all__ = [
    # Core
    "Agent",
    # Config
    "AgentConfig", "LLMConfig", "MemoryConfig", "ToolConfig", "RAGConfig",
    # LLM
    "LLMInterface",
    # Memory
    "ShortTermMemory", "LongTermMemory", "VectorStore", "KeyValueStore",
    # Tools
    "ToolRegistry", "ToolDef", "build_builtin_tools",
    # Planner
    "Planner",
    # RAG
    "RAGEngine", "TextSplitter", "DocumentLoader",
    # SubAgent
    "SubAgentManager",
    # Types
    "Message", "ToolCall", "ToolResult", "Usage",
    "Plan", "Step", "StepStatus", "AgentState",
    "LLMResponse", "Chunk", "SearchResult", "SubAgentResult",
]
