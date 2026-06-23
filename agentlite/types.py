"""
AgentLite — 共享类型定义
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional
from enum import Enum


class Role(str):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepStatus(str):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(str):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


# ── Message Types ───────────────────────────────────────────

@dataclass
class Message:
    """统一的消息格式，兼容 OpenAI chat messages"""
    role: str                          # system | user | assistant | tool
    content: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None   # DeepSeek 推理模式专用

    def to_dict(self) -> dict:
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: Optional[str] = None,
                  tool_calls: Optional[List["ToolCall"]] = None,
                  reasoning_content: Optional[str] = None) -> "Message":
        return cls(role="assistant", content=content,
                   tool_calls=tool_calls,
                   reasoning_content=reasoning_content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str,
             name: Optional[str] = None) -> "Message":
        return cls(role="tool", content=content,
                   tool_call_id=tool_call_id, name=name)


# ── Tool Types ──────────────────────────────────────────────

@dataclass
class ToolCall:
    """LLM 返回的工具调用"""
    id: str                            # tool call id
    name: str                          # 工具名称
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        args = self.arguments
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": args,
            }
        }


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_call_id: str
    name: str
    success: bool
    output: Any
    error: Optional[str] = None

    def summary(self) -> str:
        if self.success:
            s = str(self.output)
            return s[:2000] if len(s) > 2000 else s
        return f"Error: {self.error}"


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    description: str
    parameters: dict                  # JSON Schema
    function: Callable                # Python callable
    dangerous: bool = False

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ── Plan Types ──────────────────────────────────────────────

@dataclass
class Step:
    """计划中的一个步骤"""
    id: int
    description: str
    tool: Optional[str] = None         # 推荐使用的工具
    depends_on: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING


@dataclass
class Plan:
    """任务执行计划"""
    goal: str
    steps: List[Step] = field(default_factory=list)

    def current_step(self) -> Optional[Step]:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    def all_done(self) -> bool:
        return all(s.status in (StepStatus.COMPLETED, StepStatus.FAILED)
                   for s in self.steps)


# ── RAG Types ───────────────────────────────────────────────

@dataclass
class Chunk:
    """文档分块"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """检索结果"""
    chunk: Chunk
    score: float


# ── SubAgent Types ──────────────────────────────────────────

@dataclass
class SubAgentResult:
    """子 Agent 执行结果"""
    id: str
    task: str
    status: str                         # done | failed | timeout
    summary: str
    data: Any = None
    error: Optional[str] = None


# ── Token Usage ─────────────────────────────────────────────

@dataclass
class Usage:
    """Token 消耗统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __str__(self) -> str:
        return (f"↑{self.prompt_tokens} ↓{self.completion_tokens} "
                f"∑{self.total_tokens}")


# ── LLM Response Types ──────────────────────────────────────

@dataclass
class LLMResponse:
    """LLM 返回的统一响应"""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"         # stop | tool_calls | length
    reasoning_content: Optional[str] = None  # DeepSeek 推理内容
    usage: Optional[Usage] = None            # token 统计

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def is_final(self) -> bool:
        return self.content is not None and not self.is_tool_call()
