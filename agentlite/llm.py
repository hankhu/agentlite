"""
AgentLite — LLM 统一接口

支持所有 OpenAI-compatible API 的 Provider:
- OpenAI, Anthropic (via proxy), Ollama, vLLM, DeepSeek, Groq, 等
"""

import json
import time
import logging
from typing import Any, Dict, Iterable, List, Optional

from .config import LLMConfig
from .types import LLMResponse, Message, ToolCall, ToolDef, Usage

logger = logging.getLogger("agentlite.llm")


class LLMInterface:
    """统一的 LLM 调用接口"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._embed_client = None

    @property
    def client(self):
        """懒加载 OpenAI client (for chat)"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "请安装 openai 包: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    @property
    def embed_client(self):
        """嵌入专用的 OpenAI client (可独立配置 base_url/api_key)"""
        if self._embed_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "请安装 openai 包: pip install openai"
                )
            base = self.config.embedding_base_url or self.config.base_url
            key = self.config.embedding_api_key or self.config.api_key
            self._embed_client = OpenAI(
                api_key=key,
                base_url=base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._embed_client

    # ── Chat ─────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """发送消息，获取响应（非流式）"""
        kwargs = self._build_request(messages, tools, temperature, max_tokens)

        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return self._parse_response(resp)
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("unreachable")

    def chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        """流式对话，yield 文本片段"""
        kwargs = self._build_request(messages, tools, temperature, max_tokens)
        kwargs["stream"] = True

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ── Embedding ────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入"""
        if isinstance(texts, str):
            texts = [texts]

        resp = self.embed_client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        # 按输入顺序返回
        embeddings = sorted(resp.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]

    def embed_one(self, text: str) -> List[float]:
        """单个文本嵌入"""
        return self.embed([text])[0]

    # ── Token 计数 (估算) ────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """估算 token 数。优先使用 tiktoken，否则粗略估算。"""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # 粗略估算: 英文 1 token ≈ 4 chars, 中文 1 token ≈ 1.5 chars
            chars = len(text)
            # 简单 heuristic
            return max(1, chars // 3)

    def count_message_tokens(self, messages: List[Message]) -> int:
        """估算消息列表的总 token 数"""
        total = 0
        for msg in messages:
            if msg.content:
                total += self.count_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    total += self.count_tokens(tc.name + args_str)
        # 每条消息约 4 token 的格式开销
        total += len(messages) * 4
        return total

    # ── 内部方法 ─────────────────────────────────────────────

    def _build_request(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict:
        kwargs = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
        }
        if max_tokens or self.config.max_tokens:
            kwargs["max_tokens"] = max_tokens or self.config.max_tokens
        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _parse_response(self, resp) -> LLMResponse:
        """解析 OpenAI response 为统一的 LLMResponse"""
        choice = resp.choices[0]
        finish = choice.finish_reason or "stop"

        content = choice.message.content
        tool_calls = None
        # DeepSeek 推理模型会在 message 里附上 reasoning_content
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        # Token 使用统计
        usage = None
        if hasattr(resp, "usage") and resp.usage is not None:
            usage = Usage(
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
                completion_tokens=getattr(resp.usage, "completion_tokens", 0),
                total_tokens=getattr(resp.usage, "total_tokens", 0),
            )

        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
            reasoning_content=reasoning_content,
            usage=usage,
        )
