"""
AgentLite — 记忆系统

双层架构:
- ShortTermMemory: 当前对话上下文，Token 感知截断
- LongTermMemory: 向量存储 + KV 存储 + 情节存储，跨会话持久化
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .config import MemoryConfig
from .types import Message, Role

logger = logging.getLogger("agentlite.memory")


# ── Short-Term Memory ───────────────────────────────────────

class ShortTermMemory:
    """短期记忆：当前会话的对话历史 + 工作状态"""

    def __init__(self, config: MemoryConfig,
                 token_counter=None):
        self.config = config
        self.token_counter = token_counter  # LLMInterface.count_tokens 或估算函数
        self.messages: List[Message] = []
        self.working: Dict[str, Any] = {}   # 工作状态存储

    def add(self, message: Message) -> None:
        """添加消息，自动触发截断"""
        self.messages.append(message)
        self._truncate_if_needed()

    def add_system(self, content: str) -> None:
        """设置/替换 system message"""
        # 找到并移除旧的 system message
        self.messages = [m for m in self.messages if m.role != Role.SYSTEM]
        self.messages.insert(0, Message.system(content))

    def add_user(self, content: str) -> None:
        self.add(Message.user(content))

    def add_assistant(self, content: str) -> None:
        self.add(Message.assistant(content=content))

    def add_tool_result(self, tool_call_id: str, result: str,
                        name: Optional[str] = None) -> None:
        self.add(Message.tool(tool_call_id=tool_call_id,
                              content=result, name=name))

    def get_messages(self) -> List[Message]:
        """获取当前消息列表"""
        return list(self.messages)

    def get_working(self, key: str, default=None) -> Any:
        return self.working.get(key, default)

    def set_working(self, key: str, value: Any) -> None:
        self.working[key] = value

    def clear(self) -> None:
        """清空短期记忆（保留 system message）"""
        sys_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        self.messages = sys_msgs
        self.working.clear()

    def _truncate_if_needed(self) -> None:
        """Token 感知截断"""
        if self.token_counter is None:
            return
        max_tokens = self.config.short_term_max_tokens
        total = self.token_counter(self.messages)
        if total <= max_tokens:
            return

        # 保留 system message(s)
        sys_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        other_msgs = [m for m in self.messages if m.role != Role.SYSTEM]

        # 从旧到新删除非 system 消息
        while other_msgs and self.token_counter(sys_msgs + other_msgs) > max_tokens:
            other_msgs.pop(0)

        self.messages = sys_msgs + other_msgs
        logger.debug(f"Truncated memory to {len(self.messages)} messages "
                     f"({self.token_counter(self.messages)} tokens)")


# ── Vector Store (极简实现) ──────────────────────────────────

class VectorStore:
    """基于 NumPy 的向量存储，使用余弦相似度检索"""

    def __init__(self):
        self.vectors: List["np.ndarray"] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vector, metadata: Dict[str, Any]) -> None:
        import numpy as np
        self.vectors.append(np.array(vector, dtype=np.float32))
        self.metadata.append(metadata)

    def add_batch(self, vectors, metadatas: List[Dict[str, Any]]) -> None:
        for v, m in zip(vectors, metadatas):
            self.add(v, m)

    def search(self, query_vector, k: int = 5) -> List[Tuple[Dict, float]]:
        """返回 [(metadata, score), ...] 按相似度降序"""
        import numpy as np
        if not self.vectors:
            return []
        query = np.array(query_vector, dtype=np.float32)
        stack = np.stack(self.vectors)
        # 余弦相似度
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        stack_norm = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(stack_norm, query_norm)
        top_k = min(k, len(scores))
        indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.metadata[i], float(scores[i])) for i in indices]

    def delete(self, predicate) -> int:
        """删除满足 predicate(metadata) 的条目，返回删除数"""
        keep_idx = [i for i, m in enumerate(self.metadata) if not predicate(m)]
        removed = len(self.vectors) - len(keep_idx)
        import numpy as np
        self.vectors = [self.vectors[i] for i in keep_idx]
        self.metadata = [self.metadata[i] for i in keep_idx]
        return removed

    def __len__(self) -> int:
        return len(self.vectors)

    def save(self, path: str) -> None:
        """持久化到磁盘"""
        import numpy as np
        os.makedirs(path, exist_ok=True)
        if self.vectors:
            np.save(os.path.join(path, "vectors.npy"),
                    np.stack(self.vectors))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从磁盘加载"""
        import numpy as np
        vec_path = os.path.join(path, "vectors.npy")
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            arr = np.load(vec_path)
            self.vectors = [arr[i] for i in range(len(arr))]
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)


# ── Key-Value Store ─────────────────────────────────────────

class KeyValueStore:
    """简单的 KV 存储，支持持久化"""

    def __init__(self, path: Optional[str] = None):
        self._store: Dict[str, Any] = {}
        self._path = path
        if path and os.path.exists(path):
            self._load()

    def get(self, key: str, default=None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._save()

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._save()

    def search(self, query: str) -> List[Tuple[str, Any]]:
        """简单子串匹配（不依赖嵌入）"""
        results = []
        for k, v in self._store.items():
            if query.lower() in k.lower() or query.lower() in str(v).lower():
                results.append((k, v))
        return results

    def all(self) -> Dict[str, Any]:
        return dict(self._store)

    def _save(self) -> None:
        if self._path:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        try:
            with open(self._path, "r") as f:
                content = f.read().strip()
                if content:
                    self._store = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            self._store = {}


# ── Long-Term Memory ────────────────────────────────────────

class LongTermMemory:
    """长期记忆：融合向量存储、KV 存储、情节存储"""

    def __init__(self, config: MemoryConfig,
                 llm_interface=None):
        self.config = config
        self.llm = llm_interface        # 用于嵌入和总结
        base = config.persist_dir

        self.vector_store = VectorStore()
        self.kv_store = KeyValueStore(os.path.join(base, "kv.json"))
        self.episodes: List[Dict[str, Any]] = []

        # 尝试加载已有数据
        self._load(base)
        self.base = base

    # ── 语义记忆 (Vector) ────────────────────────────────

    def remember(self, text: str, metadata: Optional[Dict] = None) -> None:
        """存入一条语义记忆"""
        if self.llm is None:
            return
        try:
            vec = self.llm.embed_one(text)
        except Exception as e:
            logger.warning(f"Embedding failed in remember: {e}")
            return
        meta = metadata or {}
        meta["text"] = text
        meta["timestamp"] = import_time()
        self.vector_store.add(vec, meta)
        self._save()

    def recall(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """语义检索"""
        if self.llm is None or len(self.vector_store) == 0:
            return []
        try:
            vec = self.llm.embed_one(query)
        except Exception as e:
            logger.warning(f"Embedding failed in recall: {e}")
            return []
        return self.vector_store.search(vec, k=k)

    def recall_as_context(self, query: str, k: int = 5) -> str:
        """检索结果格式化为 LLM 上下文"""
        results = self.recall(query, k)
        if not results:
            return ""
        lines = ["[Relevant memories]:"]
        for i, (meta, score) in enumerate(results):
            lines.append(f"  {i+1}. {meta.get('text', '')} "
                         f"(relevance: {score:.2f})")
        return "\n".join(lines)

    # ── 事实记忆 (KV) ────────────────────────────────────

    def remember_fact(self, key: str, value: Any) -> None:
        self.kv_store.set(key, value)

    def recall_fact(self, key: str) -> Any:
        return self.kv_store.get(key)

    def forget_fact(self, key: str) -> None:
        self.kv_store.delete(key)

    # ── 情节记忆 (Episode) ───────────────────────────────

    def add_episode(self, task: str, summary: str,
                    success: bool = True) -> None:
        ep = {
            "task": task,
            "summary": summary,
            "success": success,
            "timestamp": import_time(),
        }
        self.episodes.append(ep)
        # 同时添加到语义存储
        self.remember(f"Task: {task}\nResult: {summary}",
                      {"type": "episode", "success": success})
        # 限制情节数量
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]

    def recent_episodes(self, n: int = 5) -> List[Dict]:
        return self.episodes[-n:]

    # ── 持久化 ───────────────────────────────────────────

    def _save(self) -> None:
        base = self.base
        os.makedirs(base, exist_ok=True)
        self.vector_store.save(base)
        # episodes
        with open(os.path.join(base, "episodes.json"), "w") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)

    def _load(self, base: str) -> None:
        self.vector_store.load(base)
        ep_path = os.path.join(base, "episodes.json")
        if os.path.exists(ep_path):
            with open(ep_path, "r") as f:
                self.episodes = json.load(f)


def import_time() -> float:
    import time
    return time.time()
