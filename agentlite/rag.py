"""
AgentLite — RAG (Retrieval-Augmented Generation) 系统

完整的文档摄取 → 分块 → 嵌入 → 存储 → 检索流水线。
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import RAGConfig
from .types import Chunk, SearchResult

logger = logging.getLogger("agentlite.rag")


# ── Text Splitter ───────────────────────────────────────────

class TextSplitter:
    """递归字符文本分割器。

    按优先级尝试分隔符:
    1. 段落 (\\n\\n)
    2. 行 (\\n)
    3. 句子 (。. )
    4. 字符强制截断
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.separators = self.config.separators

    def split(self, text: str) -> List[str]:
        """将文本分割为块"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str,
                         separators: List[str]) -> List[str]:
        if not separators:
            # 强制按 chunk_size 截断
            return self._split_by_chars(text)

        sep = separators[0]
        remaining = separators[1:]

        if sep not in text:
            return self._split_recursive(text, remaining)

        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 处理过长的单个部分
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_recursive(part, remaining)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # 添加重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev = chunks[i - 1]
                    overlap = prev[-self.chunk_overlap:]
                    chunk = overlap + chunk
                overlapped.append(chunk)
            chunks = overlapped

        return chunks

    def _split_by_chars(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


# ── Document Loader ─────────────────────────────────────────

class DocumentLoader:
    """文档加载器：支持 txt, md, py, json, csv 等格式"""

    @staticmethod
    def load(path: str) -> str:
        """加载单个文件为文本"""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if ext == ".json":
            # 美化 JSON
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return content
        return content

    @staticmethod
    def load_directory(root: str, glob_pattern: str = "**/*") -> List[Dict]:
        """加载目录下所有匹配文件。返回 [{path, content}, ...]"""
        import glob as glob_mod
        import fnmatch

        docs = []
        for fpath in glob_mod.glob(os.path.join(root, glob_pattern),
                                   recursive=True):
            if not os.path.isfile(fpath):
                continue
            # 只处理文本文件
            ext = os.path.splitext(fpath)[1].lower()
            if ext in {".txt", ".md", ".py", ".js", ".ts",
                       ".json", ".yaml", ".yml", ".toml",
                       ".csv", ".html", ".css", ".sql", ".sh",
                       ".go", ".rs", ".java", ".c", ".cpp", ".h",
                       ".xml", ".cfg", ".ini", ".env", ""}:
                try:
                    content = DocumentLoader.load(fpath)
                    docs.append({"path": fpath, "content": content})
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")
        return docs


# ── RAG Engine ──────────────────────────────────────────────

class RAGEngine:
    """RAG 引擎：编排文档摄取和检索"""

    def __init__(self, config: Optional[RAGConfig] = None,
                 llm_interface=None):
        self.config = config or RAGConfig()
        self.llm = llm_interface
        self.splitter = TextSplitter(self.config)
        self.chunks: List[Chunk] = []
        # 去重
        self._seen_hashes: set = set()
        # 反向索引
        self._file_index: Dict[str, List[str]] = {}  # file_path -> chunk_ids

    # ── Ingestion ─────────────────────────────────────────

    def ingest_file(self, path: str) -> int:
        """摄入单个文件。返回新增 chunk 数。"""
        content = DocumentLoader.load(path)
        return self._ingest_text(content, {"source": path, "type": "file"})

    def ingest_directory(self, root: str,
                         glob_pattern: str = "**/*") -> int:
        """摄入目录。返回新增 chunk 数。"""
        docs = DocumentLoader.load_directory(root, glob_pattern)
        total = 0
        for doc in docs:
            n = self._ingest_text(
                doc["content"],
                {"source": doc["path"], "type": "file"}
            )
            total += n
        return total

    def ingest_text(self, text: str,
                    metadata: Optional[Dict] = None) -> int:
        """摄入原始文本。返回新增 chunk 数。"""
        return self._ingest_text(text, metadata or {})

    def _ingest_text(self, text: str, metadata: Dict) -> int:
        """内部摄入实现"""
        if not text.strip():
            return 0
        raw_chunks = self.splitter.split(text)
        new_count = 0
        for i, raw in enumerate(raw_chunks):
            chunk_id = hashlib.md5(raw.encode()).hexdigest()[:12]
            if chunk_id in self._seen_hashes:
                continue
            self._seen_hashes.add(chunk_id)

            meta = dict(metadata)
            meta["chunk_index"] = i
            meta["chunk_id"] = chunk_id

            chunk = Chunk(id=chunk_id, text=raw, metadata=meta)
            self.chunks.append(chunk)

            # 更新文件索引
            src = metadata.get("source", "")
            if src not in self._file_index:
                self._file_index[src] = []
            self._file_index[src].append(chunk_id)

            new_count += 1
        return new_count

    def embed_all(self, progress_callback=None) -> int:
        """对所有未嵌入的 chunk 计算嵌入。返回嵌入数量。"""
        if self.llm is None:
            return 0

        unembedded = [c for c in self.chunks if c.embedding is None]
        if not unembedded:
            return 0

        total = len(unembedded)
        batch_size = 20  # OpenAI 推荐批次大小

        for start in range(0, total, batch_size):
            batch = unembedded[start:start + batch_size]
            texts = [c.text for c in batch]
            try:
                vectors = self.llm.embed(texts)
                for chunk, vec in zip(batch, vectors):
                    chunk.embedding = vec
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                raise

            if progress_callback:
                progress_callback(start + len(batch), total)

        return total

    # ── Retrieval ─────────────────────────────────────────

    def search(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        """检索相关 chunks。自动嵌入查询。"""
        if self.llm is None:
            return self._keyword_search(query, k)

        k = k or self.config.top_k
        if not self.chunks:
            return []

        # 确保有嵌入
        embedded = [c for c in self.chunks if c.embedding is not None]
        if not embedded:
            try:
                self.embed_all()
                embedded = [c for c in self.chunks if c.embedding is not None]
            except Exception as e:
                logger.warning(f"Embed all failed: {e}, using keyword fallback")
                embedded = []
        if not embedded:
            return self._keyword_search(query, k)

        # 嵌入查询
        try:
            query_vec = self.llm.embed_one(query)
        except Exception as e:
            logger.warning(f"Query embedding failed, falling back to keyword: {e}")
            return self._keyword_search(query, k)

        # 余弦相似度搜索
        import numpy as np
        q = np.array(query_vec, dtype=np.float32)
        vectors = np.array([c.embedding for c in embedded], dtype=np.float32)

        q_norm = q / (np.linalg.norm(q) + 1e-10)
        v_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(v_norm, q_norm)

        top_k = min(k, len(scores))
        indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for i in indices:
            score = float(scores[i])
            if score >= self.config.min_score:
                results.append(SearchResult(chunk=embedded[i], score=score))
        return results

    def search_as_context(self, query: str, k: Optional[int] = None) -> str:
        """检索并格式化为 LLM 上下文"""
        results = self.search(query, k)
        if not results:
            return ""

        lines = ["[Relevant documents]:"]
        for i, r in enumerate(results):
            src = r.chunk.metadata.get("source", "unknown")
            lines.append(f"\n--- Source: {src} "
                         f"(relevance: {r.score:.2f}) ---")
            lines.append(r.chunk.text[:1500])
        return "\n".join(lines)

    def _keyword_search(self, query: str,
                        k: Optional[int] = None) -> List[SearchResult]:
        """关键词搜索（支持中日韩文，无空格语言的词边界检测）"""
        k = k or self.config.top_k

        # 对无空格语言（中文、日文等）做 char n-gram + 原词混合
        query_terms = self._tokenize(query)

        scored = []
        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            # 简单的 TF 评分
            score = sum(text_lower.count(t) for t in query_terms)
            if score > 0:
                # 归一化
                score = score / (len(chunk.text) + 1)
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    @staticmethod
    def _tokenize(text: str) -> list:
        """分词：英文按空格，中日韩文拆成单字 + 2-gram"""
        text = text.lower()
        tokens = set()

        # 英文/数字 token（空格分隔）
        import re
        words = re.findall(r"[a-z0-9]+", text)
        tokens.update(words)

        # 中日韩文：拆成单字 + 2-gram
        cjk = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff"
                         r"\uac00-\ud7af]+", text)
        for chunk in cjk:
            # 单字
            for ch in chunk:
                tokens.add(ch)
            # 2-gram
            for i in range(len(chunk) - 1):
                tokens.add(chunk[i:i+2])

        return list(tokens)

    # ── Management ────────────────────────────────────────

    def clear(self) -> None:
        """清空所有数据"""
        self.chunks.clear()
        self._seen_hashes.clear()
        self._file_index.clear()

    def remove_file(self, path: str) -> int:
        """移除某个文件的所有 chunks。返回移除数。"""
        chunk_ids = self._file_index.pop(path, [])
        id_set = set(chunk_ids)
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.id not in id_set]
        self._seen_hashes -= id_set
        return before - len(self.chunks)

    def stats(self) -> Dict[str, Any]:
        """返回统计信息"""
        embedded = sum(1 for c in self.chunks if c.embedding is not None)
        return {
            "total_chunks": len(self.chunks),
            "embedded_chunks": embedded,
            "indexed_files": len(self._file_index),
            "total_chars": sum(len(c.text) for c in self.chunks),
        }

    # ── Persistence ───────────────────────────────────────

    def save(self, path: str) -> None:
        """保存 RAG 状态到磁盘"""
        import numpy as np
        os.makedirs(path, exist_ok=True)

        data = []
        vectors = []
        for chunk in self.chunks:
            data.append({
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
            if chunk.embedding:
                vectors.append(chunk.embedding)

        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if vectors:
            np.save(os.path.join(path, "embeddings.npy"),
                    np.array(vectors, dtype=np.float32))
        with open(os.path.join(path, "hashes.json"), "w") as f:
            json.dump(list(self._seen_hashes), f)
        with open(os.path.join(path, "file_index.json"), "w") as f:
            json.dump(self._file_index, f)

    def load(self, path: str) -> None:
        """从磁盘加载 RAG 状态"""
        import numpy as np

        chunks_path = os.path.join(path, "chunks.json")
        emb_path = os.path.join(path, "embeddings.npy")
        hashes_path = os.path.join(path, "hashes.json")
        index_path = os.path.join(path, "file_index.json")

        if not os.path.exists(chunks_path):
            return

        with open(chunks_path, "r") as f:
            data = json.load(f)

        embeddings = None
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)

        self.chunks = []
        for i, d in enumerate(data):
            emb = embeddings[i].tolist() if embeddings is not None and i < len(embeddings) else None
            self.chunks.append(Chunk(
                id=d["id"], text=d["text"],
                metadata=d.get("metadata", {}), embedding=emb
            ))

        if os.path.exists(hashes_path):
            with open(hashes_path, "r") as f:
                self._seen_hashes = set(json.load(f))

        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self._file_index = json.load(f)
