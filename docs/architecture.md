# AgentLite — 极简 AI Agent 框架架构设计

## 1. 概述

AgentLite 是一个极简、可移植的 AI Agent 框架。核心哲学是：**每个组件独立、可替换、零魔法**。用约 1000 行核心代码实现完整 Agent 能力。

### 1.1 核心能力矩阵

| 能力 | 描述 | 实现方式 |
|------|------|----------|
| 任务理解 | 将自然语言需求转化为结构化意图 | LLM 驱动的意图解析 |
| 计划制定 | 将复杂任务拆解为有序步骤 | 基于 LLM 的分层规划 |
| 工具选择 | 根据当前状态决定调用哪个工具 | Function Calling / ReAct 模式 |
| 结果总结 | 将工具输出合成为可读回复 | LLM 驱动的总结生成 |
| 短期记忆 | 当前对话上下文 + 工作状态 | 滑动窗口 + Token 管理 |
| 长期记忆 | 跨会话知识持久化 | 向量存储 + 摘要存储 |
| 任务拆解 | 复杂任务的递归分解 | Planner 模块 |
| 子 Agent | 隔离执行子任务的轻量 Agent | SubAgent 管理器 |
| RAG | 基于外部文档的检索增强生成 | 文档摄取 → 分块 → 嵌入 → 检索 |
| LLM 配置 | 多 Provider 支持 | 统一 LLM 接口抽象 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Core                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Main Loop (Orchestrator)             │   │
│  │                                                     │   │
│  │   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────────┐   │   │
│  │   │Perceive│──▶│ Plan │──▶│Decide│──▶│ Execute  │   │   │
│  │   └──────┘   └──────┘   └──────┘   └──────────┘   │   │
│  │       ▲                                    │        │   │
│  │       │         ┌──────┐                  │        │   │
│  │       └─────────│Reflect│◀─────────────────┘        │   │
│  │                 └──────┘                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│     ┌──────────────────────┼──────────────────────┐         │
│     │                      │                      │         │
│     ▼                      ▼                      ▼         │
│ ┌─────────┐    ┌──────────────┐    ┌──────────────────┐    │
│ │ Memory   │    │  Tool System  │    │  LLM Interface   │    │
│ │ System   │    │  (Registry +  │    │  (Unified API)   │    │
│ │          │    │   Executor)   │    │                  │    │
│ └─────────┘    └──────────────┘    └──────────────────┘    │
│     │                                                       │
│     ▼                                                       │
│ ┌──────────────────────────────────────────────────────┐    │
│ │                   Subsystem                           │    │
│ │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │    │
│ │  │Planner │  │SubAgent│  │  RAG   │  │ Config   │  │    │
│ │  │        │  │Manager │  │ System │  │ Manager  │  │    │
│ │  └────────┘  └────────┘  └────────┘  └──────────┘  │    │
│ └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
User Input (自然语言)
    │
    ▼
┌─────────────┐
│   Agent.run()│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ 1. Perceive     │  加载短期记忆 + 检索长期记忆 + 加载 RAG 上下文
│    (收集上下文)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 2. Plan         │  Planner 将复杂任务拆解为步骤序列
│    (制定计划)    │  输出: Plan { steps: [Step, ...] }
└──────┬──────────┘
       │
       ▼
┌─────────────────┐     ┌─────────────────────────┐
│ 3. Decide       │────▶│ LLM 推理                 │
│    (决策)        │     │ • 分析当前状态           │
│                  │◀────│ • 决定: ToolCall | Reply │
└──────┬──────────┘     └─────────────────────────┘
       │
       ├── ToolCall ──▶ ┌──────────────┐
       │                │ 4. Execute    │  执行工具 / 启动 SubAgent
       │                │    (执行)     │
       │                └──────┬───────┘
       │                       │
       │                       ▼
       │                ┌──────────────┐
       │                │ 5. Observe   │  收集工具输出 / SubAgent 结果
       │                │    (观察)    │
       │                └──────┬───────┘
       │                       │
       │                       ▼
       │                ┌──────────────┐
       │                │ 6. Reflect   │  更新记忆, 判断是否继续
       │                │    (反思)    │
       │                └──────┬───────┘
       │                       │
       │                       ▼
       │                  回到 Step 3 (循环)
       │
       └── Reply ──▶ ┌──────────────┐
                     │ 7. Summarize │  生成最终回复
                     │    (总结)    │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ 8. Memorize  │  存入长期记忆
                     │    (记忆)    │
                     └──────┬───────┘
                            │
                            ▼
                     最终输出 (给用户)
```

---

## 3. 模块详细设计

### 3.1 LLM Interface (`llm.py`)

**职责**: 统一不同 LLM Provider 的调用接口。

```
┌──────────────────────────────────┐
│         LLMInterface             │
├──────────────────────────────────┤
│ + chat(messages) -> Response     │  非流式对话
│ + chat_stream(messages) -> Iter  │  流式对话
│ + embed(texts) -> List[Vector]   │  文本嵌入
│ + count_tokens(text) -> int      │  Token 计数
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│         LLMConfig                │
├──────────────────────────────────┤
│ + provider: str      # openai    │
│ + model: str         # gpt-4o    │
│ + api_key: str                   │
│ + base_url: str                  │
│ + temperature: float             │
│ + max_tokens: int                │
│ + embedding_model: str           │
└──────────────────────────────────┘
```

**设计要点**:
- 使用 OpenAI-compatible API 作为标准协议（绝大多数 Provider 都兼容）
- `base_url` 可指向任何兼容服务（Ollama, vLLM, DeepSeek 等）
- 支持 Function Calling 格式的工具定义
- 嵌入接口与对话接口分离，便于使用不同的嵌入模型

**数据格式** (OpenAI-compatible):

```python
# 请求
{
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": null, "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "xxx", "content": "file content..."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }
        }
    ]
}
```

---

### 3.2 Memory System (`memory.py`)

**双层记忆架构**:

```
┌─────────────────────────────────────────────────────┐
│                   Memory System                      │
├─────────────────────────────────────────────────────┤
│  Short-Term Memory (会话级)                          │
│  ┌─────────────────────────────────────────────┐    │
│  │ messages: List[Message]   # 对话历史         │    │
│  │ working: Dict[str, Any]   # 当前工作状态      │    │
│  │ max_tokens: int           # Token 上限       │    │
│  │                                             │    │
│  │ 策略: 超过上限时，保留 system + 最近 N 条     │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Long-Term Memory (跨会话)                           │
│  ┌─────────────────────────────────────────────┐    │
│  │ VectorStore (语义记忆)                       │    │
│  │   - store: List[(id, vector, metadata)]     │    │
│  │   - search(query_vector, k) -> List[...]    │    │
│  │                                             │    │
│  │ KeyValueStore (事实记忆)                     │    │
│  │   - store: Dict[str, Any]                   │    │
│  │   - get/set/delete                          │    │
│  │                                             │    │
│  │ EpisodeStore (情节记忆)                      │    │
│  │   - episodes: List[EpisodeSummary]          │    │
│  │   - 每次会话结束后自动总结并存储              │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**短期记忆截断策略**:

```
┌────────────────────────────────────────┐
│ 总 Token > max_tokens?                 │
│   ├── 保留 system message (始终)       │
│   ├── 保留最近 working_memory 引用      │
│   └── 从旧到新删除，直到满足限制        │
└────────────────────────────────────────┘
```

**向量存储（极简实现）**:

```python
# 使用余弦相似度, 纯 Python + NumPy
class VectorStore:
    def __init__(self):
        self.vectors = []      # List[np.ndarray]
        self.metadata = []     # List[dict]

    def add(self, vector, metadata):
        self.vectors.append(vector)
        self.metadata.append(metadata)

    def search(self, query_vector, k=5):
        # 余弦相似度计算
        scores = cosine_similarity(query_vector, self.vectors)
        # 返回 top-k
        indices = argsort(scores)[-k:][::-1]
        return [(self.metadata[i], scores[i]) for i in indices]
```

**持久化**: JSON 文件 + NPY 文件，或可选 SQLite。

---

### 3.3 Tool System (`tools.py`)

**工具抽象**:

```python
Tool = {
    "name": str,           # 唯一标识
    "description": str,    # LLM 可理解的描述
    "parameters": {        # JSON Schema
        "type": "object",
        "properties": {...},
        "required": [...]
    },
    "function": Callable,  # 实际执行函数
    "dangerous": bool,     # 是否需要确认
}
```

**内置工具清单**:

| 工具 | 描述 | 参数 | 危险 |
|------|------|------|------|
| `read_file` | 读取文件内容 | path: str, offset?: int, limit?: int | 否 |
| `write_file` | 写入文件 | path: str, content: str | 是 |
| `list_dir` | 列出目录 | path: str | 否 |
| `shell_cmd` | 执行 Shell 命令 | command: str, cwd?: str, timeout?: int | 是 |
| `python_exec` | 执行 Python 代码 | code: str | 是 |
| `search_files` | 搜索文件内容 | pattern: str, path?: str, include?: str | 否 |
| `web_search` | 网络搜索 | query: str | 否 |
| `spawn_subagent` | 启动子 Agent | task: str, tools?: List[str] | 否 |

**工具注册与发现**:

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None
    def unregister(self, name: str) -> None
    def get(self, name: str) -> Tool
    def list_all(self) -> List[Tool]
    def to_openai_schema(self) -> List[dict]  # 转为 OpenAI function schema
```

---

### 3.4 Planner (`planner.py`)

**规划流程**:

```
输入: 用户任务 (自然语言)
    │
    ▼
┌─────────────────────┐
│ 分析任务复杂度       │
│ (简单 → 直接执行)    │
│ (复杂 → 分解规划)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ LLM 生成计划         │
│ Prompt:              │
│ "将以下任务分解为    │
│  有序步骤，每个步骤  │
│  标注所需工具"       │
└────────┬────────────┘
         │
         ▼
输出: Plan
{
  "goal": "原始目标",
  "steps": [
    {
      "id": 1,
      "description": "读取配置文件",
      "tool": "read_file",
      "depends_on": [],
      "status": "pending"
    },
    {
      "id": 2,
      "description": "修改配置",
      "tool": "write_file",
      "depends_on": [1],
      "status": "pending"
    },
    ...
  ]
}
```

**计划执行模式**:
- **Sequential**: 步骤顺序执行
- **Conditional**: 根据上一步结果分支
- **Parallel**: 无依赖步骤并行执行

---

### 3.5 RAG System (`rag.py`)

**完整流水线**:

```
┌──────────────────────────────────────────────────────────┐
│                     RAG Pipeline                          │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐ │
│  │ Document │──▶│  Text    │──▶│ Embedding│──▶│ Vector │ │
│  │ Loader   │   │ Splitter │   │  Model   │   │ Store  │ │
│  └──────────┘   └──────────┘   └──────────┘   └───────┘ │
│                                                      │    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │    │
│  │  Query   │──▶│ Embedding│──▶│ Retrieve │◀────────┘    │
│  └──────────┘   └──────────┘   └──────────┘              │
│                                      │                    │
│                                      ▼                    │
│                              ┌──────────────┐            │
│                              │ Context + LLM │            │
│                              │   Generate    │            │
│                              └──────────────┘            │
└──────────────────────────────────────────────────────────┘
```

**文档加载器**:

支持格式: `.txt`, `.md`, `.py`, `.json`, `.csv`
通过 MIME 类型自动选择加载器。

**文本分割器**:

```python
class TextSplitter:
    """
    递归字符分割策略:
    1. 优先按段落 (\n\n) 分割
    2. 段落太长则按句子 (\n) 分割
    3. 句子太长则按字符截断，保留重叠
    """
    chunk_size: int = 1000    # 每块最大字符数
    chunk_overlap: int = 200  # 块间重叠字符数
    separators: List[str] = ["\n\n", "\n", "。", ". ", " "]
```

**检索流程**:

```python
def retrieve(query: str, k: int = 5) -> List[Chunk]:
    """
    1. 将 query 嵌入为向量
    2. 在 VectorStore 中搜索 top-k 相似块
    3. 按相似度排序返回
    4. 可选: 使用 MMR (Maximal Marginal Relevance) 去重
    """
```

**使用方式**:

```python
# 摄入文档
agent.rag.ingest("docs/", glob="*.md")

# 查询时自动检索
response = agent.run("根据文档，如何配置数据库？")
# Agent 自动: query → embed → retrieve → 注入上下文 → LLM 生成
```

---

### 3.6 Sub-Agent (`subagent.py`)

**设计原则**:
- 子 Agent 是主 Agent 的轻量克隆
- 拥有独立的短期记忆
- 可以访问父 Agent 的部分工具
- 返回结构化结果给父 Agent

```
┌────────────────────────────────────────┐
│           SubAgentManager              │
├────────────────────────────────────────┤
│ + spawn(task, tools, context) -> id    │
│ + wait(id, timeout) -> Result          │
│ + cancel(id)                           │
│ + list_active() -> List[id]            │
└────────────────────────────────────────┘
         │
         │ 创建
         ▼
┌────────────────────────────────────────┐
│           SubAgent Instance            │
├────────────────────────────────────────┤
│ - id: str                             │
│ - task: str                           │
│ - memory: ShortTermMemory (隔离)      │
│ - tools: List[str] (受限)             │
│ - status: running | done | failed     │
│ - result: Any                         │
│ - parent: Agent (只读引用)            │
└────────────────────────────────────────┘
```

**执行模式**:
- **同步**: 父 Agent 等待子 Agent 完成
- **异步**: 父 Agent 启动后继续，稍后收集结果

**通信协议**:
- 父 → 子: task + initial context
- 子 → 父: Result { summary, data, errors }

---

### 3.7 Agent Core (`agent.py`)

**主循环伪代码**:

```python
class Agent:
    def run(self, task: str) -> str:
        # 1. Perceive
        context = self._gather_context(task)

        # 2. Plan
        if self._is_complex(task):
            plan = self.planner.plan(task, context)
        else:
            plan = None  # 直接执行

        # 3-6. Main Loop
        while not self._is_done():
            # Decide
            response = self.llm.chat(
                messages=self.memory.get_messages(),
                tools=self.tools.to_schema()
            )

            if response.is_tool_call():
                # Execute
                result = self._execute_tool(response.tool_call)
                # Observe & Reflect
                self.memory.add_tool_result(result)
                self._update_plan(plan, result)
            else:
                # Final reply
                self.memory.add_assistant(response.content)
                break

        # 7. Summarize
        summary = self._summarize()

        # 8. Memorize
        self.memory.commit_to_long_term(summary)

        return summary
```

---

## 4. 配置系统

```yaml
# config.yaml / 环境变量
agentlite:
  llm:
    provider: openai           # LLM_PROVIDER
    model: gpt-4o              # LLM_MODEL
    api_key: ${LLM_API_KEY}    # LLM_API_KEY
    base_url: https://api.openai.com/v1  # LLM_BASE_URL
    temperature: 0.7
    max_tokens: 4096
    embedding_model: text-embedding-3-small

  memory:
    short_term_max_tokens: 8000
    long_term:
      vector_store: memory/vectors
      kv_store: memory/kv.json
      episodes: memory/episodes.json

  tools:
    allowed: [read_file, write_file, shell_cmd, python_exec, search_files]
    dangerous_require_confirm: true

  rag:
    chunk_size: 1000
    chunk_overlap: 200
    top_k: 5

  subagent:
    max_depth: 3               # 最大递归深度
    default_timeout: 300       # 秒
    max_concurrent: 5
```

---

## 5. 实现指南（语言无关）

### 5.1 各语言实现要点

| 组件 | Python | Go | TypeScript | Rust |
|------|--------|----|-----------|------|
| LLM 接口 | `openai` 包 | `go-openai` | `openai` npm | `async-openai` |
| HTTP 客户端 | `requests`/`httpx` | `net/http` | `fetch` | `reqwest` |
| 向量计算 | `numpy` | `gonum` | `ml-matrix` | `ndarray` |
| JSON Schema | `pydantic`/`dataclasses` | struct tags | `zod` | `serde` |
| 并发 | `asyncio` | goroutines | Promise/async | tokio |
| 嵌入 | LLM API / `sentence-transformers` | LLM API | LLM API | LLM API |

### 5.2 关键设计决策

1. **协议优先**: 所有 LLM 调用走 OpenAI-compatible API，保证最大兼容性
2. **零魔法**: 每个组件可独立使用和测试
3. **渐进增强**: 核心极简，高级功能按需加载
4. **显式状态**: 所有状态通过 Memory 系统管理，不隐式存储

---

## 6. 目录结构

```
agentlite/
├── __init__.py          # 公开 API
├── config.py            # 配置管理
├── types.py             # 共享类型定义
├── llm.py               # LLM 统一接口
├── memory.py            # 短期 + 长期记忆
├── tools.py             # 工具注册 + 内置工具
├── planner.py           # 任务规划器
├── rag.py               # RAG 系统
├── subagent.py          # 子 Agent 管理
├── agent.py             # 主 Agent 编排器
│
examples/
├── demo.py              # 基本使用示例
│
docs/
├── architecture.md      # 本文档
│
tests/                   # (可选) 单元测试
├── test_llm.py
├── test_memory.py
└── ...
```
