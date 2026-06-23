# AgentLite 项目总结

## 做了什么

构建了一个极简、可移植的 AI Agent 框架 —— **AgentLite**，约 3,400 行 Python 核心代码 + ~600 行语言无关的架构设计文档。

---

## 核心设计理念

- **每个组件独立、可替换、零魔法** — 任何模块都可以单独使用和测试
- **协议优先** — LLM 调用全程走 OpenAI-compatible API，兼容几乎所有 Provider
- **渐进增强** — 核心极简，高级功能按需加载
- **显式状态** — 所有状态通过 Memory 系统管理，不隐式存储

---

## 模块清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `agentlite/types.py` | ~200 | 共享类型：Message、ToolCall、ToolResult、Plan、Step、LLMResponse 等 |
| `agentlite/config.py` | ~180 | 配置管理：支持 dict / JSON 文件 / 环境变量 三种加载方式 |
| `agentlite/llm.py` | ~180 | LLM 统一接口：chat / chat_stream / embed / count_tokens |
| `agentlite/memory.py` | ~310 | 双层记忆：ShortTermMemory（token 截断）+ LongTermMemory（VectorStore + KVStore + Episode） |
| `agentlite/tools.py` | ~390 | 工具系统：ToolRegistry + 6 个内置工具（文件/Shell/代码/搜索） |
| `agentlite/rag.py` | ~410 | RAG 管线：DocumentLoader → TextSplitter → Embed → VectorStore → Retrieval |
| `agentlite/planner.py` | ~220 | 任务规划器：简单启发式 + LLM 分解（JSON 计划） |
| `agentlite/subagent.py` | ~190 | 子 Agent 管理：spawn / wait / cancel，支持深度限制和工具子集 |
| `agentlite/agent.py` | ~420 | 主编排器：Perceive → Plan → Decide → Execute → Observe → Reflect 完整循环 |
| `agentlite/__init__.py` | ~60 | 公开 API 导出 |
| `docs/architecture.md` | ~610 | 语言无关架构设计文档（含数据流图、接口定义、多语言实现指南） |
| `examples/demo.py` | ~260 | 5 个功能演示 |

---

## 核心能力覆盖

| 需求 | 实现方式 |
|------|----------|
| **理解任务** | Agent 主循环 Perceive 阶段 — 收集 RAG 上下文 + 长期记忆 + 用户输入 |
| **制定计划** | `Planner` 模块 — 简单任务直通，复杂任务 LLM 分解为 Step DAG |
| **决定调用哪些工具** | LLM Function Calling（OpenAI 协议），Agent 主循环自动决策 |
| **总结结果** | 自然结束或达到 max_iterations 时强制总结 |
| **工具：文件操作** | `read_file`（支持 offset/limit）、`write_file`（覆盖/追加）、`list_dir` |
| **工具：Shell** | `shell_cmd`（超时控制、输出截断、捕获 stderr） |
| **工具：代码执行** | `python_exec`（表达式求值 + 语句执行，隔离 stdout） |
| **工具：搜索** | `search_files`（正则表达式 grep，递归目录，文件类型过滤） |
| **短期记忆** | `ShortTermMemory` — 消息缓冲区 + token 感知截断 + 工作状态 dict |
| **长期记忆** | `LongTermMemory` — VectorStore（余弦相似度）+ KVStore（持久化）+ Episode（自动总结） |
| **拆解复杂任务** | `Planner.plan()` → `Plan { steps: [Step] }` — 含依赖关系 DAG |
| **执行子 Agent** | `SubAgentManager` — 同步/异步 spawn，工具子集，深度限制，线程池 |
| **RAG 系统** | `RAGEngine` — 文档加载（多格式）→ 递归字符分割 → LLM 嵌入 → 语义检索 + 关键词 fallback |
| **配置 LLM API** | `LLMInterface` — 统一 OpenAI-compatible 协议，支持任意 provider |

---

## 测试结果

全部非 LLM 模块已通过测试：

- ✅ **Types**: Message/ToolCall 序列化往返正确
- ✅ **Config**: dict/JSON/环境变量三种加载方式正常
- ✅ **Memory**: ShortTerm 消息管理、KV 持久化、VectorStore 余弦检索
- ✅ **Tools**: 6 个内置工具全部可注册和执行
- ✅ **RAG**: TextSplitter 分块、关键词检索 fallback
- ✅ **Planner**: 简单/复杂任务识别、Plan 解析
- ✅ **Agent**: 完整构造、工具 hooks 回调

---

## 使用示例

```python
from agentlite import Agent

# 1. 基本使用
agent = Agent(llm_config={
    "model": "gpt-4o",
    "api_key": "sk-...",
})
result = agent.run("列出当前目录下所有 Python 文件")

# 2. RAG 文档检索
agent.ingest("./docs/")              # 摄入文档目录
result = agent.run("如何配置数据库？")  # 自动检索并增强

# 3. 自定义工具
from agentlite import ToolDef
agent.tool_registry.register(ToolDef(
    name="get_weather",
    description="获取城市天气",
    parameters={...},
    function=get_weather_func
))

# 4. 长期记忆
agent.long_term.remember_fact("user_preference", "dark_mode")

# 5. 配置灵活性
# 环境变量: LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
# JSON 文件: AgentConfig.from_json("config.json")
# 字典: AgentConfig.from_dict({...})
# 代码: AgentConfig() 直接设置属性
```

---

## 依赖

- `openai` — LLM API 调用
- `numpy` — 向量存储和相似度计算
- 可选: `tiktoken` — 精确 token 计数（否则自动 fallback 估算）

---

## 可移植性

`docs/architecture.md` 提供了语言无关的架构设计，包含每个模块的接口定义、数据流图和 Python/Go/TypeScript/Rust 各语言的实现要点，便于用其他语言重新实现。
