"""
AgentLite Demo — 展示核心功能

运行:
    python examples/demo.py

需要:
    pip install openai numpy
    export LLM_API_KEY=sk-...
"""

import sys
import os

# 将 agentlite 加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlite import Agent, AgentConfig, ToolDef


# ─────────────────────────────────────────────────────────────
# Demo 1: 基本 Agent 使用
# ─────────────────────────────────────────────────────────────

def demo_basic():
    """基本任务执行"""
    print("=" * 60)
    print("Demo 1: Basic Agent")
    print("=" * 60)

    agent = Agent(
        llm_config={
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "api_key": os.environ.get("LLM_API_KEY", "sk-xxx"),
            "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        },
        verbose=True,
    )

    result = agent.run("列出当前目录下的所有 .py 文件")
    print(f"\nResult:\n{result}")


# ─────────────────────────────────────────────────────────────
# Demo 2: 自定义工具
# ─────────────────────────────────────────────────────────────

def demo_custom_tool():
    """注册自定义工具"""
    print("\n" + "=" * 60)
    print("Demo 2: Custom Tool")
    print("=" * 60)

    def get_weather(city: str) -> str:
        """获取城市天气（模拟）"""
        weather_data = {
            "beijing": "晴天 25°C",
            "shanghai": "多云 28°C",
            "tokyo": "小雨 22°C",
        }
        return weather_data.get(city.lower(), f"Unknown city: {city}")

    weather_tool = ToolDef(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. Beijing, Tokyo)",
                }
            },
            "required": ["city"],
        },
        function=get_weather,
        dangerous=False,
    )

    agent = Agent(
        llm_config={
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "api_key": os.environ.get("LLM_API_KEY", "sk-xxx"),
            "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        },
        tools=[weather_tool],
    )

    result = agent.run("北京今天天气怎么样？")
    print(f"\nResult:\n{result}")


# ─────────────────────────────────────────────────────────────
# Demo 3: RAG 系统
# ─────────────────────────────────────────────────────────────

def demo_rag():
    """文档摄入和检索增强"""
    print("\n" + "=" * 60)
    print("Demo 3: RAG System")
    print("=" * 60)

    # 创建示例文档
    os.makedirs("/tmp/agentlite_docs", exist_ok=True)
    with open("/tmp/agentlite_docs/knowledge.txt", "w") as f:
        f.write("""
AgentLite 是一个极简 AI Agent 框架。

核心特性:
1. 任务理解: 使用 LLM 分析用户意图
2. 计划制定: 将复杂任务拆解为有序步骤
3. 工具选择: 基于 Function Calling 自动选择工具
4. 结果总结: 将执行结果合成为最终回复

配置:
- LLM 支持所有 OpenAI-compatible API
- 记忆系统包含短期和长期存储
- 内置工具: read_file, write_file, shell_cmd, python_exec, search_files
- RAG 支持文档摄入和语义检索
- 子 Agent 支持隔离执行

安装:
pip install agentlite openai numpy

使用方法:
from agentlite import Agent
agent = Agent(llm_config={"model": "gpt-4o", "api_key": "sk-..."})
agent.run("your task here")
        """.strip())

    agent = Agent(
        llm_config={
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "api_key": os.environ.get("LLM_API_KEY", "sk-xxx"),
            "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        },
    )

    # 摄入文档
    count = agent.ingest("/tmp/agentlite_docs/")
    print(f"Ingested {count} chunks")

    # RAG stats
    stats = agent.rag.stats()
    print(f"RAG stats: {stats}")

    # 直接查询 RAG
    context = agent.ask_rag("如何安装 agentlite？")
    print(f"\nRAG context:\n{context[:500]}...")


# ─────────────────────────────────────────────────────────────
# Demo 4: 记忆系统
# ─────────────────────────────────────────────────────────────

def demo_memory():
    """短期和长期记忆"""
    print("\n" + "=" * 60)
    print("Demo 4: Memory System")
    print("=" * 60)

    from agentlite import ShortTermMemory, LongTermMemory, MemoryConfig
    from agentlite.types import Message

    # 短期记忆
    config = MemoryConfig(short_term_max_tokens=1000)
    stm = ShortTermMemory(config)

    stm.add_system("You are a helpful assistant.")
    stm.add_user("What is Python?")
    stm.add_assistant("Python is a programming language.")

    print(f"Short-term messages: {len(stm.get_messages())}")

    # 工作记忆
    stm.set_working("current_file", "/tmp/test.py")
    print(f"Working memory: {stm.get_working('current_file')}")

    # 长期记忆
    ltm = LongTermMemory(config)
    ltm.remember_fact("user_name", "Alice")
    ltm.remember_fact("preferred_language", "Python")

    print(f"KV: {ltm.kv_store.all()}")

    # 清理
    ltm.forget_fact("user_name")


# ─────────────────────────────────────────────────────────────
# Demo 5: 配置加载
# ─────────────────────────────────────────────────────────────

def demo_config():
    """多种配置方式"""
    print("\n" + "=" * 60)
    print("Demo 5: Configuration")
    print("=" * 60)

    # 方式 1: 代码配置
    config = AgentConfig()
    config.llm.model = "gpt-4o-mini"
    config.max_iterations = 10
    print(f"Model: {config.llm.model}, Max iterations: {config.max_iterations}")

    # 方式 2: 字典
    config2 = AgentConfig.from_dict({
        "llm": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
        "max_iterations": 15,
    })
    print(f"Model: {config2.llm.model}")

    # 方式 3: 环境变量
    print(f"LLM_API_KEY from env: "
          f"{'set' if os.environ.get('LLM_API_KEY') else 'not set'}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentLite Demo")
    parser.add_argument("--demo", type=int, default=0,
                        help="Which demo to run (1-5). 0 = all (no-LLM demos only if no API key)")
    args = parser.parse_args()

    has_api_key = bool(os.environ.get("LLM_API_KEY"))

    if not has_api_key:
        print("Note: LLM_API_KEY not set. "
              "Running demos that don't require LLM calls.\n")

    demos = {
        1: demo_basic,
        2: demo_custom_tool,
        3: demo_rag,
        4: demo_memory,
        5: demo_config,
    }

    if args.demo > 0:
        if args.demo in (1, 2, 3) and not has_api_key:
            print(f"Demo {args.demo} requires LLM_API_KEY. Skipping.")
        else:
            demos[args.demo]()
    else:
        # 运行不需要 LLM 的 demo
        demo_memory()
        demo_config()
        if has_api_key:
            demo_rag()
            demo_basic()
            demo_custom_tool()
