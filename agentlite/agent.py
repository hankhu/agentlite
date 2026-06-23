"""
AgentLite — 主 Agent 编排器

实现完整的 Perceive → Plan → Decide → Execute → Observe → Reflect 循环。
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig, LLMConfig
from .llm import LLMInterface
from .memory import LongTermMemory, ShortTermMemory
from .planner import Planner
from .rag import RAGEngine
from .subagent import SubAgentManager
from .tools import ToolRegistry, ToolDef, build_builtin_tools, ToolCall
from .types import (
    AgentState, LLMResponse, Message, Plan, Usage,
    StepStatus, ToolResult,
)

logger = logging.getLogger("agentlite.agent")


class Agent:
    """极简 AI Agent

    用法:
        agent = Agent(
            llm_config={"model": "gpt-4o", "api_key": "sk-..."}
        )
        result = agent.run("列出当前目录下所有 Python 文件")
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolDef]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            config: 完整 AgentConfig 对象
            llm_config: LLM 配置字典 (与 config 二选一)
            tools: 额外的自定义工具
            system_prompt: 自定义系统提示词
            **kwargs: 其他 AgentConfig 字段
        """
        # ── 配置 ──────────────────────────────────────────
        if config is None:
            config = AgentConfig()

        # 支持快捷字典配置
        if llm_config:
            llm_cfg = config.llm
            for k, v in llm_config.items():
                setattr(llm_cfg, k, v)

        if system_prompt:
            config.system_prompt = system_prompt
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        self.config = config
        self.state = AgentState.IDLE

        # ── LLM ───────────────────────────────────────────
        self.llm = LLMInterface(config.llm)

        # ── Memory ────────────────────────────────────────
        self.short_term = ShortTermMemory(
            config.memory,
            token_counter=lambda msgs: self.llm.count_message_tokens(msgs),
        )
        self.long_term = LongTermMemory(
            config.memory,
            llm_interface=self.llm if config.memory.auto_summarize else None,
        )

        # ── Tools ─────────────────────────────────────────
        self.tool_registry = ToolRegistry(config.tools)
        # 注册内置工具
        for tool in build_builtin_tools(config.tools):
            self.tool_registry.register(tool)
        # 注册用户自定义工具
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

        # ── Sub-system ────────────────────────────────────
        self.planner = Planner(self.llm, self.tool_registry.list_all())
        self.subagent_manager = SubAgentManager(
            agent_factory=self._create_subagent,
            config=config.subagent,
        )
        self.rag = RAGEngine(config.rag, llm_interface=self.llm)

        # ── 系统提示词 ─────────────────────────────────────
        self._base_system_prompt = config.system_prompt

        # ── Hook: 执行前/后回调 ───────────────────────────
        self._before_tool_hooks: List[Callable] = []
        self._after_tool_hooks: List[Callable] = []
        self._confirm_dangerous: bool = config.tools.dangerous_require_confirm

        # ── Token 消耗跟踪 ────────────────────────────────
        self.last_usage: Usage = Usage()    # 上一轮的 token 消耗
        self.total_usage: Usage = Usage()   # 累计 token 消耗

        logger.info(f"Agent initialized with model={config.llm.model}")

    # ── Main API ────────────────────────────────────────────

    def run(self, task: str, context: Optional[str] = None) -> str:
        """执行任务的主入口。

        Args:
            task: 用户任务描述
            context: 额外的上下文信息
        Returns:
            Agent 的最终回复
        """
        self.state = AgentState.RUNNING
        self._reset_short_term()
        self.last_usage = Usage()  # 重置本轮用量统计

        try:
            # ── Phase 1: Perceive ─────────────────────────
            enriched_task = self._gather_context(task, context)

            # ── Phase 2: Plan ─────────────────────────────
            plan = self.planner.plan(enriched_task, context)
            if self.config.verbose:
                self._log_plan(plan)

            # ── Phase 3-6: Main Loop ──────────────────────
            final_answer = self._main_loop(plan)

            # ── Phase 7: Summarize ────────────────────────
            if self.config.verbose:
                logger.info(f"Task completed. Answer length: {len(final_answer)}")

            # ── Phase 8: Memorize ─────────────────────────
            self._commit_to_long_term(task, final_answer)

            self.state = AgentState.DONE
            return final_answer

        except Exception as e:
            self.state = AgentState.ERROR
            logger.exception("Agent run failed")
            return f"Error: {e}"

    def chat(self, text: str) -> str:
        """多轮对话：保留短期记忆，追加用户消息。

        与 run() 的区别:
          - 不调用 _reset_short_term()，保留历史上下文
          - 不重新构建 system prompt（除非首次）
          - 不触发长期记忆提交（每轮太频繁）
        """
        self.state = AgentState.RUNNING
        self.last_usage = Usage()

        try:
            # 首次调用时初始化 system prompt
            existing = self.short_term.get_messages()
            if not existing or not any(m.role == "system" for m in existing):
                self.short_term.add_system(self._build_system_prompt())

            # RAG / 长期记忆检索（只在有足够上下文时）
            rag_ctx = self.rag.search_as_context(text)
            mem_ctx = self.long_term.recall_as_context(text)
            enriched = text
            if rag_ctx:
                enriched = f"{text}\n\n{rag_ctx}"
            if mem_ctx:
                enriched = f"{enriched}\n\n{mem_ctx}"

            self.short_term.add_user(enriched)

            # 构造简化计划（不走 LLM 规划，直接调用主循环）
            plan = self.planner.plan(text)
            answer = self._main_loop(plan)

            self.state = AgentState.DONE
            return answer

        except Exception as e:
            self.state = AgentState.ERROR
            logger.exception("Agent.chat failed")
            return f"Error: {e}"

    def run_stream(self, task: str,
                   context: Optional[str] = None):
        """流式执行任务 (generator)"""
        # 简化版: 先用 run 获取结果，再 yield
        # 完整版需要改 LLM 调用为 streaming
        result = self.run(task, context)
        yield result

    # ── Tool Execution Hooks ─────────────────────────────────

    def on_before_tool(self, hook: Callable) -> None:
        """注册工具执行前回调: hook(tool_call, agent) -> Optional[bool]
        返回 False 可阻止执行。"""
        self._before_tool_hooks.append(hook)

    def on_after_tool(self, hook: Callable) -> None:
        """注册工具执行后回调: hook(tool_result, agent) -> None"""
        self._after_tool_hooks.append(hook)

    # ── RAG Shortcuts ────────────────────────────────────────

    def ingest(self, path: str, glob: str = "**/*") -> int:
        """摄入文档到 RAG"""
        if os.path.isdir(path):
            return self.rag.ingest_directory(path, glob)
        else:
            return self.rag.ingest_file(path)

    def ask_rag(self, query: str, k: int = 5) -> str:
        """直接查询 RAG (不经过 Agent 循环)"""
        return self.rag.search_as_context(query, k)

    # ── Internal: Main Loop ──────────────────────────────────

    def _main_loop(self, plan: Plan) -> str:
        """Execute-Decide-Observe-Reflect 循环"""
        iteration = 0
        current_step = plan.current_step()

        while iteration < self.config.max_iterations:
            iteration += 1

            # 构建当前步骤提示
            step_hint = ""
            if current_step:
                step_hint = (
                    f"\n[Current plan step: {current_step.id}. "
                    f"{current_step.description}]"
                )

            # ── Decide ────────────────────────────────────
            messages = self.short_term.get_messages()
            # 添加上下文提示（但不存入短期记忆）
            if step_hint:
                # 修改最后一条 user message 以包含步骤提示
                pass  # 步骤信息已通过系统提示传达

            response = self.llm.chat(
                messages=messages,
                tools=self.tool_registry.list_all(),
            )

            # 累计 token 用量
            if response.usage:
                self.last_usage.prompt_tokens += response.usage.prompt_tokens
                self.last_usage.completion_tokens += response.usage.completion_tokens
                self.last_usage.total_tokens += response.usage.total_tokens
                self.total_usage.prompt_tokens += response.usage.prompt_tokens
                self.total_usage.completion_tokens += response.usage.completion_tokens
                self.total_usage.total_tokens += response.usage.total_tokens

            if response.is_tool_call():
                # ── Execute ────────────────────────────────
                for tc in response.tool_calls:
                    result = self._execute_tool(tc)
                    # 将 tool_call 和 result 添加到短期记忆
                    self.short_term.add(
                        Message.assistant(tool_calls=[tc],
                                          reasoning_content=response.reasoning_content)
                    )
                    self.short_term.add_tool_result(
                        tool_call_id=tc.id,
                        result=result.summary(),
                        name=tc.name,
                    )
                    if self.config.verbose:
                        logger.info(
                            f"Tool: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:100]}) "
                            f"→ {'✓' if result.success else '✗'}"
                        )

                # 更新计划状态
                if current_step:
                    current_step.status = StepStatus.COMPLETED
                    current_step = plan.current_step()

            elif response.is_final():
                # ── Final Answer ───────────────────────────
                self.short_term.add(Message.assistant(
                    content=response.content or "",
                    reasoning_content=response.reasoning_content
                ))
                return response.content or ""

            else:
                # 空响应或异常
                logger.warning(f"Unexpected response: {response}")
                if response.content:
                    self.short_term.add_assistant(response.content)
                    return response.content
                break

        # 达到最大迭代次数，请求 LLM 总结
        logger.warning(f"Reached max iterations ({self.config.max_iterations})")
        return self._force_summarize()

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行单个工具调用（含 hooks 和确认）"""
        tool = self.tool_registry.get(tool_call.name)

        # Before hooks
        for hook in self._before_tool_hooks:
            try:
                result = hook(tool_call, self)
                if result is False:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        success=False,
                        output=None,
                        error="Blocked by before-tool hook",
                    )
            except Exception as e:
                logger.warning(f"Before-tool hook error: {e}")

        # 危险工具确认 (可通过 hook 实现，这里仅记录)
        if tool and tool.dangerous and self._confirm_dangerous:
            logger.info(
                f"Executing dangerous tool: {tool_call.name}"
            )

        # 执行
        result = self.tool_registry.execute(tool_call)

        # After hooks
        for hook in self._after_tool_hooks:
            try:
                hook(result, self)
            except Exception as e:
                logger.warning(f"After-tool hook error: {e}")

        return result

    # ── Internal: Context Gathering ───────────────────────────

    def _gather_context(self, task: str,
                        context: Optional[str] = None) -> str:
        """收集所有相关上下文"""
        parts = [task]

        # RAG 检索
        rag_context = self.rag.search_as_context(task)
        if rag_context:
            parts.append(f"\n\n{rag_context}")
            if self.config.verbose:
                logger.info("RAG context appended")

        # 长期记忆检索
        mem_context = self.long_term.recall_as_context(task)
        if mem_context:
            parts.append(f"\n\n{mem_context}")
            if self.config.verbose:
                logger.info("Long-term memory context appended")

        if context:
            parts.append(f"\n\n[Additional context]:\n{context}")

        enriched = "\n".join(parts)

        # 构建初始消息
        system_msg = self._build_system_prompt()
        self.short_term.add_system(system_msg)
        self.short_term.add_user(enriched)

        return enriched

    def _build_system_prompt(self) -> str:
        """构建系统提示词（含工具列表）"""
        tool_descs = []
        for t in self.tool_registry.list_all():
            tool_descs.append(f"- {t.name}: {t.description}")
        tools_section = "\n".join(tool_descs)

        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d %A")
        return f"""{self._base_system_prompt}

Current date: {today}

Available tools:
{tools_section}

Instructions:
- Analyze the task carefully before acting.
- Use tools when needed; provide a direct answer when you have enough information.
- When executing shell commands, be precise and safe.
- Summarize your findings clearly at the end.
- If you cannot complete the task, explain what's missing."""

    # ── Internal: Memory Management ───────────────────────────

    def _reset_short_term(self) -> None:
        """重置短期记忆（保留长期记忆）"""
        self.short_term.clear()

    def _commit_to_long_term(self, task: str, result: str) -> None:
        """将会话总结存入长期记忆"""
        try:
            # 添加情节
            summary = result[:500] if len(result) > 500 else result
            self.long_term.add_episode(task, summary)
        except Exception as e:
            logger.warning(f"Failed to commit to long-term memory: {e}")

    def _force_summarize(self) -> str:
        """达到最大迭代次数时强制总结"""
        messages = self.short_term.get_messages()
        messages.append(Message.user(
            "You have reached the maximum number of steps. "
            "Please provide a final summary of what you have accomplished "
            "and what remains to be done."
        ))
        try:
            resp = self.llm.chat(messages)
            return resp.content or "Unable to summarize."
        except Exception:
            return "Maximum iterations reached. Unable to generate summary."

    # ── Internal: SubAgent Factory ────────────────────────────

    def _create_subagent(self, task: str,
                         tools: Optional[List[str]] = None,
                         memory_context: Optional[Dict] = None) -> "Agent":
        """创建子 Agent 实例（由 SubAgentManager 调用）"""
        # 克隆配置但降低某些参数
        sub_config = AgentConfig()
        sub_config.llm = self.config.llm  # 共享 LLM 配置
        sub_config.memory = self.config.memory
        sub_config.tools = self.config.tools
        sub_config.rag = self.config.rag
        sub_config.system_prompt = (
            "You are a sub-agent. Complete the assigned subtask efficiently. "
            "Use tools as needed and return a concise result."
        )
        sub_config.max_iterations = min(10, self.config.max_iterations // 2)
        sub_config.verbose = self.config.verbose

        # 创建子 Agent
        sub = Agent(config=sub_config, llm_config=None)

        # 限制工具
        if tools:
            allowed = set(tools)
            for name in list(sub.tool_registry.list_names()):
                if name not in allowed:
                    sub.tool_registry.unregister(name)

        # 注入上下文到短期记忆
        if memory_context:
            for k, v in memory_context.items():
                sub.short_term.set_working(k, v)

        return sub

    # ── Logging ───────────────────────────────────────────────

    def _log_plan(self, plan: Plan) -> None:
        """打印计划"""
        logger.info(f"Plan for: {plan.goal}")
        for s in plan.steps:
            dep = f" (depends: {s.depends_on})" if s.depends_on else ""
            tool = f" [{s.tool}]" if s.tool else ""
            logger.info(f"  Step {s.id}: {s.description}{tool}{dep}")


# ── 辅助 ─────────────────────────────────────────────────────

import os  # noqa: E402 (used in _create_subagent)
