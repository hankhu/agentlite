"""
AgentLite — 任务规划器

将复杂任务分解为有序步骤序列。
支持: 简单任务直通 / 复杂任务 LLM 分解。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .types import Message, Plan, Step, StepStatus, ToolDef

logger = logging.getLogger("agentlite.planner")


# ── Plan Templates ──────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a task planning expert. Your job is to decompose a complex task into a sequence of simple, actionable steps.

Rules:
1. Each step must be a single, clear action.
2. Always specify which tool to use for each step when applicable.
3. Mark dependencies.
4. Output ONLY valid JSON, no explanation.

Example output:
{{"goal": "task description", "steps": [{{"id": 1, "description": "...", "tool": "read_file", "depends_on": []}}]}}

Available tools:
{tools}

For simple tasks that can be done in one step, output a single-step plan.
For tasks that require reasoning without tools, set tool to null.
"""


# ── Planner ─────────────────────────────────────────────────

class Planner:
    """任务规划器：将自然语言任务分解为结构化计划"""

    def __init__(self, llm_interface=None,
                 tools: Optional[List[ToolDef]] = None):
        self.llm = llm_interface
        self.tools = tools or []
        self._tool_names = [t.name for t in self.tools]
        self._tool_descs = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )

    def plan(self, task: str,
             context: Optional[str] = None) -> Plan:
        """分析任务并生成计划"""
        # 简单启发式：任务短且不含「然后」「接着」「首先」等词 → 单步
        if self._is_simple(task):
            return Plan(
                goal=task,
                steps=[Step(id=1, description=task, tool=None)],
            )

        # LLM 规划
        if self.llm is not None:
            return self._llm_plan(task, context)

        # Fallback: 单步计划
        return Plan(
            goal=task,
            steps=[Step(id=1, description=task, tool=None)],
        )

    def replan(self, plan: Plan, current_state: str,
               error: Optional[str] = None) -> Plan:
        """根据当前状态重新规划"""
        if self.llm is None:
            return plan

        completed = [s for s in plan.steps
                     if s.status == StepStatus.COMPLETED]
        remaining = [s for s in plan.steps
                     if s.status == StepStatus.PENDING]

        prompt = f"""The original plan was:
Goal: {plan.goal}

Completed steps:
{self._format_steps(completed)}

Remaining steps:
{self._format_steps(remaining)}

Current situation: {current_state}
{f'Error encountered: {error}' if error else ''}

Please provide a revised plan (JSON format) for the remaining work."""

        messages = [
            Message.system(PLANNER_SYSTEM_PROMPT.format(
                tools=self._tool_descs)),
            Message.user(prompt),
        ]

        response = self.llm.chat(messages)
        new_plan = self._parse_plan(response.content or "", plan.goal)

        # 继承已完成步骤
        merged_steps = list(completed)
        offset = max((s.id for s in completed), default=0)
        for s in new_plan.steps:
            s.id += offset
            s.depends_on = [d + offset for d in s.depends_on]
            merged_steps.append(s)

        return Plan(goal=plan.goal, steps=merged_steps)

    # ── Internal ──────────────────────────────────────────

    def _is_simple(self, task: str) -> bool:
        """启发式判断任务是否简单"""
        complexity_markers = [
            "然后", "接着", "首先", "其次", "最后",
            "then", "next", "first", "second", "finally",
            "步骤", "step", "计划", "plan",
            "先", "再", "之后", "之前",
            "并且", "同时", "also", "and then",
        ]
        # 短任务且无复杂度标记 → 简单
        if len(task) < 80:
            return not any(m in task.lower() for m in complexity_markers)
        # 长任务但有明确步骤枚举 → 不简单
        return False

    def _llm_plan(self, task: str,
                  context: Optional[str] = None) -> Plan:
        """使用 LLM 生成计划"""
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        messages = [
            Message.system(PLANNER_SYSTEM_PROMPT.format(
                tools=self._tool_descs)),
            Message.user(prompt),
        ]

        try:
            response = self.llm.chat(messages)
            return self._parse_plan(response.content or "", task)
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using simple plan")
            return Plan(
                goal=task,
                steps=[Step(id=1, description=task, tool=None)],
            )

    def _parse_plan(self, raw: str, goal: str) -> Plan:
        """解析 LLM 输出的 JSON 计划"""
        try:
            # 提取 JSON 块
            raw = raw.strip()
            if "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            if raw.startswith("{"):
                data = json.loads(raw)
            else:
                # 尝试找到 JSON 对象
                brace_start = raw.index("{")
                brace_end = raw.rindex("}") + 1
                data = json.loads(raw[brace_start:brace_end])

            steps = []
            for s in data.get("steps", []):
                steps.append(Step(
                    id=s.get("id", len(steps) + 1),
                    description=s.get("description", ""),
                    tool=s.get("tool"),
                    depends_on=s.get("depends_on", []),
                ))

            return Plan(
                goal=data.get("goal", goal),
                steps=steps,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse plan JSON: {e}")
            # Fallback
            return Plan(
                goal=goal,
                steps=[Step(id=1, description=goal, tool=None)],
            )

    @staticmethod
    def _format_steps(steps: List[Step]) -> str:
        lines = []
        for s in steps:
            dep = f" (depends on: {s.depends_on})" if s.depends_on else ""
            lines.append(f"  {s.id}. [{s.status}] {s.description}{dep}")
        return "\n".join(lines) if lines else "(none)"
