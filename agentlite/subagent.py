"""
AgentLite — 子 Agent 管理

支持:
- 同步执行: spawn_and_wait()
- 异步执行: spawn() + collect()
- 工具限制: 子 Agent 只能使用指定的工具子集
- 深度限制: 防止无限递归
"""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional

from .types import SubAgentResult

logger = logging.getLogger("agentlite.subagent")


class SubAgentManager:
    """子 Agent 管理器

    管理子 Agent 的生命周期:
    - 创建: 克隆父 Agent 的部分工具和配置
    - 执行: 在独立线程中运行
    - 收集: 获取执行结果
    """

    def __init__(self, agent_factory, config=None):
        """
        Args:
            agent_factory: callable(task, tools, memory_config) -> Agent
                           工厂函数，用于创建子 Agent 实例。
                           避免循环导入，由 Agent.__init__ 注入。
            config: SubAgentConfig
        """
        self._factory = agent_factory
        self.config = config
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent if config else 5
        )
        self._futures: Dict[str, Future] = {}
        self._active: Dict[str, Any] = {}
        self._depth = 0

    @property
    def max_depth(self) -> int:
        return self.config.max_depth if self.config else 3

    def spawn(self, task: str,
              tools: Optional[List[str]] = None,
              context: Optional[Dict[str, Any]] = None,
              depth: Optional[int] = None) -> str:
        """异步启动子 Agent。返回 agent_id。"""
        if depth is None:
            depth = self._depth + 1

        if depth > self.max_depth:
            raise RuntimeError(
                f"Max sub-agent depth ({self.max_depth}) exceeded. "
                f"Current depth: {depth}"
            )

        agent_id = f"subagent-{uuid.uuid4().hex[:8]}"

        future = self._executor.submit(
            self._run_subagent, agent_id, task, tools, context, depth
        )
        self._futures[agent_id] = future
        self._active[agent_id] = {
            "task": task,
            "status": "running",
            "tools": tools,
            "depth": depth,
        }

        logger.info(f"Spawned sub-agent {agent_id}: {task[:80]}...")
        return agent_id

    def spawn_and_wait(self, task: str,
                       tools: Optional[List[str]] = None,
                       context: Optional[Dict[str, Any]] = None,
                       timeout: Optional[int] = None,
                       depth: Optional[int] = None) -> SubAgentResult:
        """同步启动子 Agent，等待完成并返回结果。"""
        if timeout is None and self.config:
            timeout = self.config.default_timeout

        agent_id = self.spawn(task, tools, context, depth)
        return self.wait(agent_id, timeout=timeout)

    def wait(self, agent_id: str,
             timeout: Optional[int] = None) -> SubAgentResult:
        """等待指定子 Agent 完成。"""
        future = self._futures.get(agent_id)
        if future is None:
            return SubAgentResult(
                id=agent_id,
                task="",
                status="error",
                summary="",
                error=f"Unknown agent: {agent_id}",
            )
        try:
            result = future.result(timeout=timeout)
            self._active[agent_id]["status"] = result.status
            return result
        except Exception as e:
            self._active[agent_id]["status"] = "error"
            return SubAgentResult(
                id=agent_id,
                task=self._active.get(agent_id, {}).get("task", ""),
                status="error",
                summary="",
                error=str(e),
            )

    def cancel(self, agent_id: str) -> bool:
        """取消子 Agent"""
        future = self._futures.get(agent_id)
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                self._active[agent_id]["status"] = "cancelled"
                logger.info(f"Cancelled sub-agent {agent_id}")
            return cancelled
        return False

    def list_active(self) -> List[str]:
        """列出活跃的子 Agent ID"""
        return [aid for aid, info in self._active.items()
                if info["status"] == "running"]

    def collect_all(self, timeout: Optional[int] = None) -> List[SubAgentResult]:
        """等待所有活跃子 Agent 完成并返回结果"""
        results = []
        for agent_id in list(self._active.keys()):
            if self._active[agent_id]["status"] == "running":
                results.append(self.wait(agent_id, timeout))
        return results

    def shutdown(self) -> None:
        """关闭管理器，取消所有运行中的子 Agent"""
        for agent_id in self.list_active():
            self.cancel(agent_id)
        self._executor.shutdown(wait=False)

    # ── Internal ──────────────────────────────────────────

    def _run_subagent(self, agent_id: str, task: str,
                      tools: Optional[List[str]],
                      context: Optional[Dict],
                      depth: int) -> SubAgentResult:
        """在独立线程中运行子 Agent"""
        try:
            # 创建子 Agent 实例
            agent = self._factory(
                task=task,
                tools=tools,
                memory_context=context,
            )

            # 设置深度
            if hasattr(agent, 'subagent_manager'):
                agent.subagent_manager._depth = depth

            # 执行
            result_text = agent.run(task)

            self._active[agent_id]["status"] = "done"

            return SubAgentResult(
                id=agent_id,
                task=task,
                status="done",
                summary=result_text,
                data={"raw_output": result_text},
            )
        except Exception as e:
            logger.exception(f"Sub-agent {agent_id} failed")
            self._active[agent_id]["status"] = "failed"
            return SubAgentResult(
                id=agent_id,
                task=task,
                status="failed",
                summary="",
                error=str(e),
            )
