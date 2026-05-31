import json
from typing import Optional

from .base_agent import BaseAgent
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Planner:
    """Task planner that decomposes complex queries into sub-tasks."""

    PLAN_PROMPT = """请将以下任务分解为可执行的子步骤。以JSON数组格式返回。

任务: {task}

输出格式:
[
    {{"step": 1, "action": "步骤描述", "tool": "使用的工具或null"}},
    {{"step": 2, "action": "步骤描述", "tool": "使用的工具或null"}}
]

子步骤:"""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def plan(self, task: str) -> list[dict]:
        prompt = self.PLAN_PROMPT.format(task=task)
        response = self.agent._generate(prompt, max_new_tokens=512)

        try:
            json_match = response.strip()
            if json_match.startswith("["):
                steps = json.loads(json_match)
            else:
                import re
                match = re.search(r"\[.*\]", json_match, re.DOTALL)
                if match:
                    steps = json.loads(match.group())
                else:
                    steps = [{"step": 1, "action": task, "tool": None}]

            logger.info(f"Plan created with {len(steps)} steps")
            return steps
        except Exception as e:
            logger.warning(f"Failed to parse plan: {e}")
            return [{"step": 1, "action": task, "tool": None}]

    def execute_plan(self, task: str) -> str:
        steps = self.plan(task)
        results = []

        for step in steps:
            action = step["action"]
            tool_name = step.get("tool")

            if tool_name and tool_name in self.agent.tools:
                result = self.agent.tools[tool_name].run(action)
            else:
                result = self.agent._generate(f"请执行以下任务: {action}\n回答:", max_new_tokens=256)

            results.append({"step": step["step"], "action": action, "result": result})
            logger.info(f"Step {step['step']}: {action} -> {result[:100]}...")

        summary = "\n".join([f"步骤{r['step']}: {r['action']}\n结果: {r['result']}" for r in results])
        return summary
