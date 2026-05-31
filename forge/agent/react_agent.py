import json
import re
from typing import Optional

from .base_agent import BaseAgent, BaseTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) agent implementation."""

    SYSTEM_PROMPT = """你是一个智能助手，能够使用工具来回答问题。

可用工具：
{tools}

请使用以下格式进行推理：
Thought: 我需要思考如何回答这个问题...
Action: 工具名称
Action Input: 工具输入
Observation: 工具返回结果
... (可以重复多次)
Thought: 我现在知道答案了
Final Answer: 最终答案

开始！"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", max_turns: int = 5):
        super().__init__(model_path, device, max_turns)
        self.history: list[dict] = []

    def _build_tool_description(self) -> str:
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)

    def _parse_action(self, text: str) -> Optional[tuple[str, str]]:
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return action, action_input
        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, query: str) -> str:
        logger.info(f"ReAct Agent query: {query}")

        tool_desc = self._build_tool_description()
        system_prompt = self.SYSTEM_PROMPT.format(tools=tool_desc)

        conversation = f"{system_prompt}\n\nUser: {query}\n"

        for turn in range(self.max_turns):
            response = self._generate(conversation, max_new_tokens=512)
            conversation += response

            final_answer = self._parse_final_answer(response)
            if final_answer:
                logger.info(f"ReAct Agent answer: {final_answer}")
                self.history.append({"query": query, "answer": final_answer})
                return final_answer

            action_info = self._parse_action(response)
            if action_info:
                action_name, action_input = action_info
                if action_name in self.tools:
                    observation = self.tools[action_name].run(action_input)
                    conversation += f"\nObservation: {observation}\n"
                    logger.info(f"Tool {action_name}: {observation[:100]}...")
                else:
                    conversation += f"\nObservation: 错误 - 未知工具 '{action_name}'\n"
            else:
                conversation += "\nThought: 让我重新思考...\n"

        return "抱歉，我无法在限定步骤内找到答案。"

    def chat(self, message: str) -> str:
        """Simple chat without tools."""
        if not self.tools:
            return self._generate(f"User: {message}\nAssistant:", max_new_tokens=256)
        return self.run(message)
