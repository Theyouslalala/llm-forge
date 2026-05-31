"""Tests for agent components: tools, memory, planner."""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.agent.tools.calculator import CalculatorTool
from forge.agent.tools.web_search import WebSearchTool
from forge.agent.tools.code_executor import CodeExecutorTool
from forge.agent.tools.file_reader import FileReaderTool
from forge.agent.memory import AgentMemory
from forge.agent.planner import Planner
from forge.agent.base_agent import BaseAgent, BaseTool


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def test_basic_arithmetic(self):
        calc = CalculatorTool()
        assert "4" in calc.run("2 + 2")
        assert "6" in calc.run("2 * 3")
        assert "1" in calc.run("3 - 2")
        assert "2" in calc.run("4 / 2")

    def test_math_functions(self):
        calc = CalculatorTool()
        result = calc.run("sqrt(16)")
        assert "4" in result

    def test_trig_functions(self):
        calc = CalculatorTool()
        result = calc.run("sin(0)")
        assert "0" in result

    def test_pi_constant(self):
        calc = CalculatorTool()
        result = calc.run("pi")
        assert "3.14" in result

    def test_complex_expression(self):
        calc = CalculatorTool()
        result = calc.run("(2 + 3) * 4")
        assert "20" in result

    def test_invalid_expression(self):
        calc = CalculatorTool()
        result = calc.run("invalid!!!")
        assert "错误" in result or "error" in result.lower()

    def test_division_by_zero(self):
        calc = CalculatorTool()
        result = calc.run("1 / 0")
        assert "错误" in result or "error" in result.lower()

    def test_name_and_description(self):
        calc = CalculatorTool()
        assert calc.name == "calculator"
        assert len(calc.description) > 0

    def test_to_schema(self):
        calc = CalculatorTool()
        schema = calc.to_schema()
        assert schema["name"] == "calculator"
        assert "description" in schema

    def test_security_no_imports(self):
        calc = CalculatorTool()
        result = calc.run("__import__('os').system('echo hacked')")
        # Should fail because builtins are restricted
        assert "错误" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    def test_returns_results(self):
        tool = WebSearchTool()
        result = tool.run("Python programming")
        assert "Python programming" in result

    def test_contains_search_results(self):
        tool = WebSearchTool()
        result = tool.run("test query")
        assert "搜索结果" in result or "搜索查询" in result

    def test_name(self):
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_strips_whitespace(self):
        tool = WebSearchTool()
        result1 = tool.run("  hello  ")
        result2 = tool.run("hello")
        # Both should work without errors
        assert len(result1) > 0
        assert len(result2) > 0


# ---------------------------------------------------------------------------
# CodeExecutorTool
# ---------------------------------------------------------------------------

class TestCodeExecutorTool:
    def test_print_output(self):
        tool = CodeExecutorTool()
        result = tool.run("print(42)")
        assert "42" in result

    def test_variable_assignment(self):
        tool = CodeExecutorTool()
        result = tool.run("x = 10\nprint(x)")
        assert "10" in result

    def test_strip_code_fences(self):
        tool = CodeExecutorTool()
        result = tool.run("```python\nprint('hello')\n```")
        assert "hello" in result

    def test_syntax_error(self):
        tool = CodeExecutorTool()
        result = tool.run("def (invalid")
        assert "执行错误" in result or "Error" in result

    def test_runtime_error(self):
        tool = CodeExecutorTool()
        result = tool.run("raise ValueError('test error')")
        assert "执行错误" in result or "ValueError" in result

    def test_no_output(self):
        tool = CodeExecutorTool()
        result = tool.run("x = 1 + 1")
        assert "无输出" in result or "成功" in result

    def test_name(self):
        tool = CodeExecutorTool()
        assert tool.name == "code_executor"


# ---------------------------------------------------------------------------
# FileReaderTool
# ---------------------------------------------------------------------------

class TestFileReaderTool:
    def test_read_existing_file(self, tmp_path):
        tool = FileReaderTool()
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = tool.run(str(f))
        assert "Hello, world!" in result

    def test_file_not_found(self):
        tool = FileReaderTool()
        result = tool.run("/nonexistent/path/to/file.txt")
        assert "不存在" in result or "not found" in result.lower()

    def test_binary_file(self, tmp_path):
        tool = FileReaderTool()
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe")
        result = tool.run(str(f))
        # Should handle gracefully
        assert "二进制" in result or "无法" in result or "错误" in result

    def test_name(self):
        tool = FileReaderTool()
        assert tool.name == "file_reader"

    def test_strips_quotes(self, tmp_path):
        tool = FileReaderTool()
        f = tmp_path / "quoted.txt"
        f.write_text("content", encoding="utf-8")
        result = tool.run(f'"{f}"')
        assert "content" in result


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class TestAgentMemory:
    def test_add_short_term(self):
        mem = AgentMemory(max_short_term=5)
        mem.add_short_term("user", "hello")
        mem.add_short_term("assistant", "hi there")
        assert len(mem.short_term) == 2
        assert mem.short_term[0]["role"] == "user"
        assert mem.short_term[1]["content"] == "hi there"

    def test_short_term_max_limit(self):
        mem = AgentMemory(max_short_term=3)
        for i in range(10):
            mem.add_short_term("user", f"message {i}")
        assert len(mem.short_term) == 3
        # Should keep the last 3
        assert mem.short_term[0]["content"] == "message 7"

    def test_add_long_term(self):
        mem = AgentMemory()
        mem.add_long_term("Important fact", metadata={"type": "fact"})
        assert len(mem.long_term) == 1
        assert len(mem.long_term_embeddings) == 1
        assert mem.long_term[0]["metadata"]["type"] == "fact"

    def test_search_long_term(self):
        mem = AgentMemory(embedding_dim=384)
        mem.add_long_term("Python is a programming language")
        mem.add_long_term("Cats are fluffy animals")
        mem.add_long_term("JavaScript runs in browsers")

        results = mem.search_long_term("programming", top_k=2)
        assert len(results) <= 2
        assert all("score" in r for r in results)

    def test_search_empty_long_term(self):
        mem = AgentMemory()
        results = mem.search_long_term("query")
        assert results == []

    def test_get_context_short_term(self):
        mem = AgentMemory()
        mem.add_short_term("user", "What is AI?")
        mem.add_short_term("assistant", "AI is artificial intelligence.")

        ctx = mem.get_context(include_short_term=True)
        assert "对话历史" in ctx
        assert "What is AI?" in ctx

    def test_get_context_with_query(self):
        mem = AgentMemory()
        mem.add_long_term("Python is great for ML")
        mem.add_long_term("The sky is blue")

        ctx = mem.get_context(include_short_term=False, query="Python ML")
        assert "相关记忆" in ctx

    def test_get_context_empty(self):
        mem = AgentMemory()
        ctx = mem.get_context()
        assert ctx == ""

    def test_clear_short_term(self):
        mem = AgentMemory()
        mem.add_short_term("user", "test")
        mem.clear_short_term()
        assert len(mem.short_term) == 0

    def test_clear_all(self):
        mem = AgentMemory()
        mem.add_short_term("user", "test")
        mem.add_long_term("fact")
        mem.clear_all()
        assert len(mem.short_term) == 0
        assert len(mem.long_term) == 0
        assert len(mem.long_term_embeddings) == 0

    def test_save_and_load(self, tmp_path):
        mem = AgentMemory()
        mem.add_short_term("user", "hello")
        mem.add_long_term("important fact")

        path = str(tmp_path / "memory.json")
        mem.save(path)

        mem2 = AgentMemory()
        mem2.load(path)
        assert len(mem2.short_term) == 1
        assert len(mem2.long_term) == 1
        assert mem2.short_term[0]["content"] == "hello"
        assert mem2.long_term[0]["content"] == "important fact"


# ---------------------------------------------------------------------------
# Planner (using a stub agent)
# ---------------------------------------------------------------------------

class StubAgent(BaseAgent):
    """Agent stub that returns canned responses for testing."""

    def __init__(self):
        # Do not call super().__init__() to avoid loading model
        self.device = "cpu"
        self.max_turns = 5
        self.tools = {}
        self.model = None
        self.tokenizer = None
        self._responses = []

    def set_responses(self, responses: list[str]):
        self._responses = list(responses)

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "[]"

    def run(self, query: str) -> str:
        return "stub"


class TestPlanner:
    def test_plan_valid_json(self):
        agent = StubAgent()
        agent.set_responses([
            json.dumps([
                {"step": 1, "action": "Search for info", "tool": "web_search"},
                {"step": 2, "action": "Summarize results", "tool": None},
            ])
        ])
        planner = Planner(agent)
        steps = planner.plan("Find and summarize info about AI")
        assert len(steps) == 2
        assert steps[0]["step"] == 1
        assert steps[0]["tool"] == "web_search"

    def test_plan_json_embedded_in_text(self):
        agent = StubAgent()
        agent.set_responses([
            'Here is the plan:\n[{"step": 1, "action": "Do something", "tool": null}]\nDone.'
        ])
        planner = Planner(agent)
        steps = planner.plan("Do something")
        assert len(steps) == 1

    def test_plan_invalid_json_fallback(self):
        agent = StubAgent()
        agent.set_responses(["This is not JSON at all"])
        planner = Planner(agent)
        steps = planner.plan("Some task")
        assert len(steps) == 1
        assert steps[0]["step"] == 1

    def test_execute_plan(self):
        agent = StubAgent()
        agent.set_responses([
            json.dumps([
                {"step": 1, "action": "calculate 2+2", "tool": "calculator"},
            ])
        ])
        calc = CalculatorTool()
        agent.register_tool(calc)

        planner = Planner(agent)
        result = planner.execute_plan("Calculate something")
        assert "步骤1" in result or "calculate" in result


# ---------------------------------------------------------------------------
# BaseTool interface
# ---------------------------------------------------------------------------

class TestBaseTool:
    def test_concrete_tool_is_valid(self):
        calc = CalculatorTool()
        assert isinstance(calc, BaseTool)
        assert calc.name is not None
        assert calc.description is not None
        result = calc.run("1+1")
        assert isinstance(result, str)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseTool()
