import math
import re

from ..base_agent import BaseTool


class CalculatorTool(BaseTool):
    """Safe math expression evaluator."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算数学表达式。输入一个数学表达式，返回计算结果。支持基本运算和常用数学函数。"

    def run(self, input_text: str) -> str:
        try:
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "pi": math.pi, "e": math.e, "pow": pow,
            }
            expression = input_text.strip()
            expression = re.sub(r"[^0-9+\-*/().a-z_,\s]", "", expression)
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
