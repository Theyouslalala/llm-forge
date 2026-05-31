import ast
import math
import operator

from ..base_agent import BaseTool


class CalculatorTool(BaseTool):
    """Safe math expression evaluator using AST parsing."""

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    SAFE_FUNCTIONS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "pow": pow,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算数学表达式。输入一个数学表达式，返回计算结果。支持基本运算和常用数学函数。"

    def _safe_eval(self, node):
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in self.SAFE_OPERATORS:
            return self.SAFE_OPERATORS[type(node.op)](self._safe_eval(node.left), self._safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in self.SAFE_OPERATORS:
            return self.SAFE_OPERATORS[type(node.op)](self._safe_eval(node.operand))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.SAFE_FUNCTIONS:
            args = [self._safe_eval(arg) for arg in node.args]
            return self.SAFE_FUNCTIONS[node.func.id](*args)
        elif isinstance(node, ast.Name) and node.id in self.SAFE_FUNCTIONS:
            return self.SAFE_FUNCTIONS[node.id]
        else:
            raise ValueError(f"不支持的表达式: {ast.dump(node)}")

    def run(self, input_text: str) -> str:
        try:
            expression = input_text.strip()
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval(tree)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
