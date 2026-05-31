import sys
import io
import traceback

from ..base_agent import BaseTool


class CodeExecutorTool(BaseTool):
    """Execute Python code with restricted builtins."""

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "执行Python代码。输入Python代码字符串，返回执行结果。支持基本的计算和字符串操作。"

    def run(self, input_text: str) -> str:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            code = input_text.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            # Restricted builtins - no dangerous functions
            safe_builtins = {
                "print": print, "len": len, "range": range, "int": int,
                "float": float, "str": str, "bool": bool, "list": list,
                "dict": dict, "tuple": tuple, "set": set, "sorted": sorted,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                "type": type, "repr": repr, "any": any, "all": all,
            }
            exec_globals = {"__builtins__": safe_builtins}
            exec(code, exec_globals)

            stdout_output = sys.stdout.getvalue()
            stderr_output = sys.stderr.getvalue()

            result = ""
            if stdout_output:
                result += f"输出:\n{stdout_output}"
            if stderr_output:
                result += f"警告/错误:\n{stderr_output}"
            if not result:
                result = "代码执行成功，无输出。"
            return result

        except Exception:
            error = traceback.format_exc()
            return f"执行错误:\n{error}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
