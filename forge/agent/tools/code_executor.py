import sys
import io
import traceback

from ..base_agent import BaseTool


class CodeExecutorTool(BaseTool):
    """Execute Python code safely."""

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

            exec_globals = {"__builtins__": __builtins__}
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
