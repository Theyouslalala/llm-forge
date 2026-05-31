import os
from pathlib import Path

from ..base_agent import BaseTool


class FileReaderTool(BaseTool):
    """Read file contents with path validation."""

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".cfg", ".ini", ".toml"}

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "读取文件内容。输入文件路径，返回文件内容。支持txt、md、py等文本文件。"

    def run(self, input_text: str) -> str:
        file_path = input_text.strip().strip('"').strip("'")
        try:
            path = Path(file_path).resolve()

            if not path.exists():
                return f"文件不存在: {file_path}"

            if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                return f"不允许读取 {path.suffix} 类型的文件。允许的类型: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"

            if path.stat().st_size > 100000:
                return f"文件过大 ({path.stat().st_size} bytes)，请指定读取范围。"

            with open(path, encoding="utf-8") as f:
                content = f.read()

            return f"文件: {file_path}\n大小: {len(content)} 字符\n\n内容:\n{content[:5000]}"

        except UnicodeDecodeError:
            return "无法读取文件: 可能是二进制文件"
        except Exception as e:
            return f"读取错误: {e}"
