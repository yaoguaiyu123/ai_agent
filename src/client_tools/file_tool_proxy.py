# note 各种文件工具
import json
from pathlib import PureWindowsPath
from typing import Any

from langchain.tools import ToolRuntime, tool

from client_tools.client_tool_bridge import client_tool_bridge
from client_tools.client_tool_protocol import ClientToolName, ClientToolResult


def _validate_relative_path(path: str, allow_current_directory: bool) -> str:
    """对模型提供的 Windows 相对路径做服务端第一层只读安全检查。"""
    normalized = path.strip()
    if not normalized:
        if allow_current_directory:
            return "."
        raise ValueError("文件路径不能为空")
    if len(normalized) > 1024 or "\x00" in normalized:
        raise ValueError("路径格式无效")
    if "*" in normalized or "?" in normalized:
        raise ValueError("路径中不能包含通配符")
    windows_path = PureWindowsPath(normalized)
    if windows_path.is_absolute() or windows_path.drive or windows_path.root:
        raise ValueError("只能访问客户端当前工作目录中的相对路径")
    if any(part == ".." for part in windows_path.parts):
        raise ValueError("路径不能使用 .. 离开当前工作目录")
    return str(windows_path)


def _get_client_id(runtime: ToolRuntime) -> str:
    """从当前 Agent 调用配置中获取与 Qt 一致的 user_id。"""
    client_id = runtime.config.get("configurable", {}).get("user_id")
    if not client_id:
        raise RuntimeError("当前请求缺少 user_id，无法找到对应的 Qt 客户端")
    return str(client_id)


def _format_result(result: ClientToolResult) -> str:
    """把客户端结构化结果转换成适合大模型读取的工具文本。"""
    if result.status != "success":
        raise RuntimeError(result.error or "客户端文件工具执行失败")
    if isinstance(result.output, str):
        return result.output
    return json.dumps(result.output, ensure_ascii=False, indent=2)


async def _call_client_tool(
    runtime: ToolRuntime,
    tool_name: ClientToolName,
    arguments: dict[str, Any],
) -> str:
    """将服务端 Tool 调用转发给当前 Qt 客户端。"""
    result = await client_tool_bridge.request(
        client_id=_get_client_id(runtime),
        tool_name=tool_name,
        arguments=arguments,
        tool_call_id=runtime.tool_call_id,
    )
    return _format_result(result)


@tool
async def list_directory(runtime: ToolRuntime, path: str = ".") -> str:
    """列出客户端当前工作目录内指定文件夹的直接子项。

    Args:
        path: 相对于客户端当前工作目录的文件夹路径，默认值为当前工作目录。
    """
    safe_path = _validate_relative_path(path, allow_current_directory=True)
    return await _call_client_tool(runtime, "list_directory", {"path": safe_path})


@tool
async def read_text_file(runtime: ToolRuntime, path: str) -> str:
    """读取客户端当前工作目录内的一个文本文件。

    Args:
        path: 相对于客户端当前工作目录的文件路径。
    """
    safe_path = _validate_relative_path(path, allow_current_directory=False)
    return await _call_client_tool(runtime, "read_text_file", {"path": safe_path})


file_tools = [list_directory, read_text_file]

