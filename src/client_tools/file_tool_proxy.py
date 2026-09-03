# note 各种文件工具
import json
from pathlib import PureWindowsPath
from typing import Any

from langchain.tools import ToolRuntime, tool

from client_tools.client_tool_bridge import client_tool_bridge
from client_tools.client_tool_protocol import (
    ClientToolName,
    ClientToolResult,
)


MAX_TEXT_SIZE = 5 * 1024 * 1024


def _validate_relative_path(
    path: str,
    allow_current_directory: bool,
) -> str:
    """
    检查模型提供的路径。

    这里只允许相对于 Qt 工作目录的 Windows 路径。
    """

    if not isinstance(path, str):
        raise ValueError("路径必须是字符串")

    normalized = path.strip()

    if not normalized:
        if allow_current_directory:
            return "."

        raise ValueError("文件路径不能为空")

    if len(normalized) > 1024:
        raise ValueError("路径长度超过限制")

    if "\x00" in normalized:
        raise ValueError("路径包含非法字符")

    if "*" in normalized or "?" in normalized:
        raise ValueError("路径中不能包含通配符")

    windows_path = PureWindowsPath(normalized)

    if windows_path.is_absolute():
        raise ValueError(
            "只能访问客户端工作目录中的相对路径"
        )

    if windows_path.drive:
        raise ValueError(
            "路径不能包含盘符"
        )

    if windows_path.root:
        raise ValueError(
            "路径不能是根目录"
        )

    if any(
        part == ".."
        for part in windows_path.parts
    ):
        raise ValueError(
            "路径不能使用 .. 离开工作目录"
        )

    result = str(windows_path)

    if (
        not allow_current_directory
        and result in {".", ""}
    ):
        raise ValueError(
            "此操作必须指定具体文件路径"
        )

    return result


def _validate_text_content(content: str) -> str:
    """检查写入内容。"""

    if not isinstance(content, str):
        raise ValueError("文件内容必须是字符串")

    content_size = len(
        content.encode("utf-8")
    )

    if content_size > MAX_TEXT_SIZE:
        raise ValueError(
            "写入内容超过 5 MB 限制"
        )

    return content


def _get_client_id(runtime: ToolRuntime) -> str:
    """从当前 Agent 配置中获取 Qt 客户端 ID。"""

    configurable = runtime.config.get(
        "configurable",
        {},
    )

    client_id = configurable.get(
        "user_id"
    )

    if not client_id:
        raise RuntimeError(
            "当前请求缺少 user_id，"
            "无法找到对应的 Qt 客户端"
        )

    return str(client_id)


def _format_result(
    result: ClientToolResult,
) -> str:
    """将客户端结构化结果转换为模型可读取的文本。"""

    if result.status != "success":
        raise RuntimeError(
            result.error
            or "客户端文件工具执行失败"
        )

    if isinstance(result.output, str):
        return result.output

    return json.dumps(
        result.output,
        ensure_ascii=False,
        indent=2,
    )


async def _call_client_tool(
    runtime: ToolRuntime,
    tool_name: ClientToolName,
    arguments: dict[str, Any],
) -> str:
    """将服务端工具调用转发给 Qt 客户端。"""

    result = await client_tool_bridge.request(
        client_id=_get_client_id(runtime),
        tool_name=tool_name,
        arguments=arguments,
        tool_call_id=runtime.tool_call_id,
    )

    return _format_result(result)


@tool
async def list_directory(
    runtime: ToolRuntime,
    path: str = ".",
) -> str:
    """列出客户端工作目录内指定文件夹的直接子项。

    Args:
        path: 相对于客户端工作目录的文件夹路径。
            默认值为当前工作目录。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=True,
    )

    return await _call_client_tool(
        runtime,
        "list_directory",
        {
            "path": safe_path,
        },
    )


@tool
async def read_text_file(
    runtime: ToolRuntime,
    path: str,
) -> str:
    """读取客户端工作目录中的一个 UTF-8 文本文件。

    Args:
        path: 相对于客户端工作目录的文本文件路径。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    return await _call_client_tool(
        runtime,
        "read_text_file",
        {
            "path": safe_path,
        },
    )


@tool
async def write_text_file(
    runtime: ToolRuntime,
    path: str,
    content: str,
) -> str:
    """创建或覆盖客户端工作目录中的 UTF-8 文本文件。

    只有用户明确要求创建或覆盖文件时才能调用。
    如果目标文件的父目录不存在，客户端应返回错误。

    Args:
        path: 相对于客户端工作目录的目标文件路径。
        content: 要写入的完整文本内容。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    safe_content = _validate_text_content(
        content
    )

    return await _call_client_tool(
        runtime,
        "write_text_file",
        {
            "path": safe_path,
            "content": safe_content,
        },
    )


@tool
async def append_text_file(
    runtime: ToolRuntime,
    path: str,
    content: str,
) -> str:
    """向客户端工作目录中的 UTF-8 文本文件末尾追加内容。

    如果文件不存在，客户端可以创建该文件。
    该操作不能覆盖原有内容。

    Args:
        path: 相对于客户端工作目录的目标文件路径。
        content: 要追加的文本内容。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    safe_content = _validate_text_content(
        content
    )

    return await _call_client_tool(
        runtime,
        "append_text_file",
        {
            "path": safe_path,
            "content": safe_content,
        },
    )


@tool
async def delete_file(
    runtime: ToolRuntime,
    path: str,
) -> str:
    """删除客户端工作目录中的一个文件。

    只允许删除文件，不允许删除文件夹。
    只有用户明确要求删除时才能调用。

    Args:
        path: 相对于客户端工作目录的目标文件路径。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    return await _call_client_tool(
        runtime,
        "delete_file",
        {
            "path": safe_path,
        },
    )


@tool
async def move_file(
    runtime: ToolRuntime,
    source_path: str,
    destination_path: str,
) -> str:
    """移动或重命名客户端工作目录中的文件。

    source_path 和 destination_path 都必须是相对路径。
    不允许跨出工作目录。
    默认不应覆盖已有目标文件。

    Args:
        source_path: 原文件相对于工作目录的路径。
        destination_path: 新文件相对于工作目录的路径。
    """

    safe_source_path = _validate_relative_path(
        source_path,
        allow_current_directory=False,
    )

    safe_destination_path = _validate_relative_path(
        destination_path,
        allow_current_directory=False,
    )

    if safe_source_path == safe_destination_path:
        raise ValueError(
            "源路径和目标路径不能相同"
        )

    return await _call_client_tool(
        runtime,
        "move_file",
        {
            "source_path": safe_source_path,
            "destination_path": safe_destination_path,
        },
    )


@tool
async def create_file(
    runtime: ToolRuntime,
    path: str,
) -> str:
    """在客户端工作目录中创建一个空文件。

    如果目标已经存在，操作会失败，不会覆盖已有文件或目录。
    目标文件的父目录必须已经存在。

    Args:
        path: 相对于客户端工作目录的新文件路径。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    return await _call_client_tool(
        runtime,
        "create_file",
        {
            "path": safe_path,
        },
    )


@tool
async def create_directory(
    runtime: ToolRuntime,
    path: str,
) -> str:
    """在客户端工作目录中创建一个文件夹。

    如果目标已经存在，操作会失败，不会覆盖已有文件或目录。
    目标文件夹的父目录必须已经存在。

    Args:
        path: 相对于客户端工作目录的新文件夹路径。
    """

    safe_path = _validate_relative_path(
        path,
        allow_current_directory=False,
    )

    return await _call_client_tool(
        runtime,
        "create_directory",
        {
            "path": safe_path,
        },
    )

file_tools = [
    list_directory,
    read_text_file,
    write_text_file,
    append_text_file,
    delete_file,
    move_file,
    create_file,
    create_directory,
]