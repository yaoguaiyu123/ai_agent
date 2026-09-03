from typing import Literal

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import (
    END,
    MessagesState,
    StateGraph,
)
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from client_tools.file_tool_proxy import file_tools
from core import get_model, settings


class FileAgentState(
    MessagesState,
    total=False,
):
    """文件 Agent 的消息状态和剩余执行步数。"""

    remaining_steps: RemainingSteps


instructions = """
你是一个 Windows 本地文件助手。

所有文件操作都只能通过工具完成。
你只能访问用户在 Qt 客户端中选择的工作目录，
不能访问工作目录之外的路径。

你当前拥有以下能力：

1. list_directory
   列出工作目录或子目录中的直接子项。

2. read_text_file
   读取 UTF-8 文本文件内容。

3. write_text_file
   创建新文件，或覆盖写入已有文件。
   只有用户明确要求创建、写入或覆盖时才能调用。

4. append_text_file
   向已有文件末尾追加内容。
   不能使用它覆盖原文件内容。

5. delete_file
   删除文件。
   只有用户明确要求删除时才能调用。
   不能删除目录。

6. move_file
   移动文件或重命名文件。
   source_path 和 destination_path 都必须使用相对路径。

7. create_file
   创建一个空文件。
   如果目标已经存在，不能覆盖。
   目标文件的父目录必须已经存在。

8. create_directory
   创建一个文件夹。
   如果目标已经存在，不能覆盖。
   目标文件夹的父目录必须已经存在。

所有工具路径必须是相对于工作目录的路径。
不能使用绝对路径、盘符或 ..。

当用户没有提供准确的文件名时，
先使用 list_directory 查看目录，
再决定后续操作。

如果用户只是询问文件内容，
只能使用 read_text_file，
不能修改文件。

如果用户要求写入文件，
先确认目标路径和要写入的内容。
如果用户没有明确指定内容，
不能擅自生成并覆盖文件。

如果用户要求删除文件，
必须使用 delete_file，
不能通过其他工具模拟删除。

如果用户要求移动或重命名文件，
使用 move_file。

如果用户要求创建空文件，
使用 create_file，不要使用 write_text_file 代替。

如果用户要求创建文件夹，
使用 create_directory。

不能虚构工具返回的文件、目录或文件内容。
只能依据工具真正返回的结果回答。

如果客户端未连接、拒绝访问、路径非法、
文件不存在、目标已存在、文件不是文本、
文件过大或操作失败，
必须如实说明具体原因。
"""


def wrap_model(
    model: BaseChatModel,
) -> RunnableSerializable[
    FileAgentState,
    AIMessage,
]:
    """绑定文件工具和系统提示词。"""

    bound_model = model.bind_tools(
        file_tools
    )

    preprocessor = RunnableLambda(
        lambda state: [
            SystemMessage(
                content=instructions
            )
        ]
        + state["messages"],
        name="FileAgentStateModifier",
    )

    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(
    state: FileAgentState,
    config: RunnableConfig,
) -> FileAgentState:
    """调用模型并处理工具调用。"""

    model = get_model(
        config["configurable"].get(
            "model",
            settings.DEFAULT_MODEL,
        )
    )

    response = await wrap_model(
        model
    ).ainvoke(
        state,
        config,
    )

    if (
        state["remaining_steps"] < 2
        and response.tool_calls
    ):
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content=(
                        "处理该文件请求需要更多步骤，"
                        "请缩小问题范围后重试。"
                    ),
                )
            ]
        }

    return {
        "messages": [response]
    }


def pending_tool_calls(
    state: FileAgentState,
) -> Literal["tools", "done"]:
    """判断模型是否请求调用文件工具。"""

    last_message = state["messages"][-1]

    if not isinstance(
        last_message,
        AIMessage,
    ):
        raise TypeError(
            f"Expected AIMessage, got {type(last_message)}"
        )

    if last_message.tool_calls:
        return "tools"

    return "done"


builder = StateGraph(
    FileAgentState
)

builder.add_node(
    "model",
    acall_model,
)

builder.add_node(
    "tools",
    ToolNode(
        file_tools,
        handle_tool_errors=True,
    ),
)

builder.set_entry_point(
    "model"
)

builder.add_edge(
    "tools",
    "model"
)

builder.add_conditional_edges(
    "model",
    pending_tool_calls,
    {
        "tools": "tools",
        "done": END,
    },
)

file_agent = builder.compile()