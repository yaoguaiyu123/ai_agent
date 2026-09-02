from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from client_tools.file_tool_proxy import file_tools
from core import get_model, settings


class FileAgentState(MessagesState, total=False):
    """文件 Agent 的消息状态和剩余执行步数。"""

    remaining_steps: RemainingSteps


instructions = """
你是一个 Windows 本地文件阅读助手。
你只能通过工具读取用户在 Qt 客户端中选择的工作目录，不能访问工作目录之外的路径。
你目前只有以下能力：
- 列出工作目录或其子目录中的直接子项。
- 读取工作目录中的文本文件。
你没有新建、修改、移动、删除文件或执行任意命令的权限。
工具中的路径必须使用相对于工作目录的路径，不能使用绝对路径或 ..。
当用户没有提供准确文件名时，先使用 list_directory 查看目录，再决定需要读取的文件。
只能依据工具真正返回的内容回答，不得虚构文件、目录或文件内容。
如果客户端未连接、拒绝访问、文件不是文本、文件过大或读取失败，应如实说明原因。
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[FileAgentState, AIMessage]:
    """为当前模型绑定只读客户端文件工具和系统提示词。"""
    bound_model = model.bind_tools(file_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="FileAgentStateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: FileAgentState, config: RunnableConfig) -> FileAgentState:
    """调用模型，并在模型提出工具调用时交给后续 tools 节点。"""
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await wrap_model(model).ainvoke(state, config)
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="处理该文件请求需要更多步骤，请缩小问题范围后重试。",
                )
            ]
        }
    return {"messages": [response]}


def pending_tool_calls(state: FileAgentState) -> Literal["tools", "done"]:
    """根据模型是否产生工具调用决定继续执行工具还是结束。"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "done"


builder = StateGraph(FileAgentState)
builder.add_node("model", acall_model)
builder.add_node("tools", ToolNode(file_tools, handle_tool_errors=True))
builder.set_entry_point("model")
builder.add_edge("tools", "model")
builder.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

file_agent = builder.compile()
