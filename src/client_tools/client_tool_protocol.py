# note 定义服务端和 Qt 之间传输的数据格式
from typing import Any, Literal

from pydantic import BaseModel, Field


ClientToolName = Literal["list_directory", "read_text_file"]
ClientToolStatus = Literal["success", "error", "denied"]


class ClientToolRequest(BaseModel):
    """服务端通过 SSE 发送给 Qt 客户端的只读工具请求。"""

    type: Literal["client_tool_request"] = "client_tool_request"
    request_id: str
    client_id: str
    tool_call_id: str | None = None
    tool_name: ClientToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ClientToolResult(BaseModel):
    """Qt 客户端完成本地工具调用后返回的结果。"""

    request_id: str
    client_id: str
    status: ClientToolStatus
    output: Any = None
    error: str | None = None


class ClientToolResultAck(BaseModel):
    """服务端确认已经接收客户端工具结果。"""

    success: bool = True

