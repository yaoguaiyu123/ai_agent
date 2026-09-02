import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from client_tools.client_tool_bridge import (
    ClientToolConnection,
    client_tool_bridge,
)
from client_tools.client_tool_protocol import (
    ClientToolResult,
    ClientToolResultAck,
)


router = APIRouter(
    prefix="/client-tools",
    tags=["Client Tools"],
)


async def client_tool_event_generator(
    client_id: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """保持一个 SSE 连接，将文件工具请求发送给 Qt。"""

    connection: ClientToolConnection = (
        client_tool_bridge.connect(client_id)
    )

    try:
        while not await request.is_disconnected():
            tool_request = await client_tool_bridge.next_request(
                connection,
                timeout=15.0,
            )

            # 当前连接已经被新连接替换。
            if tool_request is None:
                if connection.closed.is_set():
                    break

                if await request.is_disconnected():
                    break

                # SSE 心跳。
                yield ": ping\n\n"
                continue

            # 请求可能已经在等待期间超时。
            if not client_tool_bridge.is_pending(
                tool_request.request_id
            ):
                continue

            # 连接在取出请求后断开，重新放回当前连接。
            if (
                connection.closed.is_set()
                or await request.is_disconnected()
            ):
                client_tool_bridge.requeue(tool_request)
                break

            data = tool_request.model_dump(
                mode="json"
            )

            payload = (
                "event: client_tool_request\n"
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            )

            try:
                yield payload

            except (asyncio.CancelledError, GeneratorExit):
                # StreamingResponse 因客户端断开而取消生成器时，
                # 避免已经取出的请求永久丢失。
                client_tool_bridge.requeue(tool_request)
                raise

    finally:
        client_tool_bridge.disconnect(connection)


@router.get(
    "/events/{client_id}",
    response_class=StreamingResponse,
)
async def client_tool_events(
    client_id: str,
    request: Request,
) -> StreamingResponse:
    """供 Qt 客户端建立文件工具 SSE 通道。"""

    return StreamingResponse(
        client_tool_event_generator(client_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/result",
    response_model=ClientToolResultAck,
)
async def client_tool_result(
    result: ClientToolResult,
) -> ClientToolResultAck:
    """接收 Qt 执行文件工具后的结果。"""

    if not client_tool_bridge.resolve(result):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "没有找到对应的客户端工具请求，"
                "请求可能已超时或 client_id 不匹配"
            ),
        )

    return ClientToolResultAck()