import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from client_tools.client_tool_protocol import (
    ClientToolName,
    ClientToolRequest,
    ClientToolResult,
)


class ClientNotConnectedError(RuntimeError):
    """目标 Qt 客户端当前没有建立工具 SSE 连接。"""


class ClientToolTimeoutError(RuntimeError):
    """等待 Qt 客户端返回工具结果超时。"""


@dataclass(slots=True)
class PendingClientToolRequest:
    """保存正在等待客户端返回的异步请求。"""

    client_id: str
    future: asyncio.Future[ClientToolResult]


@dataclass(slots=True)
class ClientToolConnection:
    """代表一个具体的 SSE 连接。"""

    client_id: str
    connection_id: str
    queue: asyncio.Queue[ClientToolRequest]
    closed: asyncio.Event


class ClientToolBridge:
    """在 LangGraph Tool 和 FastAPI SSE 路由之间传递客户端工具请求。"""

    def __init__(self) -> None:
        self._connections: dict[str, ClientToolConnection] = {}
        self._connection_events: dict[str, asyncio.Event] = {}
        self._pending_requests: dict[str, PendingClientToolRequest] = {}

    def connect(self, client_id: str) -> ClientToolConnection:
        """
        建立一个新的 SSE 连接。

        同一个 client_id 只允许一个有效连接。
        新连接建立时，旧连接会被标记为关闭。
        """
        old_connection = self._connections.get(client_id)

        if old_connection is not None:
            old_connection.closed.set()

        connection = ClientToolConnection(
            client_id=client_id,
            connection_id=str(uuid4()),
            queue=asyncio.Queue(),
            closed=asyncio.Event(),
        )

        self._connections[client_id] = connection

        connected_event = self._connection_events.setdefault(
            client_id,
            asyncio.Event(),
        )
        connected_event.set()

        # 将旧连接队列中尚未发送的请求转移到新连接。
        if old_connection is not None:
            while True:
                try:
                    pending_request = old_connection.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if self.is_pending(pending_request.request_id):
                    connection.queue.put_nowait(pending_request)

        return connection

    def disconnect(self, connection: ClientToolConnection) -> None:
        """断开指定的 SSE 连接。"""

        connection.closed.set()

        current_connection = self._connections.get(connection.client_id)

        # 只有当前有效连接断开时，才清除 client_id 的连接状态。
        if current_connection is connection:
            self._connections.pop(connection.client_id, None)

            connected_event = self._connection_events.get(
                connection.client_id
            )

            if connected_event is not None:
                connected_event.clear()

    def is_connected(self, client_id: str) -> bool:
        """判断客户端是否存在当前有效的 SSE 连接。"""

        connection = self._connections.get(client_id)

        return (
            connection is not None
            and not connection.closed.is_set()
        )

    def is_pending(self, request_id: str) -> bool:
        """判断工具请求是否仍在等待结果。"""

        return request_id in self._pending_requests

    async def _wait_for_connection(
        self,
        client_id: str,
        timeout: float,
    ) -> ClientToolConnection:
        """
        等待 Qt 建立 SSE 连接。

        不再在未连接时直接把请求放入无消费者的队列。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(timeout, 10.0)

        connected_event = self._connection_events.setdefault(
            client_id,
            asyncio.Event(),
        )

        while True:
            connection = self._connections.get(client_id)

            if (
                connection is not None
                and not connection.closed.is_set()
            ):
                return connection

            remaining = deadline - loop.time()

            if remaining <= 0:
                raise ClientNotConnectedError(
                    "Qt 客户端文件工具通道未连接，请确认客户端已经启动"
                )

            try:
                await asyncio.wait_for(
                    connected_event.wait(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                raise ClientNotConnectedError(
                    "等待 Qt 客户端文件工具通道连接超时"
                ) from exc

    async def next_request(
        self,
        connection: ClientToolConnection,
        timeout: float = 15.0,
    ) -> ClientToolRequest | None:
        """
        供 SSE 路由等待下一条工具请求。

        同时监听：
        - 工具请求队列
        - 当前连接是否被新连接替换
        """
        while True:
            if connection.closed.is_set():
                return None

            current_connection = self._connections.get(
                connection.client_id
            )

            if current_connection is not connection:
                return None

            request_task = asyncio.create_task(
                connection.queue.get()
            )

            closed_task = asyncio.create_task(
                connection.closed.wait()
            )

            done, pending = await asyncio.wait(
                {request_task, closed_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if pending:
                await asyncio.gather(
                    *pending,
                    return_exceptions=True,
                )

            # 超时，交给 SSE 发送心跳。
            if not done:
                return None

            # 连接已关闭或已经被新连接替换。
            if (
                closed_task in done
                or self._connections.get(connection.client_id)
                is not connection
            ):
                if (
                    request_task in done
                    and not request_task.cancelled()
                ):
                    request = request_task.result()
                    self.requeue(request)

                return None

            if request_task not in done:
                continue

            request = request_task.result()

            # 超时后的旧请求不再发送给 Qt。
            if not self.is_pending(request.request_id):
                continue

            return request

    async def request(
        self,
        client_id: str,
        tool_name: ClientToolName,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
        timeout: float = 20.0,
    ) -> ClientToolResult:
        """向 Qt 客户端发送工具请求，并等待执行结果。"""

        connection = await self._wait_for_connection(
            client_id,
            timeout=timeout,
        )

        request_id = str(uuid4())

        future = asyncio.get_running_loop().create_future()

        self._pending_requests[request_id] = (
            PendingClientToolRequest(
                client_id=client_id,
                future=future,
            )
        )

        tool_request = ClientToolRequest(
            request_id=request_id,
            client_id=client_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        try:
            # Queue 默认无上限，put_nowait 不会阻塞事件循环。
            connection.queue.put_nowait(tool_request)

            # 如果入队瞬间连接刚好被替换，
            # 将请求转移到新连接。
            if (
                self._connections.get(client_id) is not connection
                or connection.closed.is_set()
            ):
                self.requeue(tool_request)

            return await asyncio.wait_for(
                future,
                timeout=timeout,
            )

        except asyncio.TimeoutError as exc:
            raise ClientToolTimeoutError(
                "等待 Qt 客户端执行文件工具超时"
            ) from exc

        finally:
            self._pending_requests.pop(
                request_id,
                None,
            )

    def requeue(self, request: ClientToolRequest) -> bool:
        """
        将尚未完成的请求重新放入当前有效连接。

        用于旧 SSE 连接断开时避免请求丢失。
        """
        if not self.is_pending(request.request_id):
            return False

        connection = self._connections.get(request.client_id)

        if (
            connection is None
            or connection.closed.is_set()
        ):
            return False

        connection.queue.put_nowait(request)
        return True

    def resolve(self, result: ClientToolResult) -> bool:
        """根据 request_id 将 Qt 返回结果交给等待中的 Tool。"""

        pending = self._pending_requests.get(
            result.request_id
        )

        if pending is None:
            return False

        if pending.future.done():
            return False

        if pending.client_id != result.client_id:
            return False

        pending.future.set_result(result)

        return True


client_tool_bridge = ClientToolBridge()