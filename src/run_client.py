# note 一个简单的命令行客户端示例

import asyncio

from client import AgentClient
from core import settings
from schema import ChatMessage


# tip async def amain() 定义了一个协程 coroutine ，协程只是由事件循环 event loop 负责调度
async def amain() -> None:
    client = AgentClient(settings.BASE_URL)

    print("Agent 信息：")
    print(client.info)

    print("\n普通对话示例：")
    response = await client.ainvoke(    # tip await会告诉event loop这里现在要等待，这段时间先去执行别的 coroutine
        "给我讲一个简短的笑话",
        model=settings.DEFAULT_MODEL,
    )
    response.pretty_print()

    print("\n流式输出示例：")
    async for message in client.astream(  # tip async for = 一个内部自带 await 的异步版 for
        "分享一个有趣的小知识",
        model=settings.DEFAULT_MODEL,
    ):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"错误：未知消息类型 - {type(message)}")


def main() -> None:
    client = AgentClient(settings.BASE_URL)

    print("Agent 信息：")
    print(client.info)

    print("\n普通对话示例：")
    response = client.invoke(
        "给我讲一个简短的笑话",
        model=settings.DEFAULT_MODEL,
    )
    response.pretty_print()

    print("\n流式输出示例：")
    for message in client.stream(
        "分享一个有趣的小知识",
        model=settings.DEFAULT_MODEL,
    ):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"错误：未知消息类型 - {type(message)}")


if __name__ == "__main__":
    # print("正在运行同步模式")
    # main()

    print("\n\n")
    print("正在运行异步模式")
    asyncio.run(amain())