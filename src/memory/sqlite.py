from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings

# tip InMemoryStore:LangGraph 提供的内存存储组件，用来给 AI Agent 提供跨对话的长期记忆。

def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """初始化并返回一个 SQLite 持久化存储实例"""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """InMemoryStore 的包装器，提供异步上下文管理器接口"""

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # InMemoryStore 不需要清理操作
        pass

    async def setup(self):
        # 空方法，为了兼容 PostgresStore
        pass


@asynccontextmanager
async def get_sqlite_store():
    """
    初始化并返回一个用于长期记忆的存储实例

    LangGraph 没有提供 SQLite 专用的存储，
    所以用 InMemoryStore 包上一层异步上下文管理器来保持接口兼容
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()
