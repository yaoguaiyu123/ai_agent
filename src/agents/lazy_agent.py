"""支持异步初始化和动态创建 graph 的 Agent 类型。"""

from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


class LazyLoadingAgent(ABC):
    """需要异步加载的 Agent 的基类。"""

    def __init__(self) -> None:
        """初始化 Agent。"""
        self._loaded = False
        self._graph: CompiledStateGraph | Pregel | None = None

    @abstractmethod
    async def load(self) -> None:
        """
        执行此 Agent 的异步加载。

        此方法在服务启动时调用，应处理以下内容：
        - 建立外部连接（MCP 客户端、数据库等）
        - 加载工具或资源
        - 其他需要的异步初始化
        - 创建 Agent 的 graph
        """
        raise NotImplementedError  # pragma: no cover

    def get_graph(self) -> CompiledStateGraph | Pregel:
        """
        获取 Agent 的 graph。

        返回在 load() 中创建的 graph 实例。

        Returns:
            Agent 的 graph（CompiledStateGraph 或 Pregel）
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")
        if self._graph is None:
            raise RuntimeError("Agent graph not created during load().")
        return self._graph