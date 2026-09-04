"""GitHub MCP Agent - 使用 GitHub MCP 工具进行仓库管理的 Agent。"""

import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, StreamableHttpConnection
from langgraph.graph.state import CompiledStateGraph

from agents.lazy_agent import LazyLoadingAgent
from core import get_model, settings

logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%B %d, %Y")
prompt = f"""
你是 GitHubBot，一个专门用于 GitHub 仓库管理和开发工作流的助手。
你可以使用 GitHub MCP 工具来操作 GitHub 仓库、Issue、Pull Request 以及其他 GitHub 资源。今天是 {current_date}。

你的能力包括：
- 仓库管理（创建、克隆、浏览）
- Issue 管理（创建、列表、更新、关闭）
- Pull Request 管理（创建、审查、合并）
- 分支管理（创建、切换、合并）
- 文件操作（读取、写入、搜索）
- Commit 操作（创建、查看历史）

使用指南：
- 始终保持友好，对 GitHub 操作提供清晰的解释
- 创建或修改内容时，确保遵循最佳实践
- 对破坏性操作（删除、force push 等）要格外谨慎
- 说明你正在做什么以及为什么这样做
- 使用恰当的 commit message 和 PR 描述
- 遵守仓库权限和访问控制

注意：你可以使用 GitHub MCP 工具直接访问 GitHub API。
"""


class GitHubMCPAgent(LazyLoadingAgent):
    """支持异步初始化的 GitHub MCP Agent。"""

    def __init__(self) -> None:
        super().__init__()
        self._mcp_tools: list[BaseTool] = []
        self._mcp_client: MultiServerMCPClient | None = None

    async def load(self) -> None:
        """通过加载 MCP 工具来初始化 GitHub MCP Agent。"""
        if not settings.GITHUB_PAT:
            logger.info("GITHUB_PAT 未设置，GitHub MCP Agent 将没有可用工具")
            self._mcp_tools = []
            self._graph = self._create_graph()
            self._loaded = True
            return

        try:
            # 初始化 MCP 客户端
            github_pat = settings.GITHUB_PAT.get_secret_value()
            connections: dict[str, Connection] = {
                "github": StreamableHttpConnection(
                    transport="streamable_http",
                    url=settings.MCP_GITHUB_SERVER_URL,
                    headers={
                        "Authorization": f"Bearer {github_pat}",
                    },
                )
            }

            self._mcp_client = MultiServerMCPClient(connections)
            logger.info("MCP 客户端初始化成功")

            # 从客户端获取工具
            self._mcp_tools = await self._mcp_client.get_tools()
            logger.info(f"GitHub MCP Agent 初始化完成，共 {len(self._mcp_tools)} 个工具")

        except Exception as e:
            logger.error(f"GitHub MCP Agent 初始化失败: {e}")
            self._mcp_tools = []
            self._mcp_client = None

        # 创建并存储 graph
        self._graph = self._create_graph()
        self._loaded = True

    def _create_graph(self) -> CompiledStateGraph:
        """创建 GitHub MCP Agent 的 graph。"""
        model = get_model(settings.DEFAULT_MODEL)

        return create_agent(
            model=model,
            tools=self._mcp_tools,
            name="github-mcp-agent",
            system_prompt=prompt,
        )


# 创建 Agent 实例
github_mcp_agent = GitHubMCPAgent()