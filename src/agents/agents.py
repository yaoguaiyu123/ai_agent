from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.chatbot import chatbot
from agents.github_mcp_agent.github_mcp_agent import github_mcp_agent
from agents.knowledge_base_agent import kb_agent
from agents.lazy_agent import LazyLoadingAgent
from agents.research_assistant import research_assistant
from schema import AgentInfo
from agents.file_agent import file_agent

DEFAULT_AGENT = "chatbot"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph

AgentGraph = CompiledStateGraph | Pregel
AgentGraphLike = CompiledStateGraph | Pregel | LazyLoadingAgent


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike

# note 所有agents
agents: dict[str, Agent] = {
    "chatbot": Agent(
        description="一个简单的聊天机器人",
        graph_like=chatbot,
    ),
    "research-assistant": Agent(
        description="一个具备网页搜索和计算器功能的研究助手",
        graph_like=research_assistant,
    ),
    "knowledge-base-agent": Agent(
        description="一个使用 Amazon Bedrock Knowledge Base 的检索增强生成 Agent",
        graph_like=kb_agent,
    ),
    "github-mcp-agent": Agent(
        description="一个具备 MCP 工具的 GitHub Agent，用于仓库管理和开发工作流",
        graph_like=github_mcp_agent,
    ),
    "file_agent": Agent(
        description="一个只能读取 客户端所选工作目录的文件助手",
        graph_like=file_agent,
    ),
}

async def load_agent(agent_id: str) -> None:
    """Load lazy agents if needed."""
    graph_like = agents[agent_id].graph_like

    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """Get an agent graph, loading lazy agents if needed."""
    agent_graph = agents[agent_id].graph_like

    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")

        return agent_graph.get_graph()

    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(
            key=agent_id,
            description=agent.description,
        )
        for agent_id, agent in agents.items()
    ]