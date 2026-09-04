import logging
import os
from typing import Any, cast

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from core import get_model, settings

logger = logging.getLogger(__name__)


# 定义状态
class AgentState(MessagesState, total=False):
    """知识库 Agent 的状态。"""

    remaining_steps: RemainingSteps
    retrieved_documents: list[dict[str, Any]]
    kb_documents: str


# 创建 Retriever
def get_kb_retriever():
    """创建并返回一个 Knowledge Base Retriever 实例。"""
    # 从环境变量获取 Knowledge Base ID
    kb_id = os.environ.get("AWS_KB_ID", "")
    if not kb_id:
        raise ValueError("必须设置 AWS_KB_ID 环境变量")

    # 使用指定的 Knowledge Base ID 创建 Retriever
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    return retriever


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """用知识库 Agent 的系统提示词包装模型。"""

    def create_system_message(state):
        base_prompt = """你是一个有用的助手，能够基于检索到的文档提供准确的信息。

        你会收到一个查询以及从知识库中检索到的相关文档。请使用这些文档来辅助你的回答。

        遵循以下指南：
        1. 主要基于检索到的文档来回答
        2. 如果文档中包含答案，请清晰简洁地提供
        3. 如果文档不足，请说明你没有足够的信息
        4. 不要编造文档中没有的事实或信息
        5. 引用具体信息时请注明源文档
        6. 如果文档内容相互矛盾，请指出并解释不同的观点

        请以清晰、对话式的方式组织你的回答。适当使用 markdown 格式。
        """

        # 检查是否检索到了文档
        if "kb_documents" in state:
            # 将文档信息追加到系统提示词中
            document_prompt = f"\n\n我检索到了以下可能与查询相关的文档：\n\n{state['kb_documents']}\n\n请使用这些文档来回答用户的查询。仅使用这些文档中的信息，不确定时请明确说明。"
            return [SystemMessage(content=base_prompt + document_prompt)] + state["messages"]
        else:
            # 未检索到文档
            no_docs_prompt = (
                "\n\n在知识库中未找到与该查询相关的文档。"
            )
            return [SystemMessage(content=base_prompt + no_docs_prompt)] + state["messages"]

    preprocessor = RunnableLambda(
        create_system_message,
        name="StateModifier",
    )
    return RunnableSequence(preprocessor, model)


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """从知识库中检索相关文档。"""
    # 获取最后一条用户消息
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        # 保留原始状态中的消息
        return {"messages": [], "retrieved_documents": []}

    # 使用最后一条用户消息作为查询
    query = cast(str, human_messages[-1].content)

    try:
        # 初始化 Retriever
        retriever = get_kb_retriever()

        # 检索文档
        retrieved_docs = await retriever.ainvoke(query)

        # 为状态创建文档摘要
        document_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            summary = {
                "id": doc.metadata.get("id", f"doc-{i}"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", f"Document {i}"),
                "content": doc.page_content,
                "relevance_score": doc.metadata.get("score", 0),
            }
            document_summaries.append(summary)

        logger.info(f"查询: {query[:50]}... 检索到 {len(document_summaries)} 份文档")

        return {"retrieved_documents": document_summaries, "messages": []}

    except Exception as e:
        logger.error(f"检索文档时出错: {str(e)}")
        return {"retrieved_documents": [], "messages": []}


async def prepare_augmented_prompt(state: AgentState, config: RunnableConfig) -> AgentState:
    """准备包含检索文档内容的增强提示词。"""
    # 获取检索到的文档
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"messages": []}

    # 将检索到的文档格式化供模型使用
    formatted_docs = "\n\n".join(
        [
            f"--- 文档 {i + 1} ---\n"
            f"来源: {doc.get('source', 'Unknown')}\n"
            f"标题: {doc.get('title', 'Unknown')}\n\n"
            f"{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]
    )

    # 将格式化后的文档存入状态
    return {"kb_documents": formatted_docs, "messages": []}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """基于检索到的文档生成回答。"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


# 定义 Graph
agent = StateGraph(AgentState)

# 添加节点
agent.add_node("retrieve_documents", retrieve_documents)
agent.add_node("prepare_augmented_prompt", prepare_augmented_prompt)
agent.add_node("model", acall_model)

# 设置入口
agent.set_entry_point("retrieve_documents")

# 添加边定义流程
agent.add_edge("retrieve_documents", "prepare_augmented_prompt")
agent.add_edge("prepare_augmented_prompt", "model")
agent.add_edge("model", END)

# 编译 Agent
kb_agent = agent.compile()