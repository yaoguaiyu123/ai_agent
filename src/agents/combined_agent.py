import json
import logging
import operator
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import Send

from agents.file_agent import file_agent
from agents.github_mcp_agent.github_mcp_agent import github_mcp_agent
from agents.knowledge_base_agent import kb_agent
from agents.lazy_agent import LazyLoadingAgent
from agents.research_assistant import research_assistant
from core import get_model, settings


logger = logging.getLogger(__name__)

MAX_TASKS = 8
MAX_PARALLEL_TASKS = 4

SupportedAgent = Literal[
    "research-assistant",
    "file_agent",
    "knowledge-base-agent",
    "github-mcp-agent",
]


class PlannedTask(BaseModel):
    id: str = Field(description="任务唯一 ID，例如 research_1、file_1")
    agent: SupportedAgent = Field(description="负责执行该任务的 Agent")
    instruction: str = Field(description="交给子 Agent 的具体任务说明")
    depends_on: list[str] = Field(
        default_factory=list,
        description="该任务依赖的任务 ID。没有依赖时为空列表。",
    )


class ExecutionPlan(BaseModel):
    tasks: list[PlannedTask] = Field(
        default_factory=list,
        max_length=MAX_TASKS,
        description="需要执行的子任务列表。无需拆分时返回空列表。",
    )


class CombinedState(MessagesState, total=False):
    original_request: str
    execution_plan: list[dict[str, Any]]

    completed_task_ids: Annotated[list[str], operator.add]
    task_results: Annotated[list[dict[str, Any]], operator.add]

    planner_error: str
    schedule_error: str
    status: str

    # 通过 Send 传递给 worker 的局部状态
    task: dict[str, Any]
    dependency_results: list[dict[str, Any]]


_CHILD_AGENTS: dict[str, Any] = {
    "research-assistant": research_assistant,
    "file_agent": file_agent,
    "knowledge-base-agent": kb_agent,
    "github-mcp-agent": github_mcp_agent,
}


PLANNER_PROMPT = """
你是 combined_agent 的任务规划器，负责把用户请求拆分成若干个可执行子任务。

可用的子 Agent：

1. research-assistant
   负责网页搜索、资料查询、计算和研究性分析。

2. file_agent
   负责 Qt 客户端工作目录中的文件和文件夹操作。
   文件 Agent 只能通过客户端工具访问 Windows 文件系统。

3. knowledge-base-agent
   负责知识库检索和基于检索结果回答问题。

4. github-mcp-agent
   负责 GitHub 仓库、Issue、Pull Request 和代码相关操作。

规划规则：

- 如果用户的问题不需要多个专业 Agent，可以返回空任务列表。
- 每个任务必须是一个清晰、具体、可独立执行的子任务。
- 互不依赖的任务可以并行执行。
- 有先后关系的任务必须通过 depends_on 声明依赖。
- 同一个文件或目录上的创建、写入、删除、移动操作不要并行执行。
- 例如“先创建文件夹，再在文件夹中创建文件”，第二个任务必须依赖第一个任务。
- 不要把“汇总最终答案”作为子任务，最终汇总由 combined_agent 自己完成。
- 不要臆造用户没有要求的文件修改、删除或外部操作。
"""


SYNTHESIZER_PROMPT = """
你是 combined_agent 的最终汇总 Agent。

你会收到：
1. 用户原始请求；
2. 多个专业子 Agent 的执行结果；
3. 可能存在的任务失败信息。

请根据这些内容直接回答用户：

- 准确汇总已经完成的工作；
- 明确说明失败、跳过或未完成的子任务；
- 不要暴露内部的 LangGraph、Send、thread_id 或调度细节；
- 不要虚构子 Agent 没有返回的结果；
- 如果没有子任务结果，就直接根据用户请求正常回答。
"""


def _model_from_config(config: RunnableConfig):
    configurable = config.get("configurable") or {}
    return get_model(
        configurable.get("model", settings.DEFAULT_MODEL)
    )


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
                else:
                    parts.append(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                    )
            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content or "")


def _conversation_text(messages: list[Any]) -> str:
    lines: list[str] = []

    for message in messages[-12:]:
        if isinstance(message, dict):
            role = message.get("type", "message")
        else:
            role = getattr(message, "type", "message")

        content = _message_text(message)

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _last_human_request(state: CombinedState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _message_text(message)

    return ""


def _normalize_plan(
    plan: ExecutionPlan | dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(plan, dict):
        plan = ExecutionPlan.model_validate(plan)

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, task in enumerate(plan.tasks):
        task_id = task.id.strip() or f"task_{index + 1}"
        instruction = task.instruction.strip()

        if not instruction or task_id in seen_ids:
            continue

        seen_ids.add(task_id)

        normalized.append(
            {
                "id": task_id,
                "agent": str(task.agent),
                "instruction": instruction,
                "depends_on": list(task.depends_on),
            }
        )

    valid_ids = {task["id"] for task in normalized}

    for task in normalized:
        task["depends_on"] = [
            dependency
            for dependency in task["depends_on"]
            if dependency in valid_ids
            and dependency != task["id"]
        ]

    return normalized


async def plan_tasks(
    state: CombinedState,
    config: RunnableConfig,
) -> CombinedState:
    original_request = _last_human_request(state)

    if not original_request:
        return {
            "execution_plan": [],
            "planner_error": "没有找到用户请求",
            "status": "未找到用户请求",
        }

    conversation = _conversation_text(
        state.get("messages", [])
    )

    planner_config = dict(config)
    planner_config["tags"] = [
        *(config.get("tags") or []),
        "skip_stream",
    ]

    try:
        planner_model = _model_from_config(
            config
        ).with_structured_output(
            ExecutionPlan
        )

        result = await planner_model.ainvoke(
            [
                SystemMessage(
                    content=PLANNER_PROMPT
                ),
                HumanMessage(
                    content=(
                        "当前对话内容：\n"
                        f"{conversation}\n\n"
                        "请生成执行计划。"
                    )
                ),
            ],
            config=planner_config,
        )

        execution_plan = _normalize_plan(result)

        return {
            "original_request": original_request,
            "execution_plan": execution_plan,
            "status": (
                f"已制定 {len(execution_plan)} 个子任务"
                if execution_plan
                else "无需拆分子任务，准备直接回答"
            ),
        }

    except Exception:
        logger.exception("combined_agent 任务规划失败")

        return {
            "original_request": original_request,
            "execution_plan": [],
            "planner_error": "任务规划失败",
            "status": "任务规划失败，准备直接回答",
        }


def _runnable_tasks(
    state: CombinedState,
) -> list[dict[str, Any]]:
    completed = set(
        state.get(
            "completed_task_ids",
            [],
        )
    )

    runnable: list[dict[str, Any]] = []

    for task in state.get("execution_plan", []):
        task_id = task["id"]

        if task_id in completed:
            continue

        dependencies = set(
            task.get("depends_on", [])
        )

        if dependencies.issubset(completed):
            runnable.append(task)

    return runnable[:MAX_PARALLEL_TASKS]


def _dependency_results(
    task: dict[str, Any],
    state: CombinedState,
) -> list[dict[str, Any]]:
    dependency_ids = set(
        task.get("depends_on", [])
    )

    return [
        result
        for result in state.get("task_results", [])
        if result.get("task_id") in dependency_ids
    ]


def schedule_tasks(
    state: CombinedState,
) -> CombinedState:
    tasks = state.get("execution_plan", [])
    completed = set(
        state.get(
            "completed_task_ids",
            [],
        )
    )

    if not tasks:
        return {
            "status": "没有需要执行的子任务，准备生成回答",
        }

    if all(
        task["id"] in completed
        for task in tasks
    ):
        return {
            "status": "所有子任务已完成，准备汇总",
        }

    runnable = _runnable_tasks(state)

    if not runnable:
        return {
            "schedule_error": "任务依赖无法满足",
            "status": "任务依赖无法满足，准备汇总已有结果",
        }

    return {
        "status": (
            f"已并行派发 {len(runnable)} 个子任务"
        ),
    }


def dispatch_tasks(
    state: CombinedState,
):
    tasks = state.get("execution_plan", [])
    completed = set(
        state.get(
            "completed_task_ids",
            [],
        )
    )

    if (
        not tasks
        or state.get("schedule_error")
        or all(
            task["id"] in completed
            for task in tasks
        )
    ):
        return "synthesize"

    runnable = _runnable_tasks(state)

    if not runnable:
        return "synthesize"

    return [
        Send(
            "worker",
            {
                "task": task,
                "original_request": state.get(
                    "original_request",
                    "",
                ),
                "dependency_results": _dependency_results(
                    task,
                    state,
                ),
            },
        )
        for task in runnable
    ]


def _child_config(
    config: RunnableConfig,
    agent_id: str,
    task_id: str,
) -> RunnableConfig:
    child_config = dict(config)

    configurable = dict(
        config.get("configurable") or {}
    )

    parent_thread_id = str(
        configurable.get(
            "thread_id",
            uuid4(),
        )
    )

    configurable["thread_id"] = (
        f"{parent_thread_id}:combined:"
        f"{task_id}:{uuid4().hex[:8]}"
    )

    child_config["configurable"] = configurable

    metadata = dict(
        config.get("metadata") or {}
    )

    metadata.update(
        {
            "agent_id": agent_id,
            "parent_agent_id": "combined_agent",
            "task_id": task_id,
        }
    )

    child_config["metadata"] = metadata

    child_config["tags"] = [
        *(config.get("tags") or []),
        "combined_child",
        "skip_stream",
    ]

    # 子 Agent 单独生成自己的运行 ID
    child_config.pop("run_id", None)

    return child_config


def _resolve_child_graph(agent_id: str):
    candidate = _CHILD_AGENTS[agent_id]

    if isinstance(candidate, LazyLoadingAgent):
        return candidate.get_graph()

    return candidate


def _extract_child_result(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages", [])

        for message in reversed(messages):
            message_type = (
                message.get("type")
                if isinstance(message, dict)
                else getattr(message, "type", "")
            )

            if message_type not in {"ai", "assistant"}:
                continue

            content = _message_text(message)

            if content:
                return content

        for key in ("output", "final_output"):
            if result.get(key):
                return _message_text(
                    result[key]
                )

    return _message_text(result)


async def execute_task(
    state: CombinedState,
    config: RunnableConfig,
) -> CombinedState:
    task = state["task"]

    task_id = str(task["id"])
    agent_id = str(task["agent"])
    instruction = str(task["instruction"])

    dependency_results = state.get(
        "dependency_results",
        [],
    )

    dependency_text = json.dumps(
        dependency_results,
        ensure_ascii=False,
        indent=2,
    )

    child_prompt = f"""
原始用户请求：
{state.get("original_request", "")}

当前子任务：
{instruction}

前置任务结果：
{dependency_text}

你是专业执行 Agent，只处理当前子任务。
原始用户请求仅作为背景信息，不能因此擅自扩大操作范围。
请返回清晰、准确、可供总 Agent 汇总的执行结果。
"""

    try:
        child_graph = _resolve_child_graph(
            agent_id
        )

        child_result = await child_graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=child_prompt
                    )
                ]
            },
            config=_child_config(
                config,
                agent_id,
                task_id,
            ),
        )

        result_text = _extract_child_result(
            child_result
        )

        if not result_text:
            raise RuntimeError(
                "子 Agent 没有返回有效结果"
            )

        result = {
            "task_id": task_id,
            "agent": agent_id,
            "status": "success",
            "content": result_text,
        }

        return {
            "completed_task_ids": [task_id],
            "task_results": [result],
            "status": f"子任务 {task_id} 已完成",
        }

    except Exception as exc:
        logger.exception(
            "子任务执行失败: %s",
            task_id,
        )

        result = {
            "task_id": task_id,
            "agent": agent_id,
                       "status": "error",
            "content": str(exc),
        }

        return {
            "completed_task_ids": [task_id],
            "task_results": [result],
            "status": f"子任务 {task_id} 执行失败",
        }


async def synthesize(
    state: CombinedState,
    config: RunnableConfig,
) -> CombinedState:
    results = sorted(
        state.get("task_results", []),
        key=lambda item: str(
            item.get("task_id", "")
        ),
    )

    conversation = _conversation_text(
        state.get("messages", [])
    )

    result_text = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    )

    internal_errors = {
        key: value
        for key, value in {
            "planner_error": state.get(
                "planner_error"
            ),
            "schedule_error": state.get(
                "schedule_error"
            ),
        }.items()
        if value
    }

    synthesis_input = f"""
用户原始请求：
{state.get("original_request", "")}

最近对话：
{conversation}

子 Agent 执行结果：
{result_text}

调度异常：
{json.dumps(internal_errors, ensure_ascii=False)}
"""

    response = await _model_from_config(
        config
    ).ainvoke(
        [
            SystemMessage(
                content=SYNTHESIZER_PROMPT
            ),
            HumanMessage(
                content=synthesis_input
            ),
        ],
        config=config,
    )

    return {
        "messages": [response],
        "status": "处理完成",
    }


builder = StateGraph(CombinedState)

builder.add_node(
    "planner",
    plan_tasks,
)

builder.add_node(
    "schedule",
    schedule_tasks,
)

builder.add_node(
    "worker",
    execute_task,
)

builder.add_node(
    "synthesize",
    synthesize,
)

builder.set_entry_point("planner")
builder.add_edge("planner", "schedule")

builder.add_conditional_edges(
    "schedule",
    dispatch_tasks,
    ["worker", "synthesize"],
)

builder.add_edge("worker", "schedule")
builder.add_edge("synthesize", END)

combined_agent = builder.compile()