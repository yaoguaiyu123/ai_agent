import math
import re

import numexpr
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """使用 numexpr 计算数学表达式

        适用于需要借助 numexpr 回答数学问题的场景
        此工具仅用于数学问题，不处理其他内容。只接受数学表达式作为输入

        Args:
            expression (str): 一个符合 numexpr 格式的数学表达式

        Returns:
            str: 数学表达式的计算结果
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"

