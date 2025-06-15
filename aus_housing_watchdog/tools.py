from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import (
    data_fetching
)

async def call_data_fetching_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=data_fetching.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["data_fetching_result"] = result
    return result
