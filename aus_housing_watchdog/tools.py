from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import (
    data_cleaning,
    data_analysis,
    recommendation,
    data_visualiser
)

async def call_data_cleaning_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=data_cleaning.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["data_cleaning_result"] = result
    return result

async def call_data_analysis_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=data_analysis.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["data_analysis_result"] = result
    return result

async def call_recommendation_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=recommendation.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["recommendation_result"] = result
    return result


async def call_visualiser_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=data_visualiser.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["visualiser_result"] = result
    return result
