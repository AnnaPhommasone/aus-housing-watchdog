from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import (
    cleaning_data,
    data_analysis,
    recommendation,
    visualiser
)

async def call_cleaning_data_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=cleaning_data.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["cleaning_data_result"] = result
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
    agent_tool = AgentTool(agent=visualiser.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["visualiser_result"] = result
    return result
