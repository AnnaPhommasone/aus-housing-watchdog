from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import (
    anomaly_detection,
    cleaning_data,
    data_ingestion,
    geospatial_analysis,
    market_trends,
    report_generation,
)


async def call_data_ingestion_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=data_ingestion.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["data_ingestion_result"] = result
    return result


async def call_cleaning_data_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=cleaning_data.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["cleaning_data_result"] = result
    return result


async def call_market_trends_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=market_trends.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["market_trends_result"] = result
    return result


async def call_anomaly_detection_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=anomaly_detection.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["anomaly_detection_result"] = result
    return result


async def call_geospatial_analysis_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=geospatial_analysis.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["geospatial_analysis_result"] = result
    return result


async def call_report_generation_agent(params: dict, tool_context: ToolContext):
    agent_tool = AgentTool(agent=report_generation.agent)
    result = await agent_tool.run_async(args=params, tool_context=tool_context)
    tool_context.state["report_generation_result"] = result
    return result
