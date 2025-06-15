"""
Recommendation Agent for the NSW Housing Watchdog.

This agent provides personalized housing recommendations to users based on:
1. Their profile and preferences
2. Analyzed housing market data from NSW
3. Detected trends, anomalies, and opportunities
"""

import os
import pathlib
from typing import Dict, Any

from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import CallbackContext

from aus_housing_watchdog.sub_agents.recommendation import prompts
from aus_housing_watchdog.sub_agents.recommendation import tools

# Get model configuration from environment variables or use default
MODEL_NAME = os.environ.get("RECOMMENDATION_AGENT_MODEL", "gemini-2.0-flash")
TEMPERATURE = float(os.environ.get("RECOMMENDATION_AGENT_TEMPERATURE", "0.2"))

# Initialize the agent
root_agent = GenerativeModel(
    MODEL_NAME,
    generation_config={"temperature": TEMPERATURE},
    system_instruction=prompts.return_instructions(),
    tools=[
        tools.create_final_recommendation
    ]
)

def setup_before_agent_call(callback_context: CallbackContext):
    """
    Initialize the recommendation agent's state with the necessary data.
    Gets analysis results from the data_analysis agent's state.
    
    Args:
        callback_context: The callback context containing state information.
    """
    # Initialize empty state containers for recommendations
    callback_context.state["recommendations"] = {}
    callback_context.state["personalized_insights"] = []
    callback_context.state["recommendation_report"] = ""
    
    # Initialize processed data path (in case needed directly)
    data_dir = pathlib.Path.cwd() / "data"
    processed_data_path = data_dir / "processed-housing-data.csv"
    callback_context.state["processed_data_path"] = str(processed_data_path)
    
    # Ensure we can access analysis results from the data_analysis agent
    if "analysis_results" not in callback_context.state:
        callback_context.state["analysis_results"] = {}
    
    # Initialize user profile if not already present
    if "user_profile" not in callback_context.state:
        callback_context.state["user_profile"] = {}
