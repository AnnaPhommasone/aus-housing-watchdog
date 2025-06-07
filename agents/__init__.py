"""
Aus Housing Watchdog - Agents Package

This package contains all the agents that make up the Aus Housing Watchdog system.
Each agent is responsible for a specific task in the data processing pipeline.
"""

__all__ = [
    'ingestion_agent',
    'cleaning_agent',
    'trend_agent',
    'anomaly_agent',
    'geo_agent',
    'report_agent',
    'coordinator_agent',
]

# Import all agents to make them available when importing the package
from .ingestion_agent import IngestionAgent
from .cleaning_agent import CleaningAgent
from .trend_agent import TrendAgent
from .anomaly_agent import AnomalyAgent
from .geo_agent import GeoAgent
from .report_agent import ReportAgent
from .coordinator_agent import CoordinatorAgent
