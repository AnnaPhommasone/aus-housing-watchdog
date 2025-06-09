# aus_housing_watchdog/sub_agents/__init__.py

# Import sub-agents so they are available when you import sub_agents
# (Optional — but helps static checking and autocompletion)

from . import data_ingestion
from . import cleaning_data
from . import market_trends
from . import anomaly_detection
from . import geospatial_analysis
from . import report_generation

# Optional: define __all__ so it's clear what is public
__all__ = [
    "data_ingestion",
    "cleaning_data",
    "market_trends",
    "anomaly_detection",
    "geospatial_analysis",
    "report_generation",
]
