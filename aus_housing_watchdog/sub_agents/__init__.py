# aus_housing_watchdog/sub_agents/__init__.py

# Import sub-agents so they are available when you import sub_agents
# (Optional — but helps static checking and autocompletion)

from . import data_cleaning
from . import data_analysis
from . import recommendation
from . import data_visualiser

# Optional: define __all__ so it's clear what is public
__all__ = [
    "data_cleaning",
    "data_analysis",
    "recommendation",
    "data_visualiser",
]
