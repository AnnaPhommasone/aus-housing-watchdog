# aus_housing_watchdog/sub_agents/__init__.py

# Import sub-agents so they are available when you import sub_agents
# (Optional — but helps static checking and autocompletion)

from . import data_fetch_and_clean as data_fetching

# Optional: define __all__ so it's clear what is public
__all__ = [
    "data_fetching",
]
