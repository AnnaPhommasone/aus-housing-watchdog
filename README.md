# Aus Housing Watchdog

A multi-agent system that provides a recommendation for the best NSW suburbs to purchase a home in given the users preferences.

## Setup

1. Clone the repository
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy and configure environment variables (only required once):
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
5. Locally test:
   ```bash
   adk web
   ```
