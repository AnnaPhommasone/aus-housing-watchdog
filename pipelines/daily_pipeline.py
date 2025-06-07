"""
Daily Pipeline for Aus Housing Watchdog

This script orchestrates the daily data collection, processing, analysis, and reporting
for the Australian housing market monitoring system.
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import agents
from agents import CoordinatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def setup_environment(env_file: str = '.env') -> None:
    """Load environment variables from .env file.
    
    Args:
        env_file: Path to the .env file
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning(f"No .env file found at {env_file}. Using system environment variables.")

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the Aus Housing Watchdog pipeline')
    
    parser.add_argument(
        '--pipeline', 
        type=str, 
        default='daily',
        choices=['daily', 'ingestion', 'cleaning', 'analysis', 'reporting'],
        help='Pipeline to run (default: daily)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--env', 
        type=str, 
        default='.env',
        help='Path to .env file (default: .env)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data processing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data processing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--location',
        type=str,
        action='append',
        help='Location to process (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the pipeline in dry-run mode (no external API calls or writes)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def setup_logging(debug: bool = False) -> None:
    """Configure logging.
    
    Args:
        debug: Whether to enable debug logging
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )
    
    # Set log level for external libraries
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    
    logger.info("Logging configured")

def validate_dates(start_date: str, end_date: str) -> tuple:
    """Validate and parse date parameters.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # Default to 30 days ago
            start_dt = datetime.now() - timedelta(days=30)
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            # Default to today
            end_dt = datetime.now()
        
        # Ensure end date is not in the future
        end_dt = min(end_dt, datetime.now())
        
        # Ensure start date is before end date
        if start_dt >= end_dt:
            logger.warning(f"Start date ({start_dt}) is after end date ({end_dt}). Adjusting...")
            start_dt = end_dt - timedelta(days=1)
        
        logger.info(f"Date range: {start_dt.date()} to {end_dt.date()}")
        return start_dt, end_dt
        
    except ValueError as e:
        logger.error(f"Invalid date format. Please use YYYY-MM-DD: {e}")
        raise

def main():
    """Main entry point for the pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup environment and logging
    setup_environment(args.env)
    setup_logging(args.debug)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize coordinator agent
        coordinator = CoordinatorAgent(args.config)
        
        # Prepare pipeline parameters
        params = {
            'dry_run': args.dry_run,
            'locations': args.location or config.get('locations', ['all']),
            'debug': args.debug
        }
        
        # Add date range if specified
        if args.start_date or args.end_date:
            start_dt, end_dt = validate_dates(args.start_date, args.end_date)
            params.update({
                'start_date': start_dt.strftime('%Y-%m-%d'),
                'end_date': end_dt.strftime('%Y-%m-%d')
            })
        
        logger.info(f"Starting {args.pipeline} pipeline with params: {params}")
        
        # Run the specified pipeline
        if args.pipeline == 'daily':
            result = coordinator.run_daily_watchdog()
        elif args.pipeline == 'ingestion':
            result = coordinator.run_ingestion(**params)
        elif args.pipeline == 'cleaning':
            result = coordinator.run_cleaning(**params)
        elif args.pipeline == 'analysis':
            result = coordinator.run_analysis(**params)
        elif args.pipeline == 'reporting':
            result = coordinator.generate_report(**params)
        else:
            raise ValueError(f"Unknown pipeline: {args.pipeline}")
        
        # Log results
        if result.get('status') == 'completed':
            logger.info(f"{args.pipeline.capitalize()} pipeline completed successfully")
        else:
            logger.error(f"{args.pipeline.capitalize()} pipeline failed: {result.get('error')}")
            
        # Print summary
        print("\n=== Pipeline Execution Summary ===")
        print(f"Pipeline: {args.pipeline}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Start Time: {result.get('start_time')}")
        print(f"End Time: {result.get('end_time')}")
        
        if 'duration_seconds' in result:
            print(f"Duration: {result['duration_seconds']:.2f} seconds")
        
        if 'steps_completed' in result and 'steps_failed' in result:
            print(f"Steps Completed: {result['steps_completed']}")
            print(f"Steps Failed: {result['steps_failed']}")
        
        if 'errors' in result and result['errors']:
            print("\nErrors:")
            for i, error in enumerate(result['errors'], 1):
                print(f"  {i}. {error}")
        
        # Exit with appropriate status code
        if result.get('status') == 'completed':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Fatal error in pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
