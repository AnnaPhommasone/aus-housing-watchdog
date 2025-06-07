"""
Coordinator Agent for Aus Housing Watchdog

Orchestrates the workflow between different agents and manages the overall pipeline.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from google.adk.agents import Agent

# Import all agent classes
from .ingestion_agent import IngestionAgent
from .cleaning_agent import CleaningAgent
from .trend_agent import TrendAgent
from .anomaly_agent import AnomalyAgent
from .geo_agent import GeoAgent
from .report_agent import ReportAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """Agent responsible for orchestrating the housing market analysis pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Coordinator Agent.
        
        Args:
            config_path: Path to the configuration file
        """
        self.agent = Agent(
            name="coordinator_agent",
            model="gemini-2.0-flash",
            description="Orchestrates the workflow between different agents",
            instruction="""
            You are the central coordinator for the Aus Housing Watchdog system. Your role is to 
            manage the workflow between different specialized agents, handle errors, and ensure 
            the smooth execution of the housing market analysis pipeline.
            """,
            tools=[
                self.run_pipeline,
                self.run_ingestion,
                self.run_cleaning,
                self.run_analysis,
                self.generate_report,
                self.get_status
            ]
        )
        
        # Load configuration
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
        
        # Initialize agents
        self.agents = {
            'ingestion': IngestionAgent(self.config.get('ingestion', {})),
            'cleaning': CleaningAgent(self.config.get('cleaning', {})),
            'trend': TrendAgent(self.config.get('trend', {})),
            'anomaly': AnomalyAgent(self.config.get('anomaly', {})),
            'geo': GeoAgent(self.config.get('geo', {})),
            'report': ReportAgent(self.config.get('report', {}))
        }
        
        # Pipeline state
        self.state = {
            'current_pipeline': None,
            'status': 'idle',
            'start_time': None,
            'end_time': None,
            'results': {},
            'errors': []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary with configuration settings
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def run_pipeline(self, pipeline_name: str = 'daily', params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a complete analysis pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to run ('daily', 'weekly', 'monthly')
            params: Additional parameters for the pipeline
            
        Returns:
            Dictionary with pipeline execution results
        """
        params = params or {}
        self.state.update({
            'current_pipeline': pipeline_name,
            'status': 'running',
            'start_time': datetime.utcnow().isoformat(),
            'results': {},
            'errors': []
        })
        
        pipeline_steps = self.config.get('pipelines', {}).get(pipeline_name, [])
        if not pipeline_steps:
            pipeline_steps = self._get_default_pipeline(pipeline_name)
        
        results = {}
        
        try:
            # Execute each step in the pipeline
            for step in pipeline_steps:
                step_name = step.get('name')
                step_params = {**step.get('params', {}), **params}
                
                logger.info(f"Executing pipeline step: {step_name}")
                
                # Update state
                self.state['current_step'] = step_name
                self.state['step_start_time'] = datetime.utcnow().isoformat()
                
                # Execute the step
                step_result = self._execute_step(step_name, step_params)
                
                # Store results
                results[step_name] = step_result
                
                # Check for errors
                if step_result.get('status') == 'error':
                    self.state['errors'].append({
                        'step': step_name,
                        'error': step_result.get('error'),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    # Handle error based on configuration
                    if step.get('on_error') == 'stop':
                        logger.error(f"Pipeline failed at step '{step_name}': {step_result.get('error')}")
                        self.state['status'] = 'failed'
                        self.state['end_time'] = datetime.utcnow().isoformat()
                        return self._format_pipeline_result(results)
                    elif step.get('on_error') == 'skip':
                        logger.warning(f"Skipping step '{step_name}' due to error")
                        continue
                
                # Update state with step results
                self.state['results'][step_name] = {
                    'status': step_result.get('status'),
                    'execution_time': (datetime.utcnow() - 
                                     datetime.fromisoformat(self.state['step_start_time'])).total_seconds(),
                    'result': step_result.get('result', {})
                }
            
            # Pipeline completed successfully
            self.state.update({
                'status': 'completed',
                'end_time': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Pipeline '{pipeline_name}' completed successfully")
            
        except Exception as e:
            logger.error(f"Unexpected error in pipeline execution: {e}")
            self.state.update({
                'status': 'failed',
                'end_time': datetime.utcnow().isoformat(),
                'errors': self.state['errors'] + [{
                    'step': 'pipeline_execution',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }]
            })
        
        return self._format_pipeline_result(results)
    
    def _execute_step(self, step_name: str, params: Dict) -> Dict[str, Any]:
        """Execute a single pipeline step.
        
        Args:
            step_name: Name of the step to execute
            params: Parameters for the step
            
        Returns:
            Dictionary with step execution results
        """
        try:
            # Parse step name (e.g., 'ingestion.fetch')
            parts = step_name.split('.')
            agent_name = parts[0]
            action = parts[1] if len(parts) > 1 else 'run'
            
            # Get the agent
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            # Execute the action
            if hasattr(agent, action):
                method = getattr(agent, action)
                result = method(**params)
                return {'status': 'completed', 'result': result}
            else:
                raise ValueError(f"Agent '{agent_name}' has no method '{action}'")
                
        except Exception as e:
            logger.error(f"Error executing step '{step_name}': {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_default_pipeline(self, pipeline_name: str) -> List[Dict]:
        """Get the default steps for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            List of step configurations
        """
        if pipeline_name == 'daily':
            return [
                {'name': 'ingestion.fetch', 'params': {'source': 'all'}, 'on_error': 'stop'},
                {'name': 'cleaning.process', 'params': {'source': 'ingestion'}, 'on_error': 'stop'},
                {'name': 'trend.analyze', 'params': {'source': 'cleaning'}, 'on_error': 'continue'},
                {'name': 'anomaly.detect', 'params': {'source': 'cleaning'}, 'on_error': 'continue'},
                {'name': 'geo.aggregate', 'params': {'source': 'cleaning'}, 'on_error': 'continue'},
                {'name': 'report.generate', 'params': {'report_type': 'daily_summary'}, 'on_error': 'continue'}
            ]
        elif pipeline_name == 'weekly':
            return [
                {'name': 'ingestion.fetch', 'params': {'source': 'all', 'full_refresh': True}, 'on_error': 'stop'},
                {'name': 'cleaning.process', 'params': {'source': 'ingestion'}, 'on_error': 'stop'},
                {'name': 'trend.analyze', 'params': {'source': 'cleaning', 'time_window': '7d'}, 'on_error': 'continue'},
                {'name': 'anomaly.detect', 'params': {'source': 'cleaning'}, 'on_error': 'continue'},
                {'name': 'geo.aggregate', 'params': {'source': 'cleaning', 'resolution': 0.05}, 'on_error': 'continue'},
                {'name': 'report.generate', 'params': {'report_type': 'weekly_analysis'}, 'on_error': 'continue'}
            ]
        elif pipeline_name == 'monthly':
            return [
                {'name': 'ingestion.fetch', 'params': {'source': 'all', 'full_refresh': True}, 'on_error': 'stop'},
                {'name': 'cleaning.process', 'params': {'source': 'ingestion'}, 'on_error': 'stop'},
                {'name': 'trend.analyze', 'params': {'source': 'cleaning', 'time_window': '30d'}, 'on_error': 'continue'},
                {'name': 'anomaly.detect', 'params': {'source': 'cleaning', 'sensitivity': 'high'}, 'on_error': 'continue'},
                {'name': 'geo.aggregate', 'params': {'source': 'cleaning', 'resolution': 0.02}, 'on_error': 'continue'},
                {'name': 'report.generate', 'params': {'report_type': 'monthly_report'}, 'on_error': 'continue'}
            ]
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    def _format_pipeline_result(self, results: Dict) -> Dict[str, Any]:
        """Format the pipeline execution results.
        
        Args:
            results: Dictionary with step results
            
        Returns:
            Formatted results dictionary
        """
        return {
            'pipeline': self.state['current_pipeline'],
            'status': self.state['status'],
            'start_time': self.state['start_time'],
            'end_time': self.state['end_time'],
            'duration_seconds': (datetime.fromisoformat(self.state['end_time']) - 
                               datetime.fromisoformat(self.state['start_time'])).total_seconds() \
                               if self.state['end_time'] else None,
            'steps_completed': len([r for r in results.values() if r.get('status') == 'completed']),
            'steps_failed': len([r for r in results.values() if r.get('status') == 'error']),
            'results': {k: v.get('result') for k, v in self.state['results'].items()},
            'errors': self.state['errors']
        }
    
    def run_ingestion(self, source: str = 'all', **kwargs) -> Dict[str, Any]:
        """Run the data ingestion process.
        
        Args:
            source: Data source to fetch from ('all' or specific source name)
            **kwargs: Additional parameters for the ingestion process
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting data ingestion from source: {source}")
            result = self.agents['ingestion'].run(source=source, **kwargs)
            return {'status': 'completed', 'result': result}
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_cleaning(self, source: str = 'ingestion', **kwargs) -> Dict[str, Any]:
        """Run the data cleaning process.
        
        Args:
            source: Source of the data to clean ('ingestion' or path to file)
            **kwargs: Additional parameters for the cleaning process
            
        Returns:
            Dictionary with cleaning results
        """
        try:
            logger.info("Starting data cleaning")
            
            # Get data from the specified source
            if source == 'ingestion' and 'ingestion' in self.state['results']:
                data = self.state['results']['ingestion']['result']
            else:
                data = source
            
            result = self.agents['cleaning'].run(data, **kwargs)
            return {'status': 'completed', 'result': result}
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_analysis(self, analysis_type: str = 'all', **kwargs) -> Dict[str, Any]:
        """Run data analysis.
        
        Args:
            analysis_type: Type of analysis to run ('trend', 'anomaly', 'geo', or 'all')
            **kwargs: Additional parameters for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Starting {analysis_type} analysis")
            
            # Get cleaned data from state or use provided source
            if 'cleaning' in self.state['results']:
                data = self.state['results']['cleaning']['result']
            elif 'source' in kwargs:
                data = kwargs.pop('source')
            else:
                raise ValueError("No cleaned data available and no source provided")
            
            results = {}
            
            # Run the requested analysis
            if analysis_type in ['trend', 'all']:
                trend_result = self.agents['trend'].run(data, **kwargs.get('trend_params', {}))
                results['trend'] = trend_result
            
            if analysis_type in ['anomaly', 'all']:
                anomaly_result = self.agents['anomaly'].run(data, **kwargs.get('anomaly_params', {}))
                results['anomaly'] = anomaly_result
            
            if analysis_type in ['geo', 'all']:
                geo_result = self.agents['geo'].run(data, **kwargs.get('geo_params', {}))
                results['geo'] = geo_result
            
            return {'status': 'completed', 'result': results}
            
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_report(self, report_type: str = 'summary', **kwargs) -> Dict[str, Any]:
        """Generate a report based on analysis results.
        
        Args:
            report_type: Type of report to generate
            **kwargs: Additional parameters for report generation
            
        Returns:
            Dictionary with report generation results
        """
        try:
            logger.info(f"Generating {report_type} report")
            
            # Prepare data for the report
            report_data = {}
            
            # Include analysis results if available
            if 'analysis' in self.state['results']:
                report_data.update(self.state['results']['analysis']['result'])
            
            # Include trend data if available
            if 'trend' in self.state['results']:
                report_data['trend'] = self.state['results']['trend']['result']
            
            # Include anomaly data if available
            if 'anomaly' in self.state['results']:
                report_data['anomaly'] = self.state['results']['anomaly']['result']
            
            # Include geo data if available
            if 'geo' in self.state['results']:
                report_data['geo'] = self.state['results']['geo']['result']
            
            # Generate the report
            result = self.agents['report'].generate_market_report(
                report_data,
                report_type=report_type,
                **kwargs
            )
            
            return {'status': 'completed', 'result': result}
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the coordinator.
        
        Returns:
            Dictionary with status information
        """
        return {
            'status': self.state['status'],
            'current_pipeline': self.state['current_pipeline'],
            'current_step': self.state.get('current_step'),
            'start_time': self.state['start_time'],
            'step_start_time': self.state.get('step_start_time'),
            'end_time': self.state.get('end_time'),
            'step_duration': (datetime.utcnow() - datetime.fromisoformat(self.state['step_start_time'])).total_seconds() 
                            if self.state.get('step_start_time') else None,
            'pipeline_duration': (datetime.utcnow() - datetime.fromisoformat(self.state['start_time'])).total_seconds() 
                               if self.state.get('start_time') else None,
            'steps_completed': len([s for s in self.state['results'].values() if s.get('status') == 'completed']),
            'steps_failed': len([s for s in self.state['results'].values() if s.get('status') == 'error']),
            'errors': self.state['errors']
        }
    
    def run_daily_watchdog(self) -> Dict[str, Any]:
        """Run the daily watchdog pipeline.
        
        This is the main entry point for the daily scheduled job.
        
        Returns:
            Dictionary with pipeline execution results
        """
        logger.info("Starting daily housing market watchdog pipeline")
        return self.run_pipeline('daily')
