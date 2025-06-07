"""
Trend Agent for Aus Housing Watchdog

Responsible for analyzing housing market trends over time.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAgent:
    """Agent responsible for analyzing housing market trends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Trend Agent.
        
        Args:
            config: Configuration dictionary with analysis settings
        """
        self.agent = Agent(
            name="trend_agent",
            model="gemini-2.0-flash",
            description="Analyzes housing market trends over time",
            instruction="""
            You are an expert data analysis agent specializing in real estate trends. 
            Your role is to identify and analyze patterns, compute key metrics, and 
            generate insights about housing market trends.
            """,
            tools=[
                self.calculate_moving_average,
                self.calculate_yoy_change,
                self.identify_trends,
                self.generate_market_summary,
                self.query_bigquery
            ]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.bq_client = bigquery.Client() if self.config.get('use_bigquery', False) else None
        
        # Default analysis parameters
        self.default_params = {
            'window_sizes': [7, 30, 90],  # days
            'metrics': ['mean', 'median', 'count'],
            'group_by': ['suburb', 'property_type'],
            'min_data_points': 5
        }
        
        # Merge with user-provided parameters
        self.params = {**self.default_params, **self.config.get('trend_params', {})}
    
    def calculate_moving_average(self, 
                               df: pd.DataFrame, 
                               value_col: str, 
                               date_col: str = 'date',
                               window: int = 7) -> pd.Series:
        """Calculate a simple moving average for a time series.
        
        Args:
            df: DataFrame containing the time series data
            value_col: Name of the column with values to average
            date_col: Name of the date column
            window: Window size in days for the moving average
            
        Returns:
            Series with the moving average values
        """
        if df.empty:
            return pd.Series()
            
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Calculate moving average
        ma = df.set_index(date_col)[value_col].rolling(window=f'{window}D').mean()
        
        return ma.reset_index(drop=True)
    
    def calculate_yoy_change(self, 
                           df: pd.DataFrame, 
                           value_col: str, 
                           date_col: str = 'date',
                           period: str = 'Y') -> pd.Series:
        """Calculate year-over-year percentage change.
        
        Args:
            df: DataFrame containing the time series data
            value_col: Name of the column with values
            date_col: Name of the date column
            period: Resampling period ('Y' for year, 'Q' for quarter, etc.)
            
        Returns:
            Series with the year-over-year percentage change
        """
        if df.empty:
            return pd.Series()
            
        # Ensure date column is datetime and sort
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Resample and calculate percentage change
        yoy = df.set_index(date_col)[value_col].resample(period).mean().pct_change(periods=1) * 100
        
        return yoy
    
    def identify_trends(self, 
                       df: pd.DataFrame, 
                       value_col: str, 
                       date_col: str = 'date',
                       min_points: int = 5) -> Dict[str, Any]:
        """Identify trends in a time series using simple linear regression.
        
        Args:
            df: DataFrame containing the time series data
            value_col: Name of the column with values
            date_col: Name of the date column
            min_points: Minimum number of data points required
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(df) < min_points:
            return {
                'trend_direction': 'insufficient_data',
                'trend_strength': 0,
                'r_squared': 0,
                'n_points': len(df)
            }
            
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Convert dates to numeric (days since first date)
            start_date = df[date_col].min()
            df['days'] = (df[date_col] - start_date).dt.days
            
            # Fit linear regression
            X = df[['days']].values
            y = df[value_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate R-squared
            y_pred = model.predict(X)
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction and strength
            slope = model.coef_[0]
            
            if slope > 0:
                direction = 'increasing'
            elif slope < 0:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Normalize slope to 0-1 range for strength
            strength = min(abs(slope) / (y.mean() / 100) if y.mean() != 0 else 0, 1.0)
            
            return {
                'trend_direction': direction,
                'trend_strength': float(strength),
                'r_squared': float(r_squared),
                'slope': float(slope),
                'intercept': float(model.intercept_),
                'n_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {
                'trend_direction': 'error',
                'error': str(e),
                'n_points': len(df)
            }
    
    def generate_market_summary(self, 
                             df: pd.DataFrame, 
                             value_col: str = 'price',
                             date_col: str = 'date',
                             group_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a summary of market trends.
        
        Args:
            df: DataFrame with housing data
            value_col: Column with values to analyze (e.g., 'price')
            date_col: Column with dates
            group_cols: Columns to group by (e.g., ['suburb', 'property_type'])
            
        Returns:
            Dictionary with market summary
        """
        if df.empty:
            return {}
            
        group_cols = group_cols or self.params.get('group_by', [])
        results = {}
        
        # Overall summary
        overall = {
            'current_period': {
                'start': df[date_col].min().strftime('%Y-%m-%d'),
                'end': df[date_col].max().strftime('%Y-%m-%d'),
                'n_listings': len(df),
                'avg_value': df[value_col].mean(),
                'median_value': df[value_col].median(),
                'min_value': df[value_col].min(),
                'max_value': df[value_col].max()
            }
        }
        
        # Time-based analysis
        df[date_col] = pd.to_datetime(df[date_col])
        current_year = pd.Timestamp.now().year
        
        # Year-over-year comparison if we have enough data
        if df[date_col].dt.year.nunique() > 1:
            prev_year = current_year - 1
            current_year_data = df[df[date_col].dt.year == current_year][value_col]
            prev_year_data = df[df[date_col].dt.year == prev_year][value_col]
            
            if not prev_year_data.empty and not current_year_data.empty:
                yoy_change = (
                    (current_year_data.median() - prev_year_data.median()) / 
                    prev_year_data.median() * 100
                )
                overall['yoy_change_percent'] = float(yoy_change)
        
        # Add moving averages
        for window in self.params.get('window_sizes', [7, 30, 90]):
            ma_col = f'ma_{window}d'
            df[ma_col] = self.calculate_moving_average(
                df, value_col, date_col, window
            )
        
        # Grouped analysis
        if group_cols and all(col in df.columns for col in group_cols):
            grouped = df.groupby(group_cols)
            
            # Calculate metrics for each group
            group_metrics = grouped[value_col].agg(
                ['count', 'mean', 'median', 'min', 'max']
            ).reset_index()
            
            # Sort by count (descending)
            group_metrics = group_metrics.sort_values('count', ascending=False)
            
            # Convert to list of dicts for JSON serialization
            results['groups'] = group_metrics.to_dict('records')
        
        # Add trend analysis
        trend = self.identify_trends(df, value_col, date_col)
        overall['trend_analysis'] = trend
        
        results['overall'] = overall
        return results
    
    def query_bigquery(self, 
                      query: str, 
                      params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a BigQuery query and return results as a DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client not initialized. Set use_bigquery=True in config.")
        
        try:
            query_job = self.bq_client.query(query, job_config=bigquery.QueryJobConfig(
                query_parameters=params or []
            ))
            return query_job.to_dataframe()
        except Exception as e:
            logger.error(f"Error executing BigQuery query: {e}")
            raise
    
    def run(self, data: Union[pd.DataFrame, str], params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the trend analysis pipeline.
        
        Args:
            data: Input data (DataFrame or path to file/query)
            params: Parameters to customize the analysis
            
        Returns:
            Dictionary with analysis results
        """
        params = params or {}
        results = {
            'status': 'started',
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'analysis_params': {**self.params, **params}
        }
        
        try:
            # Load data if a path or query is provided
            if isinstance(data, str):
                if data.strip().lower().startswith('select'):
                    # It's a SQL query
                    df = self.query_bigquery(data, params.get('query_params'))
                elif data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(data)
                elif data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                else:
                    raise ValueError(f"Unsupported data source: {data}")
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be a DataFrame, file path, or SQL query")
            
            # Ensure required columns exist
            required_cols = [params.get('value_col', 'price'), 
                           params.get('date_col', 'date')]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Generate market summary
            summary = self.generate_market_summary(
                df,
                value_col=params.get('value_col', 'price'),
                date_col=params.get('date_col', 'date'),
                group_cols=params.get('group_by')
            )
            
            results['summary'] = summary
            results['status'] = 'completed'
            
            # Add sample of the processed data if requested
            if params.get('include_sample', False):
                results['sample_data'] = df.head(10).to_dict('records')
            
        except Exception as e:
            logger.error(f"Error in trend analysis pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['completed_at'] = pd.Timestamp.utcnow().isoformat()
        return results
