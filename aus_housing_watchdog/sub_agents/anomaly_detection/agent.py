"""
Anomaly Agent for Aus Housing Watchdog

Responsible for detecting unusual patterns and outliers in housing market data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from google.cloud import bigquery
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyAgent:
    """Agent responsible for detecting anomalies in housing market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Anomaly Agent.
        
        Args:
            config: Configuration dictionary with detection settings
        """
        self.agent = Agent(
            name="anomaly_agent",
            model="gemini-2.0-flash",
            description="Detects unusual patterns and outliers in housing data",
            instruction="""
            You are an expert in anomaly detection for real estate data. Your role is to 
            identify unusual patterns, outliers, and potential data quality issues in 
            housing market data using statistical and machine learning methods.
            """,
            tools=[
                self.detect_statistical_outliers,
                self.detect_isolation_forest,
                self.detect_price_jumps,
                self.analyze_seasonality,
                self.query_bigquery
            ]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.bq_client = bigquery.Client() if self.config.get('use_bigquery', False) else None
        
        # Default detection parameters
        self.default_params = {
            'zscore_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'isolation_forest': {
                'contamination': 0.05,
                'random_state': 42
            },
            'price_jump': {
                'window': 30,  # days
                'threshold': 0.2  # 20% change
            },
            'min_data_points': 10
        }
        
        # Merge with user-provided parameters
        self.params = {**self.default_params, **self.config.get('anomaly_params', {})}
    
    def detect_statistical_outliers(self, 
                                 df: pd.DataFrame, 
                                 value_col: str,
                                 method: str = 'zscore',
                                 group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Detect outliers using statistical methods.
        
        Args:
            df: Input DataFrame
            value_col: Column with values to analyze
            method: Detection method ('zscore' or 'iqr')
            group_cols: Columns to group by before detection
            
        Returns:
            DataFrame with added 'is_outlier' and 'outlier_score' columns
        """
        if df.empty or value_col not in df.columns:
            return df
            
        df = df.copy()
        
        if group_cols and all(col in df.columns for col in group_cols):
            # Group data and apply outlier detection within each group
            groups = df.groupby(group_cols, group_keys=False)
            
            if method.lower() == 'zscore':
                # Use z-score method
                def zscore_detector(group):
                    z_scores = np.abs(stats.zscore(group[value_col]))
                    group['outlier_score'] = z_scores
                    group['is_outlier'] = z_scores > self.params['zscore_threshold']
                    return group
                
                df = groups.apply(zscore_detector)
                
            else:  # IQR method
                def iqr_detector(group):
                    Q1 = group[value_col].quantile(0.25)
                    Q3 = group[value_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.params['iqr_multiplier'] * IQR
                    upper_bound = Q3 + self.params['iqr_multiplier'] * IQR
                    
                    group['is_outlier'] = ~group[value_col].between(lower_bound, upper_bound)
                    # Calculate normalized distance from IQR bounds as score
                    group['outlier_score'] = group[value_col].apply(
                        lambda x: max((lower_bound - x) / (Q1 - lower_bound + 1e-10), 
                                    (x - upper_bound) / (upper_bound - Q3 + 1e-10))
                    )
                    return group
                
                df = groups.apply(iqr_detector)
        else:
            # Apply detection to the entire dataset
            if method.lower() == 'zscore':
                z_scores = np.abs(stats.zscore(df[value_col]))
                df['outlier_score'] = z_scores
                df['is_outlier'] = z_scores > self.params['zscore_threshold']
            else:  # IQR method
                Q1 = df[value_col].quantile(0.25)
                Q3 = df[value_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.params['iqr_multiplier'] * IQR
                upper_bound = Q3 + self.params['iqr_multiplier'] * IQR
                
                df['is_outlier'] = ~df[value_col].between(lower_bound, upper_bound)
                df['outlier_score'] = df[value_col].apply(
                    lambda x: max((lower_bound - x) / (Q1 - lower_bound + 1e-10), 
                                (x - upper_bound) / (upper_bound - Q3 + 1e-10))
                )
        
        return df
    
    def detect_isolation_forest(self, 
                              df: pd.DataFrame, 
                              feature_cols: List[str],
                              contamination: Optional[float] = None) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest algorithm.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to use for detection
            contamination: Expected proportion of outliers in the data
            
        Returns:
            DataFrame with added 'anomaly_score' and 'is_anomaly' columns
        """
        if df.empty or not feature_cols or not all(col in df.columns for col in feature_cols):
            return df
            
        df = df.copy()
        
        # Handle missing values
        df_features = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)
        
        # Fit Isolation Forest
        contamination = contamination or self.params['isolation_forest']['contamination']
        clf = IsolationForest(
            contamination=contamination,
            random_state=self.params['isolation_forest']['random_state']
        )
        
        # Predict anomalies
        y_pred = clf.fit_predict(X_scaled)
        
        # Add results to DataFrame
        df['anomaly_score'] = -clf.score_samples(X_scaled)  # Lower score = more anomalous
        df['is_anomaly'] = y_pred == -1
        
        return df
    
    def detect_price_jumps(self, 
                         df: pd.DataFrame, 
                         price_col: str, 
                         date_col: str,
                         window: Optional[int] = None,
                         threshold: Optional[float] = None) -> pd.DataFrame:
        """Detect significant price jumps over time.
        
        Args:
            df: Input DataFrame with price and date columns
            price_col: Name of the price column
            date_col: Name of the date column
            window: Time window in days to compare against
            threshold: Minimum percentage change to consider as a jump
            
        Returns:
            DataFrame with added 'price_jump' and 'jump_magnitude' columns
        """
        if df.empty or price_col not in df.columns or date_col not in df.columns:
            return df
            
        window = window or self.params['price_jump']['window']
        threshold = threshold or self.params['price_jump']['threshold']
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calculate rolling statistics
        df['rolling_mean'] = df[price_col].rolling(window=f'{window}D', on=date_col).mean()
        df['rolling_std'] = df[price_col].rolling(window=f'{window}D', on=date_col).std()
        
        # Detect jumps (price changes exceeding threshold)
        df['price_change'] = df[price_col].pct_change()
        df['price_jump'] = df['price_change'].abs() > threshold
        
        # Calculate magnitude of jumps
        df['jump_magnitude'] = np.where(
            df['price_jump'],
            df['price_change'],
            0
        )
        
        return df
    
    def analyze_seasonality(self, 
                          df: pd.DataFrame, 
                          value_col: str, 
                          date_col: str,
                          freq: str = 'M') -> Dict[str, Any]:
        """Analyze seasonal patterns in the data.
        
        Args:
            df: Input DataFrame
            value_col: Column with values to analyze
            date_col: Column with dates
            freq: Frequency for seasonal decomposition ('D' for daily, 'M' for monthly)
            
        Returns:
            Dictionary with seasonal analysis results
        """
        if df.empty or value_col not in df.columns or date_col not in df.columns:
            return {}
            
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Prepare time series
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            
            # Resample to the specified frequency
            ts = df[value_col].resample(freq).mean()
            
            # Handle missing values
            ts = ts.fillna(method='ffill').fillna(method='bfill')
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts, period=12 if freq == 'M' else 7)
            
            # Calculate seasonality strength
            residual = decomposition.resid.dropna()
            seasonal = decomposition.seasonal.dropna()
            
            if len(residual) > 0 and len(seasonal) > 0:
                strength = max(0, 1 - (residual.var() / (residual + seasonal).var()))
            else:
                strength = 0.0
            
            return {
                'seasonal_strength': float(strength),
                'seasonal_avg': decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean().to_dict(),
                'trend_strength': float(np.abs(decomposition.trend.pct_change().mean())),
                'residual_variance': float(residual.var())
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal analysis: {e}")
            return {
                'error': str(e),
                'seasonal_strength': 0.0
            }
    
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
    
    def run(self, 
            data: Union[pd.DataFrame, str], 
            params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the anomaly detection pipeline.
        
        Args:
            data: Input data (DataFrame, path to file, or SQL query)
            params: Parameters to customize the detection
            
        Returns:
            Dictionary with detection results
        """
        params = params or {}
        results = {
            'status': 'started',
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'detection_params': {**self.params, **params}
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
            required_cols = [params.get('value_col', 'price')]
            if params.get('use_date', True):
                required_cols.append(params.get('date_col', 'date'))
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Apply detection methods
            detection_results = {}
            
            # 1. Statistical outlier detection
            if params.get('use_statistical', True):
                method = params.get('statistical_method', 'zscore')
                group_cols = params.get('group_cols')
                df_outliers = self.detect_statistical_outliers(
                    df, 
                    params.get('value_col', 'price'),
                    method=method,
                    group_cols=group_cols
                )
                
                outlier_stats = {
                    'n_outliers': int(df_outliers['is_outlier'].sum()),
                    'outlier_pct': float(df_outliers['is_outlier'].mean() * 100),
                    'avg_outlier_score': float(df_outliers.loc[df_outliers['is_outlier'], 'outlier_score'].mean())
                }
                
                detection_results['statistical_outliers'] = outlier_stats
                
                # Add top outliers to results
                if params.get('include_examples', True):
                    top_outliers = df_outliers[df_outliers['is_outlier']].nlargest(
                        5, 'outlier_score'
                    )
                    detection_results['top_outliers'] = top_outliers.to_dict('records')
            
            # 2. Isolation Forest (if feature columns are provided)
            feature_cols = params.get('feature_cols')
            if feature_cols and all(col in df.columns for col in feature_cols):
                df_anomalies = self.detect_isolation_forest(
                    df,
                    feature_cols,
                    contamination=params.get('contamination')
                )
                
                anomaly_stats = {
                    'n_anomalies': int(df_anomalies['is_anomaly'].sum()),
                    'anomaly_pct': float(df_anomalies['is_anomaly'].mean() * 100),
                    'avg_anomaly_score': float(df_anomalies['anomaly_score'].mean())
                }
                
                detection_results['isolation_forest'] = anomaly_stats
                
                # Add top anomalies to results
                if params.get('include_examples', True):
                    top_anomalies = df_anomalies[df_anomalies['is_anomaly']].nsmallest(
                        5, 'anomaly_score'  # Lower score = more anomalous
                    )
                    detection_results['top_anomalies'] = top_anomalies.to_dict('records')
            
            # 3. Price jump detection (if date column is available)
            if params.get('use_date', True) and params.get('detect_jumps', True):
                date_col = params.get('date_col', 'date')
                if date_col in df.columns:
                    df_jumps = self.detect_price_jumps(
                        df,
                        price_col=params.get('value_col', 'price'),
                        date_col=date_col,
                        window=params.get('jump_window'),
                        threshold=params.get('jump_threshold')
                    )
                    
                    jump_stats = {
                        'n_jumps': int(df_jumps['price_jump'].sum()),
                        'jump_pct': float(df_jumps['price_jump'].mean() * 100),
                        'avg_jump_magnitude': float(df_jumps.loc[df_jumps['price_jump'], 'jump_magnitude'].abs().mean() * 100)
                    }
                    
                    detection_results['price_jumps'] = jump_stats
                    
                    # Add top jumps to results
                    if params.get('include_examples', True) and 'jump_magnitude' in df_jumps.columns:
                        top_jumps = df_jumps[df_jumps['price_jump']].nlargest(
                            5, 'jump_magnitude'
                        )
                        detection_results['top_jumps'] = top_jumps.to_dict('records')
            
            # 4. Seasonality analysis (if date column is available)
            if params.get('use_date', True) and params.get('analyze_seasonality', True):
                date_col = params.get('date_col', 'date')
                if date_col in df.columns:
                    seasonal_analysis = self.analyze_seasonality(
                        df,
                        value_col=params.get('value_col', 'price'),
                        date_col=date_col,
                        freq=params.get('seasonality_freq', 'M')
                    )
                    detection_results['seasonality'] = seasonal_analysis
            
            results['detection_results'] = detection_results
            results['status'] = 'completed'
            
            # Add summary statistics
            results['summary'] = {
                'total_records': len(df),
                'n_outliers': sum(1 for r in detection_results.values() if 'n_outliers' in r),
                'n_anomalies': sum(1 for r in detection_results.values() if 'n_anomalies' in r),
                'n_jumps': sum(1 for r in detection_results.values() if 'n_jumps' in r)
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['completed_at'] = pd.Timestamp.utcnow().isoformat()
        return results
