"""
Cleaning Agent for Aus Housing Watchdog

Responsible for cleaning and normalizing raw housing data from various sources.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleaningAgent:
    """Agent responsible for cleaning and normalizing housing data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Cleaning Agent.
        
        Args:
            config: Configuration dictionary with cleaning rules and settings
        """
        self.agent = Agent(
            name="cleaning_agent",
            model="gemini-2.0-flash",
            description="Cleans and normalizes raw housing data",
            instruction="""
            You are an expert data cleaning agent. Your role is to clean and normalize 
            housing data from various sources, ensuring consistency, handling missing values, 
            and applying standard formatting rules.
            """,
            tools=[
                self.clean_dataframe,
                self.validate_data,
                self.standardize_address,
                self.normalize_price,
                self.upload_to_bigquery
            ]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.bq_client = bigquery.Client() if self.config.get('use_bigquery', False) else None
        
        # Default cleaning rules
        self.default_rules = {
            'drop_duplicates': True,
            'drop_na_subset': ['price', 'suburb', 'state'],
            'price_columns': ['price'],
            'date_columns': ['sale_date', 'listing_date'],
            'categorical_columns': ['property_type', 'suburb', 'state'],
            'boolean_columns': ['has_pool', 'is_auction'],
            'string_columns': ['address', 'description']
        }
        
        # Merge with user-provided rules
        self.rules = {**self.default_rules, **self.config.get('cleaning_rules', {})}
    
    def clean_dataframe(self, df: pd.DataFrame, rules: Optional[Dict] = None) -> pd.DataFrame:
        """Apply cleaning operations to a pandas DataFrame.
        
        Args:
            df: Input DataFrame with raw data
            rules: Optional override for cleaning rules
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
            
        rules = rules or self.rules
        df_clean = df.copy()
        
        # Drop duplicates
        if rules.get('drop_duplicates', True):
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Dropped {initial_rows - len(df_clean)} duplicate rows")
        
        # Drop rows with missing values in required columns
        if rules.get('drop_na_subset'):
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=rules['drop_na_subset'])
            logger.info(f"Dropped {initial_rows - len(df_clean)} rows with missing values")
        
        # Clean price columns
        for col in rules.get('price_columns', []):
            if col in df_clean.columns:
                df_clean[col] = self._clean_price_column(df_clean[col])
        
        # Convert date columns
        for col in rules.get('date_columns', []):
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean categorical columns
        for col in rules.get('categorical_columns', []):
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        # Clean boolean columns
        for col in rules.get('boolean_columns', []):
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(bool)
        
        # Clean string columns
        for col in rules.get('string_columns', []):
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean
    
    def _clean_price_column(self, series: pd.Series) -> pd.Series:
        """Clean a price column by removing non-numeric characters and converting to float."""
        if series.dtype == 'object':
            # Remove dollar signs, commas, and any non-numeric characters except decimal point
            cleaned = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Convert to float, coerce errors to NaN
            return pd.to_numeric(cleaned, errors='coerce')
        return series
    
    def validate_data(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate the cleaned data against a schema.
        
        Args:
            df: DataFrame to validate
            schema: Optional schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        if schema is None:
            schema = self.config.get('validation_schema', {})
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = schema.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check data types
        type_checks = schema.get('type_checks', {})
        for col, expected_type in type_checks.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    results['warnings'].append(
                        f"Column '{col}' has type {actual_type}, expected {expected_type}"
                    )
        
        # Basic statistics
        results['stats'] = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe(include='number').to_dict() if not df.select_dtypes(include='number').empty else {}
        }
        
        return results
    
    def standardize_address(self, address: str) -> Dict[str, str]:
        """Standardize an address string into components.
        
        Args:
            address: Raw address string
            
        Returns:
            Dictionary with standardized address components
        """
        # This is a simplified example - in practice, you'd want to use a more robust
        # geocoding service or address parsing library
        components = {
            'street': '',
            'suburb': '',
            'state': '',
            'postcode': '',
            'full_address': address.strip()
        }
        
        # Basic parsing (customize based on your address format)
        parts = address.split(',')
        if len(parts) >= 3:
            components['street'] = parts[0].strip()
            components['suburb'] = parts[-2].strip()
            state_postcode = parts[-1].strip().split()
            if len(state_postcode) >= 2:
                components['state'] = state_postcode[0]
                components['postcode'] = state_postcode[1]
        
        return components
    
    def normalize_price(self, price: float, from_currency: str = 'AUD', 
                        to_currency: str = 'AUD', date: Optional[str] = None) -> float:
        """Normalize price to a standard currency and time period.
        
        Args:
            price: Original price
            from_currency: Original currency code
            to_currency: Target currency code
            date: Date for currency conversion (YYYY-MM-DD)
            
        Returns:
            Normalized price in target currency
        """
        # In a real implementation, you would fetch exchange rates from an API
        # This is a simplified version that only handles AUD to AUD
        if from_currency == to_currency:
            return price
            
        # Placeholder for actual currency conversion
        logger.warning(f"Currency conversion from {from_currency} to {to_currency} not implemented")
        return price
    
    def upload_to_bigquery(self, df: pd.DataFrame, table_id: str, 
                          write_disposition: str = 'WRITE_TRUNCATE',
                          schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Upload a DataFrame to BigQuery.
        
        Args:
            df: DataFrame to upload
            table_id: Full table ID in format 'project_id.dataset.table'
            write_disposition: BigQuery write disposition
            schema: Optional schema for the table
            
        Returns:
            Dictionary with upload results
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client not initialized. Set use_bigquery=True in config.")
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            schema=schema
        )
        
        try:
            job = self.bq_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()  # Wait for the job to complete
            
            # Get the table to check the number of rows
            table = self.bq_client.get_table(table_id)
            
            return {
                'status': 'success',
                'table_id': table_id,
                'num_rows': table.num_rows,
                'job_id': job.job_id
            }
            
        except Exception as e:
            logger.error(f"Error uploading to BigQuery: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'table_id': table_id
            }
    
    def run(self, data: Any, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the cleaning pipeline.
        
        Args:
            data: Input data (DataFrame or path to file)
            params: Parameters to customize the cleaning process
            
        Returns:
            Dictionary with cleaning results and metadata
        """
        params = params or {}
        results = {
            'status': 'started',
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'cleaning_rules': self.rules,
            'validation': {}
        }
        
        try:
            # Load data if a path is provided
            if isinstance(data, str):
                # Handle different file formats
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(data)
                elif data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                else:
                    raise ValueError(f"Unsupported file format: {data}")
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be a DataFrame or file path")
            
            # Clean the data
            df_clean = self.clean_dataframe(df, params.get('rules'))
            
            # Validate the cleaned data
            validation_results = self.validate_data(df_clean, params.get('validation_schema'))
            results['validation'] = validation_results
            
            # Upload to BigQuery if configured
            if self.bq_client and params.get('upload_to_bigquery', False):
                table_id = params.get('table_id', 'aus_housing.cleaned_listings')
                upload_results = self.upload_to_bigquery(
                    df_clean, 
                    table_id,
                    write_disposition=params.get('write_disposition', 'WRITE_TRUNCATE'),
                    schema=params.get('schema')
                )
                results['bigquery'] = upload_results
            
            results['status'] = 'completed'
            results['cleaned_data_stats'] = {
                'row_count': len(df_clean),
                'column_count': len(df_clean.columns),
                'missing_values': df_clean.isnull().sum().to_dict(),
                'data_types': {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
            }
            
            # Keep a reference to the cleaned data if requested
            if params.get('return_data', False):
                results['data'] = df_clean
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['completed_at'] = pd.Timestamp.utcnow().isoformat()
        return results
