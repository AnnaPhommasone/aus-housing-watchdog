"""
Ingestion Agent for Aus Housing Watchdog

Responsible for fetching raw housing data from various sources including:
- ABS (Australian Bureau of Statistics)
- Domain API
- Other real estate data providers
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from google.cloud import storage
import pandas as pd
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionAgent:
    """Agent responsible for ingesting raw housing data from various sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ingestion Agent.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.agent = Agent(
            name="ingestion_agent",
            model="gemini-2.0-flash",
            description="Fetches raw housing data from various sources",
            instruction="""
            You are an expert data ingestion agent. Your role is to fetch housing data from 
            various sources including ABS and Domain API. Ensure you handle pagination, 
            rate limiting, and error cases appropriately.
            """,
            tools=[self.fetch_abs_data, self.fetch_domain_data, self.upload_to_gcs]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.gcs_client = storage.Client() if self.config.get('use_gcs', False) else None
        
        # Set up API clients
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AusHousingWatchdog/1.0',
            'Accept': 'application/json'
        })
        
        if 'abs_api_key' in self.config:
            self.session.params.update({'key': self.config['abs_api_key']})
        
        if 'domain_api_key' in self.config:
            self.session.headers.update({
                'X-API-Key': self.config['domain_api_key'],
                'Content-Type': 'application/json'
            })
    
    def fetch_abs_data(self, dataset_id: str, params: Optional[Dict] = None) -> Dict:
        """Fetch data from the Australian Bureau of Statistics API.
        
        Args:
            dataset_id: The ABS dataset identifier
            params: Query parameters for the API request
            
        Returns:
            Dictionary containing the API response data
        """
        base_url = self.config.get('abs_base_url', 'https://api.data.abs.gov.au/data')
        url = f"{base_url}/{dataset_id}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ABS data: {e}")
            raise
    
    def fetch_domain_data(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Fetch data from the Domain API.
        
        Args:
            endpoint: The API endpoint to call (e.g., 'listings/residential/_search')
            params: Query parameters for the API request
            
        Returns:
            Dictionary containing the API response data
        """
        base_url = self.config.get('domain_base_url', 'https://api.domain.com.au/v1')
        url = f"{base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Domain data: {e}")
            raise
    
    def upload_to_gcs(self, data: Any, destination_path: str) -> str:
        """Upload data to Google Cloud Storage.
        
        Args:
            data: Data to upload (can be dict, list, or pandas DataFrame)
            destination_path: GCS path in format 'bucket-name/path/to/file.json'
            
        Returns:
            GCS URI of the uploaded file
        """
        if not self.gcs_client:
            raise RuntimeError("GCS client not initialized. Set use_gcs=True in config.")
        
        bucket_name, *path_parts = destination_path.split('/', 1)
        if len(path_parts) == 0:
            raise ValueError("Invalid destination_path. Format: 'bucket-name/path/to/file.json'")
            
        blob_path = path_parts[0]
        
        # Convert data to appropriate format if needed
        if isinstance(data, (dict, list)):
            import json
            content = json.dumps(data).encode('utf-8')
        elif isinstance(data, pd.DataFrame):
            content = data.to_csv(index=False).encode('utf-8')
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Upload to GCS
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(content)
        
        gs_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Uploaded data to {gs_uri}")
        return gs_uri
    
    def run(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the ingestion pipeline.
        
        Args:
            params: Parameters to customize the ingestion process
            
        Returns:
            Dictionary with ingestion results and metadata
        """
        params = params or {}
        results = {
            'status': 'started',
            'timestamp': datetime.utcnow().isoformat(),
            'sources': {}
        }
        
        try:
            # Example: Fetch ABS property price index
            if params.get('fetch_abs', False):
                abs_data = self.fetch_abs_data(
                    dataset_id="RP/1/ABS.ABS_ANNUAL_RESIDENTIAL_PROPERTY_PRICE_INDEX",
                    params=params.get('abs_params', {})
                )
                results['sources']['abs'] = {
                    'status': 'success',
                    'record_count': len(abs_data.get('data', []))
                }
                
                # Optionally upload to GCS
                if self.gcs_client and params.get('upload_to_gcs', False):
                    gcs_path = f"{self.config.get('gcs_bucket', 'aus-housing-watchdog')}/abs/property_prices/{datetime.utcnow().strftime('%Y%m%d')}.json"
                    self.upload_to_gcs(abs_data, gcs_path)
                    results['sources']['abs']['gcs_uri'] = gcs_path
            
            # Example: Fetch Domain listings
            if params.get('fetch_domain', False):
                domain_data = self.fetch_domain_data(
                    endpoint="listings/residential/_search",
                    params=params.get('domain_params', {})
                )
                results['sources']['domain'] = {
                    'status': 'success',
                    'listing_count': len(domain_data.get('listing', []))
                }
                
                # Optionally upload to GCS
                if self.gcs_client and params.get('upload_to_gcs', False):
                    gcs_path = f"{self.config.get('gcs_bucket', 'aus-housing-watchdog')}/domain/listings/{datetime.utcnow().strftime('%Y%m%d')}.json"
                    self.upload_to_gcs(domain_data, gcs_path)
                    results['sources']['domain']['gcs_uri'] = gcs_path
            
            results['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['completed_at'] = datetime.utcnow().isoformat()
        return results
