"""
Geo Agent for Aus Housing Watchdog

Responsible for handling geospatial analysis and geographic data processing.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.cloud import bigquery, bigquery_storage
from google.cloud.exceptions import NotFound
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoAgent:
    """Agent responsible for geospatial analysis of housing data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Geo Agent.
        
        Args:
            config: Configuration dictionary with geospatial settings
        """
        self.agent = Agent(
            name="geo_agent",
            model="gemini-2.0-flash",
            description="Handles geospatial analysis and geographic data processing",
            instruction="""
            You are an expert in geospatial analysis for real estate. Your role is to 
            process geographic data, perform spatial joins, calculate distances, and 
            generate geospatial insights for the housing market.
            """,
            tools=[
                self.geocode_address,
                self.calculate_distance,
                self.aggregate_by_geography,
                self.generate_geojson,
                self.query_bigquery_geo
            ]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.bq_client = bigquery.Client() if self.config.get('use_bigquery', False) else None
        if self.bq_client:
            self.bq_storage_client = bigquery_storage.BigQueryReadClient()
        
        # Default geocoding parameters
        self.default_geocoding_params = {
            'region': 'au',
            'bounds': '96.8169,-43.7405,159.1092,-9.1422',  # Rough bounds of Australia
            'components': 'country:AU'
        }
        
        # Merge with user-provided parameters
        self.geocoding_params = {
            **self.default_geocoding_params, 
            **self.config.get('geocoding_params', {})
        }
    
    def geocode_address(self, 
                       address: str, 
                       api_key: Optional[str] = None) -> Dict[str, Any]:
        """Geocode an address using a geocoding service.
        
        Args:
            address: Address string to geocode
            api_key: Optional API key for the geocoding service
            
        Returns:
            Dictionary with geocoding results
        """
        try:
            # In a real implementation, this would call a geocoding API like Google Maps
            # This is a simplified version that returns mock data
            logger.info(f"Geocoding address: {address}")
            
            # Mock response - replace with actual API call
            mock_geocoding = {
                'formatted_address': f"{address}, Australia",
                'geometry': {
                    'location': {
                        'lat': -33.8688 + np.random.uniform(-0.1, 0.1),  # Near Sydney
                        'lng': 151.2093 + np.random.uniform(-0.1, 0.1)
                    },
                    'location_type': 'APPROXIMATE',
                    'viewport': {
                        'northeast': {'lat': -33.5695, 'lng': 151.3426},
                        'southwest': {'lat': -34.1183, 'lng': 150.5209}
                    }
                },
                'place_id': f"mock_place_id_{hash(address)}",
                'types': ['street_address']
            }
            
            return {
                'status': 'OK',
                'results': [mock_geocoding],
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Error geocoding address: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def calculate_distance(self, 
                         point1: Dict[str, float], 
                         point2: Dict[str, float], 
                         units: str = 'km') -> float:
        """Calculate the distance between two geographic points.
        
        Args:
            point1: Dictionary with 'lat' and 'lng' keys
            point2: Dictionary with 'lat' and 'lng' keys
            units: Distance units ('km' or 'mi')
            
        Returns:
            Distance between the points in the specified units
        """
        try:
            from math import radians, sin, cos, sqrt, atan2, asin
            
            # Convert latitude and longitude from degrees to radians
            lat1, lon1 = radians(point1['lat']), radians(point1['lng'])
            lat2, lon2 = radians(point2['lat']), radians(point2['lng'])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Radius of Earth in kilometers
            r = 6371.0 if units.lower() == 'km' else 3956.0  # miles
            
            return c * r
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            raise
    
    def aggregate_by_geography(self, 
                             df: pd.DataFrame, 
                             lat_col: str = 'latitude',
                             lon_col: str = 'longitude',
                             value_col: str = 'price',
                             agg_method: str = 'mean',
                             resolution: float = 0.1) -> pd.DataFrame:
        """Aggregate data by geographic grid cells.
        
        Args:
            df: Input DataFrame with latitude/longitude columns
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column
            value_col: Column to aggregate
            agg_method: Aggregation method ('mean', 'sum', 'count', etc.)
            resolution: Grid cell size in degrees
            
        Returns:
            DataFrame with aggregated values by grid cell
        """
        if df.empty or lat_col not in df.columns or lon_col not in df.columns:
            return pd.DataFrame()
        
        try:
            # Create grid cells
            df_geo = df.copy()
            df_geo['lat_bin'] = (df_geo[lat_col] / resolution).round() * resolution
            df_geo['lon_bin'] = (df_geo[lon_col] / resolution).round() * resolution
            
            # Group by grid cell and aggregate
            aggregation = {value_col: agg_method}
            if 'geometry' in df_geo.columns:
                # Keep the first geometry in each cell
                aggregation['geometry'] = 'first'
            
            df_agg = df_geo.groupby(['lat_bin', 'lon_bin']).agg(aggregation).reset_index()
            
            # Calculate center point of each cell
            df_agg['latitude'] = df_agg['lat_bin'] + (resolution / 2)
            df_agg['longitude'] = df_agg['lon_bin'] + (resolution / 2)
            
            return df_agg
            
        except Exception as e:
            logger.error(f"Error in geographic aggregation: {e}")
            raise
    
    def generate_geojson(self, 
                       df: pd.DataFrame, 
                       lat_col: str = 'latitude',
                       lon_col: str = 'longitude',
                       properties: Optional[List[str]] = None,
                       as_dict: bool = False) -> Union[Dict, str]:
        """Convert a DataFrame with latitude/longitude to GeoJSON format.
        
        Args:
            df: Input DataFrame
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column
            properties: List of columns to include as properties
            as_dict: If True, return as Python dict instead of JSON string
            
        Returns:
            GeoJSON as a dictionary or JSON string
        """
        if df.empty or lat_col not in df.columns or lon_col not in df.columns:
            return {'type': 'FeatureCollection', 'features': []} if as_dict else '{"type":"FeatureCollection","features":[]}'
        
        try:
            features = []
            properties = properties or []
            
            for _, row in df.iterrows():
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    
                    # Skip invalid coordinates
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        continue
                    
                    # Create feature properties
                    props = {}
                    for prop in properties:
                        if prop in row and prop not in [lat_col, lon_col]:
                            # Convert numpy types to Python native types for JSON serialization
                            val = row[prop]
                            if pd.isna(val):
                                props[prop] = None
                            elif hasattr(val, 'item') and callable(val.item):
                                props[prop] = val.item()
                            else:
                                props[prop] = val
                    
                    # Create GeoJSON feature
                    feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [lon, lat]
                        },
                        'properties': props
                    }
                    
                    features.append(feature)
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping row with invalid coordinates: {e}")
                    continue
            
            geojson = {
                'type': 'FeatureCollection',
                'features': features
            }
            
            return geojson if as_dict else json.dumps(geojson, default=str)
            
        except Exception as e:
            logger.error(f"Error generating GeoJSON: {e}")
            return {'type': 'FeatureCollection', 'features': []} if as_dict else '{"type":"FeatureCollection","features":[]}'
    
    def query_bigquery_geo(self, 
                         query: str, 
                         params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a BigQuery SQL query with geospatial functions.
        
        Args:
            query: SQL query string with geospatial functions
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client not initialized. Set use_bigquery=True in config.")
        
        try:
            # Set up query job configuration
            job_config = bigquery.QueryJobConfig(
                query_parameters=params or [],
                use_legacy_sql=False
            )
            
            # Execute query
            query_job = self.bq_client.query(query, job_config=job_config)
            
            # Return results as a DataFrame
            return query_job.to_dataframe(bqstorage_client=self.bq_storage_client)
            
        except Exception as e:
            logger.error(f"Error executing BigQuery geo query: {e}")
            raise
    
    def run(self, 
            data: Union[pd.DataFrame, str], 
            operation: str,
            params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a geospatial operation.
        
        Args:
            data: Input data (DataFrame, address string, or SQL query)
            operation: Operation to perform ('geocode', 'distance', 'aggregate', 'geojson')
            params: Additional parameters for the operation
            
        Returns:
            Dictionary with operation results
        """
        params = params or {}
        results = {
            'status': 'started',
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            if operation == 'geocode':
                # Geocode an address
                if not isinstance(data, str):
                    raise ValueError("For geocoding, data must be an address string")
                
                api_key = params.get('api_key') or self.config.get('geocoding_api_key')
                results.update(self.geocode_address(data, api_key))
            
            elif operation == 'distance':
                # Calculate distance between two points
                if not isinstance(params.get('point1'), dict) or not isinstance(params.get('point2'), dict):
                    raise ValueError("For distance calculation, params must include 'point1' and 'point2' with lat/lng")
                
                distance = self.calculate_distance(
                    params['point1'],
                    params['point2'],
                    units=params.get('units', 'km')
                )
                results['distance'] = distance
                results['units'] = params.get('units', 'km')
            
            elif operation == 'aggregate':
                # Aggregate data by geography
                if isinstance(data, str):
                    if data.strip().lower().startswith('select'):
                        # It's a SQL query
                        df = self.query_bigquery_geo(data, params.get('query_params'))
                    else:
                        # Assume it's a file path
                        if data.endswith('.csv'):
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
                
                # Perform aggregation
                df_agg = self.aggregate_by_geography(
                    df,
                    lat_col=params.get('lat_col', 'latitude'),
                    lon_col=params.get('lon_col', 'longitude'),
                    value_col=params.get('value_col', 'price'),
                    agg_method=params.get('agg_method', 'mean'),
                    resolution=params.get('resolution', 0.1)
                )
                
                # Convert to GeoJSON if requested
                if params.get('as_geojson', False):
                    properties = params.get('properties')
                    if properties is None:
                        # Include all non-geometry columns except lat/lon bins
                        properties = [col for col in df_agg.columns 
                                    if col not in ['lat_bin', 'lon_bin', 'latitude', 'longitude']]
                    
                    results['geojson'] = self.generate_geojson(
                        df_agg,
                        lat_col='latitude',
                        lon_col='longitude',
                        properties=properties,
                        as_dict=True
                    )
                else:
                    results['data'] = df_agg.to_dict('records')
            
            elif operation == 'geojson':
                # Convert data to GeoJSON
                if isinstance(data, str) and data.strip().lower().startswith('select'):
                    # It's a SQL query
                    df = self.query_bigquery_geo(data, params.get('query_params'))
                else:
                    # Assume it's already a DataFrame
                    df = data
                
                properties = params.get('properties')
                if properties is None:
                    # Include all columns except lat/lon
                    lat_col = params.get('lat_col', 'latitude')
                    lon_col = params.get('lon_col', 'longitude')
                    properties = [col for col in df.columns if col not in [lat_col, lon_col]]
                
                geojson = self.generate_geojson(
                    df,
                    lat_col=params.get('lat_col', 'latitude'),
                    lon_col=params.get('lon_col', 'longitude'),
                    properties=properties,
                    as_dict=True
                )
                
                results.update(geojson)
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            results['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error in geo operation '{operation}': {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['completed_at'] = datetime.utcnow().isoformat()
        return results
