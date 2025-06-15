import os
import logging
import pandas as pd
from datetime import date
import time

# Constants
DATA_DIR = "./data"
PROCESSED_CSV_PATH = "processed-housing-data.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("data_processing.log"), logging.StreamHandler()])

# ============================================================================================================================
# Load local data

def find_latest_data_file():
    """Find the latest NSW housing data file in the data directory."""
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return None
    
    # Look for CSV files in the data directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
    
    if not csv_files:
        logging.error(f"No CSV files found in {DATA_DIR}")
        return None
    
    # Sort by date if files have dates in their names, otherwise just take the first one
    latest_file = sorted(csv_files, reverse=True)[0]
    return os.path.join(DATA_DIR, latest_file)

def load_raw_data():
    """Load raw data from the local data file."""
    logging.info('Starting to load local housing data')
    start_time = time.time()
    
    data_file = find_latest_data_file()
    if not data_file:
        logging.error("Could not find a valid data file")
        return False
    
    logging.info(f"Found data file: {data_file}")
    logging.info('Complete: the data file has been located.')
    logging.info(f'Total elapsed time was {int(time.time() - start_time)} seconds')
    
    return True

# ============================================================================================================================
# Process and clean data

def process_clean_data():
    """Process and clean the housing data from the local CSV file."""
    start_time = time.time()
    logging.info('Starting data processing and cleaning')
    
    # Find the latest data file
    data_file = find_latest_data_file()
    if not data_file:
        logging.error("No data file found to process")
        return False
    
    try:
        # Read the CSV file
        logging.info(f"Reading data from {data_file}")
        df = pd.read_csv(data_file)
        
        logging.info(f"Original data shape: {df.shape}")
        
        # Prioritize specific columns
        # Define the priority columns based on the requirements
        priority_columns = [
            # Purchase price
            'Purchase price', 
            # Contract date
            'Contract date',
            # Locational fields
            'Property name', 'Property unit number', 'Property house number', 'Property street name', 
            'Property locality', 'Property post code', 'Property legal description',
            # Property type  
            'Nature of property',
            # Primary purpose
            'Primary purpose',
            # Zoning
            'Zoning'
        ]
        
        # Keep only the priority columns (if they exist in the dataframe)
        existing_columns = [col for col in priority_columns if col in df.columns]
        df = df[existing_columns]
        
        logging.info(f"Processed data shape: {df.shape}")
        
        # Basic cleaning
        # Convert date columns to datetime format if they're not already
        if 'Contract date' in df.columns:
            df['Contract date'] = pd.to_datetime(df['Contract date'], errors='coerce')
        
        # Clean string fields
        for col in ['Property name', 'Property street name', 'Property locality']:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title() if df[col].dtype == 'object' else df[col]
        
        # Ensure purchase price is numeric
        if 'Purchase price' in df.columns:
            df['Purchase price'] = pd.to_numeric(df['Purchase price'], errors='coerce')
            
        # Save processed data to CSV
        output_path = os.path.join(DATA_DIR, PROCESSED_CSV_PATH)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Data processed and saved to {output_path}")
        logging.info(f"Total processing time: {int(time.time() - start_time)} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return False


