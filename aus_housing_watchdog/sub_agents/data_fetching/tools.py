import time
import os
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import date, timedelta
import logging
import io
import zipfile
import csv
import pandas as pd

# Constants
URL_BASE = 'https://www.valuergeneral.nsw.gov.au/__psi/'
WEEKLY_URL = URL_BASE + 'weekly/'
YEARLY_URL = URL_BASE + 'yearly/'
DOWNLOAD_DIR = 'data/'
YEARS_TO_COLLECT = 6
RECENT_WEEKS_TO_EXCLUDE = 14  # Number of days to exclude from recent weekly downloads.
RETRY_ATTEMPTS = 3
DATA_DIR = "./data"
RAW_FILE_PATH = "extract-1-raw.txt"
CLEAN_FILE_PATH = "extract-2-clean.txt"
FINAL_CSV_PATH = "extract-3-very-clean.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("propsales.log"), logging.StreamHandler()])

# ============================================================================================================================
# Download data

def download_file(url, filepath):
    """Downloads a file from a URL to a specified filepath."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            logging.info(f'Downloading {url} to {filepath} (attempt {attempt + 1})')
            urllib.request.urlretrieve(url, filepath)
            logging.info(f'Downloaded {url} to {filepath}')
            return True
        except (URLError, HTTPError) as e:
            logging.error(f'Error downloading {url} (attempt {attempt + 1}): {e}')
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(5)  # Wait before retrying
            else:
                return False
        except Exception as e:
            logging.error(f'An unexpected error occurred during download {url} : {e}')
            return False
    return False

def download_weekly_data(start_date, end_date):
    """Downloads weekly data files."""
    end_date = end_date - timedelta(days=RECENT_WEEKS_TO_EXCLUDE)
    current_date = start_date
    while current_date < end_date:
        filename = current_date.strftime('%Y%m%d') + '.zip'
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        url = WEEKLY_URL + filename
        download_file(url, filepath)
        current_date += timedelta(days=7)

def download_yearly_data(start_year, end_year):
    """Downloads yearly data files."""
    for year in range(start_year, end_year):
        filename = str(year) + '.zip'
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        url = YEARLY_URL + filename
        download_file(url, filepath)

def download_latest_data():
    """Main function to download data."""
    logging.info('Start downloading the data')
    start_time = time.time()

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created directory: {DOWNLOAD_DIR}")

    today = date.today()
    start_weekly_date = date(today.year, 1, 7) - timedelta(days=date(today.year, 1, 7).weekday())
    end_weekly_date = today

    download_weekly_data(start_weekly_date, end_weekly_date)
    download_yearly_data(today.year - YEARS_TO_COLLECT, today.year)

    logging.info('Complete: the data has been downloaded.')
    logging.info(f'Total elapsed time was {int(time.time() - start_time)} seconds')

# ============================================================================================================================
# Extract and export data into csv file

def extract_data_from_zip(zip_filepath):
    """Extracts .dat files from a zip archive, including nested zips."""
    raw_data_lines = []
    try:
        with zipfile.ZipFile(zip_filepath) as zip_file:
            for file_info in zip_file.namelist():
                if file_info.lower().endswith(".dat"):
                    raw_data_lines.append(zip_file.read(file_info).decode("utf-8") + "\n")
                elif file_info.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(zip_file.read(file_info))) as inner_zip:
                        for inner_file_info in inner_zip.namelist():
                            if inner_file_info.lower().endswith(".dat"):
                                raw_data_lines.append(inner_zip.read(inner_file_info).decode("utf-8") + "\n")
    except FileNotFoundError:
        logging.error(f"File not found: {zip_filepath}")
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_filepath}")
    return raw_data_lines

def merge_data(raw_data_string):
    """Merges and cleans the extracted raw data."""
    merged_lines = []
    for line in raw_data_string.splitlines():
        if line.startswith("B"):
            merged_lines.append("\n" + line)
        elif line.startswith("C"):
            merged_lines.append(line.split(";")[-2])
    return ''.join(merged_lines)

def process_data(clean_file_path):
    """Processes the cleaned data using pandas."""
    date_converter = lambda x: pd.to_datetime(x, format="%Y%m%d", errors='coerce')
    columns_with_dates = ["Contract date", "Settlement date"]
    column_names = ["Record type", "District code", "Property ID", "Sale counter", "Download date / time", "Property name", "Property unit number", "Property house number", "Property street name", "Property locality", "Property post code", "Area", "Area type", "Contract date", "Settlement date", "Purchase price", "Zoning", "Nature of property", "Primary purpose", "Strata lot number", "Component code", "Sale code", "Per cent interest of sale", "Dealing number", "Property legal description"]
    include_columns = ["Property ID", "Sale counter", "Download date / time", "Property name", "Property unit number", "Property house number", "Property street name", "Property locality", "Property post code", "Area", "Area type", "Contract date", "Settlement date", "Purchase price", "Zoning", "Nature of property", "Primary purpose", "Strata lot number", "Dealing number", "Property legal description"]

    df = pd.read_csv(clean_file_path, delimiter=";", header=None, names=column_names, encoding='utf8', usecols=include_columns, parse_dates=columns_with_dates, quoting=csv.QUOTE_NONE)
    for col in columns_with_dates:
      df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors='coerce')

    # Processing the data
    df.loc[df['Area type'] == "H", 'Area'] = df['Area'] * 10000
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['Property post code'] = pd.to_numeric(df['Property post code'], errors='coerce', downcast='float')
    df['Primary purpose'] = df['Primary purpose'].str.capitalize()
    df['Property name'] = df['Property name'].str.title()
    df['Property street name'] = df['Property street name'].str.title()
    df['Property locality'] = df['Property locality'].str.title()

    # Zoning logic removed as it was not working as expected.
    
    return df

def extract_latest_data():
    start_time = time.time()
    logging.info('Start extracting and processing data')

    # Extraction
    raw_data_lines = []
    for file_name in os.listdir(DATA_DIR):
        if file_name.lower().endswith(".zip"):
            zip_filepath = os.path.join(DATA_DIR, file_name)
            raw_data_lines.extend(extract_data_from_zip(zip_filepath))

    raw_data_string = ''.join(raw_data_lines)
    with open(RAW_FILE_PATH, "w") as raw_file:
        raw_file.write(raw_data_string)

    logging.info(f"{int(time.time() - start_time)} seconds elapsed")
    logging.info("Begin merging the data")

    # Merging
    merged_data_string = merge_data(raw_data_string)
    with open(CLEAN_FILE_PATH, "w") as clean_file:
        clean_file.write(merged_data_string)

    logging.info(f"{int(time.time() - start_time)} seconds elapsed")
    logging.info("Begin processing the data")

    # Processing
    df = process_data(CLEAN_FILE_PATH)

    # Exporting
    logging.info(f"{int(time.time() - start_time)} seconds elapsed")
    logging.info("Begin exporting to CSV")

    export_columns = ["Property ID", "Sale counter", "Download date / time", "Property name", "Property unit number", "Property house number", "Property street name", "Property locality", "Property post code", "Area", "Area type", "Contract date", "Settlement date", "Purchase price", "Zoning", "Nature of property", "Primary purpose", "Strata lot number", "Dealing number", "Property legal description"]
    df.to_csv(FINAL_CSV_PATH, columns=export_columns, index=False)

    logging.info("Complete: data has been extracted and processed.")
    logging.info(f"Total elapsed time was {int(time.time() - start_time)} seconds")
