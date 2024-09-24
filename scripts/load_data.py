import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from file in notebooks dir")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
    return df