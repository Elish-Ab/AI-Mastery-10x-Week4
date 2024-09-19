import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    logging.info("Loading data from file notebooks dir")
    df = pd.read_csv(file_path, parse_dates = ['Date'])
    df.set_index('Date', inplace=True)
    logging.info(f"Data loaded with shape {df.shape}")
    return df

    