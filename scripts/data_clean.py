import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df):
    logging.info("Checking for missing values")
    
    # Convert data types
    df['StoreType'] = df['StoreType'].astype('category')
    df['Assortment'] = df['Assortment'].astype('category')
    
    # Convert CompetitionOpenSinceYear and CompetitionOpenSinceMonth to string
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].astype(str).str.split('.').str[0] # Remove decimals
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].astype(str).str.split('.').str[0] # Remove decimals
    
    # Handle missing values
    df['CompetitionOpenSinceYear'].fillna('0', inplace=True)
    df['CompetitionOpenSinceMonth'].fillna('0', inplace=True)
    
    # Feature engineering - create a new feature indicating the duration since competition opened
    df['CompetitionOpenSince'] = pd.to_datetime(df['CompetitionOpenSinceYear'] + '-' + df['CompetitionOpenSinceMonth'] + '-01', errors='coerce')
    
    # Categorical encoding
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment'])
    
    # Splitting the PromoInterval into separate binary columns for each month
    promo_months = df['PromoInterval'].str.split(',', expand=True)

    # List of unique months
    unique_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create binary columns for each month
    for month in unique_months:
        df[f'IsPromo{month}'] = promo_months.apply(lambda x: month in x.values, axis=1).astype(int)

    # Drop the original PromoInterval column if needed
    df.drop('PromoInterval', axis=1, inplace=True)
    # Convert the Date column to datetime format
    logging.info("Store data is now clean")
    return df