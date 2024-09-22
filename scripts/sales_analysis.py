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
    logging.info("Loading data from file in notebooks dir")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
    return df

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

def plot_sales_distribution(df):
    # Distribution of Sales
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=50, kde=True)
    plt.title('Distribution of Sales')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_sales_correlation(df):
    # Correlation matrix
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    
def plot_sales_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=30, kde=True)
    plt.title('Distribution of Sales')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_sales_vs_customers(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Sales', y='Customers', data=df)
    plt.title('Sales vs Customers')
    plt.xlabel('Sales')
    plt.ylabel('Customers')
    plt.show()
    
def plot_christmas_sales(df):
    logging.info("plotting christmas sales")
    christmas_sales = df[df['Date'].str.contains('-12-')]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Sales', data=christmas_sales)
    plt.title('Christmas Season Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.show()
    
def plot_sales_behavior(df):
    logging.info("Analyzing sales behavior before, during, and after holidays")

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Identify holiday dates based on 'StateHoliday' and 'SchoolHoliday' columns
    holidays = df[(df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1)]

    # Group data based on holiday dates
    sales_before_holiday = df[df['Date'] < holidays['Date'].min()]
    sales_during_holiday = df[(df['Date'] >= holidays['Date'].min()) & (df['Date'] <= holidays['Date'].max())]
    sales_after_holiday = df[df['Date'] > holidays['Date'].max()]

    # Create a new column 'Holiday Period' to differentiate before, during, and after holiday periods
    sales_before_holiday['Holiday Period'] = 'Before'
    sales_during_holiday['Holiday Period'] = 'During'
    sales_after_holiday['Holiday Period'] = 'After'

    # Concatenate the dataframes with the assigned holiday periods
    combined_data = pd.concat([sales_before_holiday, sales_during_holiday, sales_after_holiday])

    # Define color palette for each holiday period
    palette = {"Before": "tab:blue", "During": "tab:orange", "After": "tab:green"}

    # Plot the data with different colors for each period
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Sales', hue='Holiday Period', data=combined_data, palette=palette)
    plt.title('Sales Behavior Before, During, and After Holidays')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(title='Holiday Period')
    plt.show()
    
def plot_holiday_effect(df):
    logging.info("Plotting holiday effect...")

    # Identify holiday dates
    holidays = df[(df['StateHoliday'] != 0) | (df['SchoolHoliday'] == 1)]

    # Create a column 'IsHoliday' to mark holiday and non-holiday days
    df['IsHoliday'] = df['Date'].isin(holidays['Date'])

    # Calculate the average sales for holiday vs. non-holiday days
    holiday_effect = df.groupby('IsHoliday')['Sales'].mean()

    # Plot the data
    holiday_effect.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales: Holiday vs Non-Holiday')
    plt.ylabel('Average Sales')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.show()
    
def plot_sales_vs_customers(df):
    logging.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

def plot_promo_effect(df):
    logging.info("Plotting promo effect over time...")
    monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
    monthly_promo_sales.columns = ['No Promo', 'Promo']

    monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()
    
def plot_store_opening_closing_sales_behavior(df):
    logging.info("Analyzing customer behavior during store opening and closing times")

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract hour from the timestamp
    df['Hour'] = df['Date'].dt.hour

    # Group data based on opening and closing times (assumed here as 8 AM to 8 PM)
    opening_hours = range(8, 20)
    df['StoreStatus'] = df['Hour'].apply(lambda x: 'Open' if x in opening_hours else 'Closed')

    # Plot the data
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Hour', hue='StoreStatus', data=df, palette={"Open": "tab:blue", "Closed": "tab:orange"})
    plt.title('Customer Behavior During Store Opening and Closing Times')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Count')
    plt.legend(title='Store Status')
    plt.show()
    
def plot_weekly_sales(df):
    logging.info("plotting weekly sales over time")
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' column as the index
    df.set_index('Date', inplace=True)
    
    # Resample the data to weekly frequency and calculate sum of sales
    weekly_sales = df['Sales'].resample('W').sum()
    
    # Plot the weekly sales
    plt.figure(figsize=(16, 8))
    plt.plot(weekly_sales.index, weekly_sales)
    plt.title('Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()