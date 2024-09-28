import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
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

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' column as the index
    df.set_index('Date', inplace=True)

    monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
    monthly_promo_sales.columns = ['No Promo', 'Promo']

    monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()
    
def analyze_store_opening_closing(df):
    logging.info("Analyzing customer behavior during store opening and closing times")

    # Ensure that 'Date' is set as the index in the DataFrame
    if 'Date' not in df.index.names:
        logging.error("Date index is missing in the DataFrame")
        return

    # Extract hour from the index
    df.index = pd.to_datetime(df.index)
    df['Hour'] = df.index.hour

    # Group data based on opening and closing times (assumed here as 8 AM to 8 PM)
    opening_hours = range(8, 20)
    df['StoreStatus'] = df['Hour'].apply(lambda x: 'Open' if x in opening_hours else 'Closed')

    # Count the number of customers for each hour and store status
    hourly_data = df.groupby(['Hour', 'StoreStatus']).size().unstack().fillna(0)

    # Plot the data
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_data, markers=True)
    plt.title('Customer Behavior During Store Opening and Closing Times')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Count')
    plt.legend(title='Store Status', labels=['Closed', 'Open'])
    plt.show()

def analyze_promo_effectiveness(df):
    # Analyze promo effectiveness
    promo_effect = df.groupby('Promo')['Sales'].mean()

    # Predictive modeling
    # Implement predictive modeling to estimate promo impact on sales

    # Store clustering
    X = df[['Store', 'Sales', 'Customers']]  # Features for clustering
    kmeans = KMeans(n_clusters=3)  # Assuming 3 clusters
    df['Cluster'] = kmeans.fit_predict(X)

    # A/B testing
    # Conduct A/B tests to measure promo impact on sales in different stores

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    df.plot.scatter(x='Sales', y='Customers', c='Cluster', cmap='viridis', ax=ax[0])
    ax[0].set_title('Store Clustering based on Sales and Customers')
    promo_effect.plot(kind='bar', ax=ax[1])
    ax[1].set_title('Average Sales with and without Promo')
    plt.show()
    
def analyze_open_weekdays_sales(df):
    # Filter stores that are open on all weekdays
    open_all_weekdays = df.groupby('Store')['Open'].sum() == 7
    stores_open_all_weekdays = open_all_weekdays[open_all_weekdays].index.tolist()

    # Filter data for stores open on all weekdays
    stores_data = df[df['Store'].isin(stores_open_all_weekdays)]

    # Analyze sales on weekends for these stores
    weekend_sales = stores_data[stores_data['DayOfWeek'] >= 6].groupby('Store')['Sales'].mean()

    # Plot the sales on weekends for stores open on all weekdays
    plt.figure(figsize=(12, 6))
    weekend_sales.plot(kind='bar', color='skyblue')
    plt.title('Average Sales on Weekends for Stores Open on All Weekdays')
    plt.xlabel('Store')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.show()
    
def analyze_assortment_sales(df):
    # Group data by assortment type and calculate average sales
    assortment_sales = df.groupby('Assortment')['Sales'].mean()

    # Plot the average sales for each assortment type
    plt.figure(figsize=(8, 6))
    assortment_sales.plot(kind='bar', color='skyblue')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.show()
    
def analyze_competitor_distance_sales(df):
    # Plot the relationship between distance to the nearest competitor and sales
    plt.figure(figsize=(10, 6))
    plt.scatter(df['CompetitionDistance'], df['Sales'], color='skyblue', alpha=0.6)
    plt.title('Sales vs. Competition Distance')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()

def analyze_new_competitor_effect(df):
    # Filter stores with NA competitor distance that later have values
    stores_with_na_distance = df[~df['CompetitionDistance'].notnull()]
    stores_with_values_distance = df[df['CompetitionDistance'].notnull()]

    # Plot the effect of new competitor openings on stores
    plt.figure(figsize=(10, 6))
    plt.hist([stores_with_na_distance['Sales'], stores_with_values_distance['Sales']], 
             color=['skyblue', 'salmon'], alpha=0.7, bins=20, stacked=True, label=['NA Distance', 'Values Distance'])
    plt.title('Effect of New Competitor Openings on Stores')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.legend()
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