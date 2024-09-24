import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
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
    
def preprocess_data(df):
    logging.info("Cleaning given data")
    # Extract features from datetime columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Calculate number of days to holidays, days after a holiday, etc.
    # Add more feature engineering steps here based on the requirements
    
    # Convert non-numeric columns to numeric using one-hot encoding
    df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment'])
    
    # Scale the data
    scaler = StandardScaler()
    df['CompetitionDistance'] = scaler.fit_transform(df[['CompetitionDistance']])
    logging.info("Store data is now clean")
    return df
    
def build_model(data):
    X = data.drop(['Sales'], axis=1)
    y = data['Sales']
    
    # Define preprocessing steps
    preprocess = ColumnTransformer(transformers=[('num', StandardScaler(), X.columns)])
    
    # Define the model
    model = RandomForestRegressor()
    
    # Create a pipeline
    pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_val, y_val

def custom_loss(y_true, y_pred):
    # Mean Absolute Percentage Error (MAPE)
    error = abs((y_true - y_pred) / y_true)
    return error.mean() * 100

def analyze_predictions(model, X_val, y_val):
    # Get feature importances
    feature_importances = model.named_steps['model'].feature_importances_
    print("Feature Importances:")
    for feature, importance in zip(X_val.columns, feature_importances):
        print(f"{feature}: {importance}")

    # Make predictions on validation set
    predictions = model.predict(X_val)

    # Calculate confidence intervals
    residuals = y_val - predictions
    confidence_interval = 1.96 * residuals.std()  # Assuming a 95% confidence interval
    print(f"Confidence Interval: ±{confidence_interval}")
    
def serialize_model(model, timestamp):
    # Save the model with a timestamp
    model_filename = f"{timestamp}.pkl"
    joblib.dump(model, model_filename)
    
def build_lstm_model(time_series_data, n_input=10, n_output=1):
    # Scale the data in the (-1, 1) range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(time_series_data.values.reshape(-1, 1))

    # Create supervised learning data
    def create_supervised_data(data, n_in, n_out):
        X, y = [], []
        for i in range(len(data)-n_in-n_out+1):
            X.append(data[i:i+n_in])
            y.append(data[i+n_in:i+n_in+n_out])
        return np.array(X), np.array(y)

    X, y = create_supervised_data(scaled_data, n_input, n_output)

    # Reshape input data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], n_input, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(n_output))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    return model, scaler