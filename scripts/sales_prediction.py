import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    # Check if the columns 'StoreType' and 'Assortment' exist in the DataFrame
    missing_columns = [col for col in ['StoreType', 'Assortment'] if col not in df.columns]
    if missing_columns:
        print(f"Columns {missing_columns} are missing in the DataFrame. Skipping encoding for these columns.")
        
    # Perform data preprocessing steps
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Convert non-numeric columns to numeric using one-hot encoding (excluding missing columns)
    columns_to_encode = [col for col in ['StateHoliday', 'StoreType', 'Assortment'] if col in df.columns]
    df = pd.get_dummies(df, columns=columns_to_encode)
    
    # Separate features and target
    X = df.drop(['Sales'], axis=1)
    y = df['Sales']

    return X, y

def build_model(data):
    logging.info("Store data is now clean")
    
    X, y = preprocess_data(data)
    
    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocess = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

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

def test_data(df):
    scaled_test_data = data_scaler.transform(test_data.values.reshape(-1, 1))
    X_test, y_test = create_supervised_data(scaled_test_data, n_input, n_output)
    X_test = X_test.reshape(X_test.shape[0], n_input, 1)

    # Make predictions on the test data
    predictions = lstm_model.predict(X_test)

    # Inverse transform the predicted values
    predictions = data_scaler.inverse_transform(predictions)

    # Inverse transform the actual values
    y_test_original = data_scaler.inverse_transform(y_test)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")