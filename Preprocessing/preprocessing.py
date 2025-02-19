import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from contextlib import contextmanager
import time
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} completed in {elapsed:.2f} seconds")

import config

def validate_config():
    numeric_configs = ['time_steps', 'train_split', 'val_split', 'outlier_threshold']
    non_numeric_configs = ['data_path']
    
    for key in numeric_configs:
        if key not in config.CONFIG:
            raise ValueError(f"Missing required config: {key}")
        if not isinstance(config.CONFIG[key], (int, float)):
            raise TypeError(f"Invalid type for config {key}")
    for key in non_numeric_configs:
        if key not in config.CONFIG:
            raise ValueError(f"Missing required config: {key}")

validate_config()
logger.info("Configuration validated successfully.")

logger.info("Starting data preprocessing...")

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    if df.empty:
        raise ValueError("Dataset is empty")
    return df

def validate_dataframe(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def process_chunk(chunk):
    return chunk

def process_large_dataset(data_path, chunksize=10000):
    with open(data_path, encoding='cp1252') as f:
        total_rows = sum(1 for _ in f) - 1
    chunks = pd.read_csv(data_path, chunksize=chunksize)
    for chunk in tqdm(chunks, total=total_rows // chunksize, desc="Processing chunks"):
        yield process_chunk(chunk)

import config
data_path = config.CONFIG['data_path']
logger.info(f"Loading data from {data_path}...")
with timer("Loading data"):
    df = load_data(data_path)
logger.info("Data loaded successfully.")

df.columns = [col.strip().replace('Â', '') for col in df.columns]

df.columns = [
    'SOC', 'SOH', 'Voltage', 'Current', 'Battery_Temp',
    'Speed', 'Acceleration', 'Road_Incline', 'External_Temp',
    'Charge_Cycles', 'Charge_Rate', 'Distance_Traveled',
    'Energy_Consumed', 'Remaining_Distance'
]

required_columns = [
    'SOC', 'SOH', 'Voltage', 'Current', 'Battery_Temp',
    'Speed', 'Acceleration', 'Road_Incline', 'External_Temp',
    'Charge_Cycles', 'Charge_Rate', 'Distance_Traveled',
    'Energy_Consumed', 'Remaining_Distance'
]
validate_dataframe(df, required_columns)
logger.info("Dataframe validation complete.")

logger.info("Starting data cleaning...")
with timer("Data cleaning"):
    df.dropna(inplace=True)
    logger.info("Missing values dropped.")

    def remove_outliers(dataframe, columns, threshold):
        df_filtered = dataframe.copy()
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df_filtered[col]):
                logger.info(f"Skipping non-numeric column: {col}")
                continue
            median = df_filtered[col].median()
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            df_filtered = df_filtered[abs(df_filtered[col] - median) <= threshold * IQR]
        return df_filtered

    target_column = 'Remaining_Distance'
    feature_columns = [col for col in df.columns if col != target_column]

    df = remove_outliers(df, feature_columns, config.CONFIG['outlier_threshold'])
    logger.info("Outliers removed.")
logger.info("Data cleaning completed.")

logger.info("Starting feature scaling...")
with timer("Feature scaling"):
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    numeric_feature_columns = [col for col in feature_columns if col != "Charge_Rate"]
    features = scaler_features.fit_transform(df[numeric_feature_columns])

    target = scaler_target.fit_transform(df[[target_column]])
logger.info("Feature scaling completed.")

logger.info("Starting sequence generation...")
with timer("Sequence generation"):
    def create_sequences(features, target, time_steps=config.CONFIG['time_steps']):
        for i in range(len(features) - time_steps):
            yield (features[i:i+time_steps], target[i+time_steps])

    sequences = list(create_sequences(features, target, time_steps=config.CONFIG['time_steps']))
    X = np.array([seq[0] for seq in sequences])
    y = np.array([seq[1] for seq in sequences])
logger.debug(f"Created sequences with shape: {X.shape}")

logger.info("Splitting data into training, validation, and test sets...")
with timer("Data splitting"):
    train_size = int(len(X) * config.CONFIG['train_split'])
    val_size = int(len(X) * config.CONFIG['val_split'])

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
logger.info(f"Data split complete: Training set shape: {X_train.shape}, "
            f"Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

output_dir = os.path.join(os.getcwd(), "data")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
logger.info(f"Preprocessed data saved in folder: {output_dir}")

joblib.dump(scaler_features, os.path.join(output_dir, 'scaler_features.pkl'))
joblib.dump(scaler_target, os.path.join(output_dir, 'scaler_target.pkl'))
logger.info(f"Scalers saved in folder: {output_dir}")
