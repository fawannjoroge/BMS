# scripts/evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ['data_dir', 'scaler_filename', 'final_model_filename', 'plot_save_dir']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")
    for key in config:
        if not isinstance(config[key], str):
            raise TypeError(f"Config value for '{key}' must be a string")

def ensure_file_exists(filepath):
    """Check if a file exists and raise an error if it doesn’t."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    return filepath

def load_test_data(config):
    """Load test data and target scaler."""
    data_dir = config['data_dir']
    
    X_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    ensure_file_exists(X_test_path)
    ensure_file_exists(y_test_path)
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    logger.info(f"Loaded X_test with shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    scaler_path = os.path.join(data_dir, config['scaler_filename'])
    ensure_file_exists(scaler_path)
    scaler_target = joblib.load(scaler_path)
    
    return X_test, y_test, scaler_target

def load_model(config):
    """Load the trained LSTM model."""
    model_path = os.path.join(config['data_dir'], config['final_model_filename'])
    ensure_file_exists(model_path)
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from: {model_path}")
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def evaluate_model(model, X_test, y_test, scaler_target):
    """Evaluate the model and compute metrics."""
    y_pred = model.predict(X_test, verbose=0)
    
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    try:
        y_test_orig = scaler_target.inverse_transform(y_test)
        y_pred_orig = scaler_target.inverse_transform(y_pred)
    except ValueError as e:
        logger.error(f"Scaler inverse transform failed: {e}")
        raise ValueError(f"Ensure scaler was fitted correctly: {e}")
    
    y_test_orig = y_test_orig.flatten()
    y_pred_orig = y_pred_orig.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    logger.info(f"Test RMSE: {rmse:.4f} km")
    logger.info(f"Test MAE: {mae:.4f} km")
    
    return y_test_orig, y_pred_orig, rmse, mae

def plot_results(y_test_orig, y_pred_orig, config, figsize=(10, 6), dpi=300):
    """Plot actual vs predicted values."""
    if len(y_test_orig) != len(y_pred_orig):
        raise ValueError(f"Mismatched lengths: y_test_orig ({len(y_test_orig)}), y_pred_orig ({len(y_pred_orig)})")
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
        
    plot_save_dir = config['plot_save_dir']
    os.makedirs(plot_save_dir, exist_ok=True)
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, label='Predictions')
    min_val, max_val = min(y_test_orig.min(), y_pred_orig.min()), max(y_test_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Remaining Distance (km)')
    plt.ylabel('Predicted Remaining Distance (km)')
    plt.title('Actual vs Predicted Remaining Distance')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plot_save_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

def plot_errors(y_test_orig, y_pred_orig, config, figsize=(10, 6), dpi=300):
    """Plot the distribution of prediction errors."""
    errors = y_pred_orig - y_test_orig
    plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(errors, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Prediction Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plot_path = os.path.join(config['plot_save_dir'], "prediction_errors.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

def main():
    """Main function to evaluate the model."""
    try:
        validate_config(config.CONFIG)
        X_test, y_test, scaler_target = load_test_data(config.CONFIG)
        model = load_model(config.CONFIG)
        y_test_orig, y_pred_orig, rmse, mae = evaluate_model(model, X_test, y_test, scaler_target)
        plot_results(y_test_orig, y_pred_orig, config.CONFIG)
        plot_errors(y_test_orig, y_pred_orig, config.CONFIG)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()