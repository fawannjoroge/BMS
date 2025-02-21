import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config):
    required_keys = ['data_dir', 'scaler_filename', 'final_model_filename', 'plot_save_dir']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

def ensure_file_exists(filepath):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(filepath)
    return filepath

def load_test_data(config):
    data_dir = config['data_dir']
    
    X_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    ensure_file_exists(X_test_path)
    ensure_file_exists(y_test_path)
    
    with open(X_test_path, 'rb') as f:
        X_test = np.load(f, allow_pickle=False)
    with open(y_test_path, 'rb') as f:
        y_test = np.load(f, allow_pickle=False)
    
    logger.info(f"Loaded X_test with shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    scaler_path = os.path.join(data_dir, config['scaler_filename'])
    ensure_file_exists(scaler_path)
    scaler_target = joblib.load(scaler_path)
    
    return X_test, y_test, scaler_target

def load_model(config):
    model_path = os.path.join(config['data_dir'], config['final_model_filename'])
    ensure_file_exists(model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.metrics.MeanSquaredError})
    logger.info(f"Loaded model from: {model_path}")
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def evaluate_model(model, X_test, y_test, scaler_target):
    y_pred = model.predict(X_test)
    # Check if the target scaler is fitted; if not, skip inverse transformation.
    if scaler_target is not None and hasattr(scaler_target, "scale_"):
        try:
            y_test_orig = scaler_target.inverse_transform(y_test)
            y_pred_orig = scaler_target.inverse_transform(y_pred)
        except Exception as e:
            logger.error(f"Error during inverse transform: {e}")
            raise e
    else:
        logger.info("Scaler target not fitted. Skipping inverse transform.")
        y_test_orig = y_test
        y_pred_orig = y_pred

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    logger.info(f"Test RMSE: {rmse:.2f}")
    logger.info(f"Test MAE: {mae:.2f}")
    
    return y_test_orig, y_pred_orig, rmse, mae

def plot_results(y_test_orig, y_pred_orig, config, figsize=(10, 6), dpi=300):
    if len(y_test_orig) != len(y_pred_orig):
        raise ValueError("Mismatched lengths between test and prediction arrays")
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
        
    os.makedirs(config['plot_save_dir'], exist_ok=True)
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, label='Predictions')
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Remaining Distance (km)')
    plt.ylabel('Predicted Remaining Distance (km)')
    plt.title('Actual vs Predicted Remaining Distance')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config['plot_save_dir'], "actual_vs_predicted.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.show()

def plot_errors(y_test_orig, y_pred_orig, config):
    errors = y_pred_orig - y_test_orig
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.xlabel('Prediction Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plot_path = os.path.join(config['plot_save_dir'], "prediction_errors.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.show()

def main():
    validate_config(CONFIG)
    X_test, y_test, scaler_target = load_test_data(CONFIG)
    model = load_model(CONFIG)
    y_test_orig, y_pred_orig, rmse, mae = evaluate_model(model, X_test, y_test, scaler_target)
    plot_results(y_test_orig, y_pred_orig, CONFIG)
    plot_errors(y_test_orig, y_pred_orig, CONFIG)

if __name__ == "__main__":
    main()