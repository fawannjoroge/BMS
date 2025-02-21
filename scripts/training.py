import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import logging
import gc

import config
from model.lstm import create_lstm_model

def validate_config(cfg):
    """
    Validate the configuration dictionary to ensure all required parameters are present.
    """
    required_params = ['lstm_units', 'dropout_rate', 'batch_size', 'epochs']
    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required config parameter: {param}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Random seeds set.")

def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("GPU memory growth enabled.")
        except Exception as e:
            logger.error(f"Error configuring GPU: {e}")

def load_and_preprocess_data(cfg):
    """
    Load preprocessed numpy data files from the 'data' directory.
    Adjust this function if you need to load and preprocess your CSV data.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    X_train_path = os.path.join(data_dir, "X_train.npy")
    y_train_path = os.path.join(data_dir, "y_train.npy")
    X_val_path = os.path.join(data_dir, "X_val.npy")
    y_val_path = os.path.join(data_dir, "y_val.npy")

    for path in [X_train_path, y_train_path, X_val_path, y_val_path]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(path)

    chunk_size = cfg.get('chunk_size', 1000)
    X_train = np.load(X_train_path, mmap_mode='r')
    y_train = np.load(y_train_path, mmap_mode='r')
    X_val = np.load(X_val_path, mmap_mode='r')
    y_val = np.load(y_val_path, mmap_mode='r')

    logger.info(f"Loaded X_train with shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"Loaded X_val with shape: {X_val.shape}, y_val shape: {y_val.shape}")

    gc.collect()
    tf.keras.backend.clear_session()

    return X_train, y_train, X_val, y_val

def build_model(input_shape, cfg):
    """
    Build and compile the LSTM model using parameters from the configuration.
    """
    model = create_lstm_model(
        input_shape=input_shape,
        lstm_units=cfg['lstm_units'],
        dropout_rate=cfg['dropout_rate'],
        bidirectional=cfg['bidirectional'],
        use_batchnorm=cfg['use_batchnorm'],
        learning_rate=cfg['learning_rate']
    )
    model.summary(print_fn=logger.info)
    return model

def train_model(model, X_train, y_train, X_val, y_val, cfg):
    """
    Train the model using a set of callbacks and parameters from the configuration.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(os.getcwd(), "data", "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        TensorBoard(log_dir=os.path.join(os.getcwd(), "data", "logs"))
    ]

    history = None
    try:
        history = model.fit(
            X_train, y_train,
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    except tf.errors.ResourceExhaustedError:
        logger.error("GPU memory exhausted during training")
    except ValueError as e:
        logger.error(f"Invalid input dimensions: {e}")
    return history

def save_model(model):
    final_model_path = os.path.join(os.getcwd(), "data", "final_model.h5")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

def main():
    validate_config(config.CONFIG)
    set_random_seeds()
    configure_gpu()

    X_train, y_train, X_val, y_val = load_and_preprocess_data(config.CONFIG)

    time_steps = X_train.shape[1]
    num_features = X_train.shape[2]
    input_shape = (time_steps, num_features)
    logger.info(f"Model input shape: {input_shape}")

    model = build_model(input_shape, config.CONFIG)
    history = train_model(model, X_train, y_train, X_val, y_val, config.CONFIG)

    save_model(model)

if __name__ == "__main__":
    main()

