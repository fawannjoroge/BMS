import os

# Base directory for the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    # Data configuration (for preprocessing)
    'data_path': os.path.join(BASE_DIR, 'rawData', 'range_updated.csv'),
    'time_steps': 10,
    'train_split': 0.8,
    'val_split': 0.05,
    'outlier_threshold': 1.0,

    # Preprocessed data and scaler output (shared with evaluate)
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'scaler_filename': 'scaler_target.pkl',

    # Model training configuration (not used by preprocessing or evaluate yet)
    'epochs': 50,
    'batch_size': 32,
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'bidirectional': True,
    'use_batchnorm': True,
    'learning_rate': 0.001,

    # Evaluation configuration
    'final_model_filename': 'final_model.h5',
    'plot_save_dir': os.path.join(BASE_DIR, 'plots')
}