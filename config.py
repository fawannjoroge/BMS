import os

CONFIG = {
    # Data configuration
    'data_path': 'rawData/predictive_bms_dataset.csv',
    'time_steps': 10,
    'train_split': 0.8,
    'val_split': 0.15,
    'outlier_threshold': 1.0,
    "charge_rate_threshold": 0.5,
    
    # Training configuration
    'epochs': 50,
    'batch_size': 32,
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'bidirectional': True,
    'use_batchnorm': True,
    'learning_rate': 0.001,

    
    # Additional paths for preprocessed data, model saving, and plots
    'preprocessed_data_dir': os.path.join(os.getcwd(), "data"),
    'scaler_filename': "scaler_target.pkl",
    'data_dir': os.path.join(os.getcwd(), "data"),
    'final_model_filename': "final_model.h5",
    'plot_save_dir': os.path.join(os.getcwd(), "plots")
}