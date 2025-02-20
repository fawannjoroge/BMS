CONFIG = {
    'data_path': 'rawData/predictive_bms_dataset.csv',
    'time_steps': 50,
    'train_split': 0.7,
    'val_split': 0.15,
    'outlier_threshold': 1.5,
    'epochs': 50,
    'batch_size': 32,
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'bidirectional': True,
    'use_batchnorm': True,
    'learning_rate': 0.001
}