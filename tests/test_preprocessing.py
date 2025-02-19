import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from Preprocessing.preprocessing import (
    load_data,
    validate_dataframe,
    remove_outliers,
    create_sequences,
    validate_config,
    process_large_dataset,
    timer
)
from contextlib import contextmanager  # Needed for the timer test

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            'SOC': [80, 75, 70, 65],
            'SOH': [95, 94, 93, 92],
            'Voltage': [380, 375, 370, 365],
            'Current': [10, 12, 11, 13],
            'Battery_Temp': [25, 26, 27, 28],
            'Speed': [60, 65, 55, 50],
            'Acceleration': [2, 1.5, 1, 0.5],
            'Road_Incline': [0, 1, 2, 1],
            'External_Temp': [20, 22, 21, 23],
            'Charge_Cycles': [100, 101, 102, 103],
            'Charge_Rate': [0.5, 0.6, 0.4, 0.3],
            'Distance_Traveled': [1000, 1050, 1100, 1150],
            'Energy_Consumed': [20, 22, 21, 23],
            'Remaining_Distance': [150, 140, 130, 120]
        })

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')

    def test_load_data_empty_file(self):
        empty_file = Path("empty.csv")
        pd.DataFrame().to_csv(empty_file, index=False)
        with self.assertRaises(ValueError):
            load_data(empty_file)
        empty_file.unlink()

    def test_validate_dataframe_missing_columns(self):
        required_columns = ['SOC', 'NonexistentColumn']
        with self.assertRaises(ValueError):
            validate_dataframe(self.sample_df, required_columns)

    def test_remove_outliers(self):
        columns_to_check = ['Speed', 'Acceleration']
        result_df = remove_outliers(self.sample_df, columns_to_check, threshold=1.5)
        self.assertLessEqual(len(result_df), len(self.sample_df))
        self.assertTrue(all(col in result_df.columns for col in columns_to_check))

    def test_create_sequences(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        target = np.array([[10], [20], [30], [40]])
        time_steps = 2
        
        sequences = list(create_sequences(features, target, time_steps))
        X = np.array([seq[0] for seq in sequences])
        y = np.array([seq[1] for seq in sequences])
        
        self.assertEqual(X.shape, (2, 2, 2))
        self.assertEqual(y.shape, (2, 1))

    def test_timer_context_manager(self):
        # Since the timer context manager yields no value, we expect it to return None.
        with timer("Test operation") as t:
            pass
        self.assertIsNone(t)

    def test_process_large_dataset(self):
        # Create a temporary CSV file
        test_df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10)
        })
        csv_path = Path("test_data.csv")
        test_df.to_csv(csv_path, index=False)
        
        chunk_size = 2
        chunks = list(process_large_dataset(csv_path, chunksize=chunk_size))
        self.assertEqual(len(chunks), (len(test_df) + chunk_size - 1) // chunk_size)
        csv_path.unlink()

    def test_validate_config_missing_config(self):
        import config
        original_config = getattr(config, 'CONFIG', None)
        # Provide all numeric keys but omit the non-numeric key 'data_path'
        config.CONFIG = {
            'time_steps': 50,
            'train_split': 0.8,
            'val_split': 0.1,
            'outlier_threshold': 1.5
            # 'data_path' is missing
        }
        with self.assertRaises(ValueError):
            validate_config()
        if original_config is not None:
            config.CONFIG = original_config

    def test_validate_config_invalid_type(self):
        import config
        original_config = getattr(config, 'CONFIG', None)
        # Provide a dummy data_path so that only 'time_steps' is invalid
        config.CONFIG = {
            'time_steps': 'invalid',  # should be numeric
            'train_split': 0.8,
            'val_split': 0.1,
            'outlier_threshold': 1.5,
            'data_path': 'dummy_path.csv'
        }
        with self.assertRaises(TypeError):
            validate_config()
        if original_config is not None:
            config.CONFIG = original_config

if __name__ == '__main__':
    unittest.main()
