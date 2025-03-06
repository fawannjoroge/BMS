import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

from scripts.preprocessing import (
    validate_config, load_data, validate_dataframe, remove_outliers, 
    create_sequences, timer
)

class TestPreprocessing(unittest.TestCase):

    @patch('scripts.preprocessing.config')
    def test_validate_config(self, mock_config):
        mock_config.CONFIG = {
            'data_path': 'dummy_path.csv',
            'time_steps': 10,
            'train_split': 0.7,
            'val_split': 0.15,
            'outlier_threshold': 1.5
        }
        try:
            validate_config()
        except Exception as e:
            self.fail(f"validate_config raised an exception: {e}")

    @patch('scripts.preprocessing.config')
    def test_validate_config_invalid(self, mock_config):
        # Test missing key
        mock_config.CONFIG = {'time_steps': 10, 'train_split': 0.7}
        with self.assertRaises(ValueError):
            validate_config()
        # Test invalid type
        mock_config.CONFIG = {
            'data_path': 'dummy_path.csv',
            'time_steps': '10',
            'train_split': 0.7,
            'val_split': 0.15,
            'outlier_threshold': 1.5
        }
        with self.assertRaises(TypeError):
            validate_config()
        # Test invalid range
        mock_config.CONFIG = {
            'data_path': 'dummy_path.csv',
            'time_steps': 10,
            'train_split': 0.9,
            'val_split': 0.2,
            'outlier_threshold': 1.5
        }
        with self.assertRaises(ValueError):
            validate_config()

    @patch('scripts.preprocessing.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'time': [0, 1],
            'voltage': [12.6, 12.5],
            'current': [0.1, 0.2],
            'temperature': [20, 21],
            'speed': [0, 10],
            'soc': [100, 99],
            'range': [50, 49]
        })
        mock_read_csv.return_value = mock_df
        df = load_data('dummy_path.csv')
        pd.testing.assert_frame_equal(df, mock_df)

    @patch('scripts.preprocessing.pd.read_csv')
    def test_load_data_empty_dataset(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            load_data('dummy.csv')

    @patch('scripts.preprocessing.pd.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_data('dummy.csv')

    def test_validate_dataframe(self):
        df = pd.DataFrame({
            'time': [0, 1],
            'voltage': [12.6, 12.5],
            'current': [0.1, 0.2],
            'temperature': [20, 21],
            'speed': [0, 10],
            'soc': [100, 99],
            'range': [50, 49]
        })
        required_columns = ['time', 'voltage', 'current', 'temperature', 'speed', 'soc', 'range']
        try:
            validate_dataframe(df, required_columns)
        except Exception as e:
            self.fail(f"validate_dataframe raised an exception: {e}")

    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({
            'time': [0, 1],
            'voltage': [12.6, 12.5]
        })
        required_columns = ['time', 'voltage', 'current', 'temperature', 'speed', 'soc', 'range']
        with self.assertRaises(ValueError):
            validate_dataframe(df, required_columns)

    def test_remove_outliers(self):
        df = pd.DataFrame({
            'Voltage': [12.5, 12.6, 12.7, 12.8, 20.0],  # 20.0 is a clear outlier
            'Current': [0.1, 0.2, 0.3, 0.4, 10.0]      # 10.0 is a clear outlier
        })
        cleaned_df = remove_outliers(df, ['Voltage', 'Current'], threshold=1.5)
        expected_df = pd.DataFrame({
            'Voltage': [12.5, 12.6, 12.7, 12.8],
            'Current': [0.1, 0.2, 0.3, 0.4]
        }).reset_index(drop=True)
        pd.testing.assert_frame_equal(cleaned_df.reset_index(drop=True), expected_df)

    def test_create_sequences(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        X, y = create_sequences(data, time_steps=2)
        expected_X = np.array([
            [[1, 2], [4, 5]],
            [[4, 5], [7, 8]]
        ])
        expected_y = np.array([9, 12])
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

    @patch('scripts.preprocessing.os.makedirs')
    @patch('scripts.preprocessing.np.save')
    @patch('scripts.preprocessing.joblib.dump')
    def test_preprocess_data(self, mock_dump, mock_save, mock_makedirs):
        with patch('scripts.preprocessing.config') as mock_config:
            mock_config.CONFIG = {
                'data_path': 'dummy_path.csv',
                'time_steps': 2,
                'train_split': 0.5,
                'val_split': 0.25,
                'outlier_threshold': 1.5
            }
            with patch('scripts.preprocessing.pd.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({
                    'time': [0, 1, 2, 3],
                    'voltage': [12.6, 12.5, 12.4, 12.3],
                    'current': [0.1, 0.2, 0.3, 0.4],
                    'temperature': [20, 21, 22, 23],
                    'speed': [0, 10, 20, 30],
                    'soc': [100, 99, 98, 97],
                    'range': [50, 49, 48, 47]
                })
                mock_read_csv.return_value = mock_df
                from scripts.preprocessing import preprocess_data
                preprocess_data()
                self.assertEqual(mock_save.call_count, 6)  # 6 saves for X/y train/val/test
                self.assertEqual(mock_dump.call_count, 2)  # 2 scaler dumps
if __name__ == '__main__':
    unittest.main()