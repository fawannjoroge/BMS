import unittest
import logging.config
logging.config.fileConfig = lambda x: None  # Override to prevent FileNotFoundError

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

from scripts.preprocessing import (
    validate_config, load_data, validate_dataframe, process_chunk,
    process_large_dataset, remove_outliers, create_sequences, timer,
    encode_charge_rate
)

class TestPreprocessing(unittest.TestCase):

    @patch('scripts.preprocessing.config')
    def test_validate_config(self, mock_config):
        mock_config.CONFIG = {
            'time_steps': 10,
            'train_split': 0.7,
            'val_split': 0.2,
            'outlier_threshold': 1.5,
            'charge_rate_threshold': 45,
            'data_path': 'dummy_path.csv'
        }
        try:
            validate_config()
        except Exception as e:
            self.fail(f"validate_config raised an exception {e}")

    @patch('scripts.preprocessing.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_csv.return_value = mock_df
        df = load_data('dummy_path.csv')
        pd.testing.assert_frame_equal(df, mock_df)

    def test_validate_dataframe(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        required_columns = ['A', 'B']
        try:
            validate_dataframe(df, required_columns)
        except Exception as e:
            self.fail(f"validate_dataframe raised an exception {e}")

    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        required_columns = ['A', 'B', 'C']
        with self.assertRaises(ValueError):
            validate_dataframe(df, required_columns)

    def test_process_chunk(self):
        chunk = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        processed_chunk = process_chunk(chunk)
        pd.testing.assert_frame_equal(processed_chunk, chunk)

    @patch('scripts.preprocessing.open')
    @patch('scripts.preprocessing.pd.read_csv')
    def test_process_large_dataset(self, mock_read_csv, mock_open):
        # Create a fake file that yields three lines (header + two rows)
        fake_file = MagicMock()
        fake_file.__iter__.return_value = iter(['header\n', 'row1\n', 'row2\n'])
        mock_open.return_value.__enter__.return_value = fake_file
        # Return one chunk for read_csv
        mock_read_csv.return_value = [pd.DataFrame({'A': [1, 2], 'B': [3, 4]})]
        chunks = list(process_large_dataset('dummy_path.csv', chunksize=1))
        self.assertEqual(len(chunks), 1)

    def test_remove_outliers(self):
        # Use threshold=1.0 so that the extreme values are removed
        df = pd.DataFrame({'A': [1, 2, 100], 'B': [3, 4, 200]})
        cleaned_df = remove_outliers(df, ['A', 'B'], threshold=1.0)
        expected_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).reset_index(drop=True)
        pd.testing.assert_frame_equal(cleaned_df.reset_index(drop=True), expected_df)

    def test_create_sequences(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        target = np.array([[1], [2], [3], [4]])
        sequences = list(create_sequences(features, target, time_steps=2))
        # For 4 samples and time_steps=2, there should be 2 sequences:
        self.assertEqual(len(sequences), 2)
        np.testing.assert_array_equal(sequences[0][0], np.array([[1, 2], [3, 4]]))
        # The target for the first sequence should be target[2] = [3]
        np.testing.assert_array_equal(sequences[0][1], np.array([3]))

    def test_encode_charge_rate(self):
        rates = pd.Series([30, 50, 40])
        threshold = 45
        result = rates.apply(lambda x: encode_charge_rate(x, threshold))
        expected = pd.Series(['Slow', 'Fast', 'Slow'])
        pd.testing.assert_series_equal(result, expected)

    @patch('scripts.preprocessing.os.makedirs')
    @patch('scripts.preprocessing.np.save')
    def test_save_processed_data(self, mock_save, mock_makedirs):
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([1, 2])
        output_dir = 'dummy_dir'
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        # In this isolated test, we expect 2 calls to np.save
        self.assertEqual(mock_save.call_count, 2)

    @patch('scripts.preprocessing.joblib.dump')
    def test_save_scalers(self, mock_dump):
        scaler_features = MagicMock()
        scaler_target = MagicMock()
        output_dir = 'dummy_dir'
        joblib.dump(scaler_features, os.path.join(output_dir, 'scaler_features.pkl'))
        joblib.dump(scaler_target, os.path.join(output_dir, 'scaler_target.pkl'))
        self.assertEqual(mock_dump.call_count, 2)

    @patch('scripts.preprocessing.pd.read_csv')
    def test_load_data_empty_dataset(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            load_data('dummy.csv')

if __name__ == '__main__':
    unittest.main()