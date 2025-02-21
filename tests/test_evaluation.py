import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import joblib
import tensorflow as tf
from scripts.evaluation import validate_config, ensure_file_exists, load_test_data, load_model, evaluate_model, plot_results, plot_errors

class TestEvaluation(unittest.TestCase):

    @patch('scripts.evaluation.CONFIG', {
        'data_dir': 'test_data',
        'scaler_filename': 'scaler.pkl',
        'final_model_filename': 'model.h5',
        'plot_save_dir': 'plots'
    })
    def test_validate_config(self):
        config = {
            'data_dir': 'test_data',
            'scaler_filename': 'scaler.pkl',
            'final_model_filename': 'model.h5',
            'plot_save_dir': 'plots'
        }
        try:
            validate_config(config)
        except KeyError:
            self.fail("validate_config raised KeyError unexpectedly!")

    def test_validate_config_missing_keys(self):
        config = {
            'data_dir': 'test_data',
            'scaler_filename': 'scaler.pkl'
        }
        with self.assertRaises(KeyError):
            validate_config(config)

    @patch('os.path.exists', return_value=True)
    def test_ensure_file_exists(self, mock_exists):
        filepath = 'test_data/file.npy'
        self.assertEqual(ensure_file_exists(filepath), filepath)

    @patch('os.path.exists', return_value=False)
    def test_ensure_file_exists_not_found(self, mock_exists):
        filepath = 'test_data/file.npy'
        with self.assertRaises(FileNotFoundError):
            ensure_file_exists(filepath)

    @patch('numpy.load')
    @patch('joblib.load')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('scripts.evaluation.CONFIG', {
        'data_dir': 'test_data',
        'scaler_filename': 'scaler.pkl',
        'final_model_filename': 'model.h5',
        'plot_save_dir': 'plots'
    })
    def test_load_test_data(self, mock_open, mock_exists, mock_joblib_load, mock_np_load):
        mock_np_load.return_value = np.array([[1, 2, 3]])
        mock_joblib_load.return_value = MagicMock()
        X_test, y_test, scaler_target = load_test_data({
            'data_dir': 'test_data',
            'scaler_filename': 'scaler.pkl',
            'final_model_filename': 'model.h5',
            'plot_save_dir': 'plots'
        })
        self.assertTrue(X_test.shape, (1, 3))
        self.assertTrue(y_test.shape, (1, 3))
        self.assertIsNotNone(scaler_target)

    @patch('tensorflow.keras.models.load_model')
    @patch('os.path.exists', return_value=True)
    @patch('scripts.evaluation.CONFIG', {
        'data_dir': 'test_data',
        'scaler_filename': 'scaler.pkl',
        'final_model_filename': 'model.h5',
        'plot_save_dir': 'plots'
    })
    def test_load_model(self, mock_exists, mock_load_model):
        mock_load_model.return_value = MagicMock()
        model = load_model({
            'data_dir': 'test_data',
            'scaler_filename': 'scaler.pkl',
            'final_model_filename': 'model.h5',
            'plot_save_dir': 'plots'
        })
        self.assertIsNotNone(model)

    @patch('tensorflow.keras.models.Model.predict')
    @patch('scripts.evaluation.mean_squared_error')
    @patch('scripts.evaluation.mean_absolute_error')
    def test_evaluate_model(self, mock_mae, mock_mse, mock_predict):
        model = MagicMock()
        X_test = np.array([[1, 2, 3]])
        y_test = np.array([[1, 2, 3]])
        scaler_target = MagicMock()
        scaler_target.inverse_transform.side_effect = lambda x: x
        mock_predict.return_value = np.array([[1, 2, 3]])
        mock_mse.return_value = 0.0
        mock_mae.return_value = 0.0

        y_test_orig, y_pred_orig, rmse, mae = evaluate_model(model, X_test, y_test, scaler_target)
        self.assertEqual(rmse, 0.0)
        self.assertEqual(mae, 0.0)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('os.makedirs')
    def test_plot_results(self, mock_makedirs, mock_show, mock_savefig):
        y_test_orig = np.array([1, 2, 3])
        y_pred_orig = np.array([1, 2, 3])
        config = {
            'plot_save_dir': 'plots'
        }
        try:
            plot_results(y_test_orig, y_pred_orig, config)
        except Exception as e:
            self.fail(f"plot_results raised {e} unexpectedly!")

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('os.makedirs')
    def test_plot_errors(self, mock_makedirs, mock_show, mock_savefig):
        y_test_orig = np.array([1, 2, 3])
        y_pred_orig = np.array([1, 2, 3])
        config = {
            'plot_save_dir': 'plots'
        }
        try:
            plot_errors(y_test_orig, y_pred_orig, config)
        except Exception as e:
            self.fail(f"plot_errors raised {e} unexpectedly!")

if __name__ == '__main__':
    unittest.main()