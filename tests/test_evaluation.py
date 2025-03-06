import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scripts.evaluation import validate_config, load_test_data, load_model, evaluate_model

class TestEvaluation(unittest.TestCase):

    @patch('scripts.evaluation.config')
    def test_validate_config(self, mock_config):
        mock_config.CONFIG = {
            'data_dir': 'data',
            'scaler_filename': 'scaler_target.pkl',
            'final_model_filename': 'final_model.h5',
            'plot_save_dir': 'plots'
        }
        try:
            validate_config(mock_config.CONFIG)
        except Exception as e:
            self.fail(f"validate_config raised an exception: {e}")

    @patch('scripts.evaluation.config')
    @patch('scripts.evaluation.np.load')
    @patch('scripts.evaluation.joblib.load')
    @patch('scripts.evaluation.os.path.exists')
    def test_load_test_data(self, mock_exists, mock_joblib_load, mock_np_load, mock_config):
        mock_config.CONFIG = {
            'data_dir': 'data',
            'scaler_filename': 'scaler_target.pkl'
        }
        mock_exists.return_value = True
        mock_np_load.side_effect = [np.array([[1, 2]]), np.array([3])]
        mock_scaler = MinMaxScaler()
        mock_joblib_load.return_value = mock_scaler
        X_test, y_test, scaler = load_test_data(mock_config.CONFIG)
        self.assertEqual(X_test.shape, (1, 2))
        self.assertEqual(y_test.shape, (1,))
        self.assertEqual(scaler, mock_scaler)

    @patch('scripts.evaluation.config') 
    @patch('scripts.evaluation.tf.keras.models.load_model')
    @patch('scripts.evaluation.os.path.exists')
    def test_load_model(self, mock_exists, mock_load_model, mock_config):
        mock_config.CONFIG = {
            'data_dir': 'data',
            'final_model_filename': 'final_model.h5'
        }
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        model = load_model(mock_config.CONFIG)
        self.assertEqual(model, mock_model)


    @patch('scripts.evaluation.config')
    def test_evaluate_model(self, mock_config):
        mock_config.CONFIG = {
            'data_dir': 'data',
            'scaler_filename': 'scaler_target.pkl',
            'final_model_filename': 'final_model.h5',
            'plot_save_dir': 'plots'
        }
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        X_test = np.array([[1, 2]])
        y_test = np.array([0.6])
        scaler = MinMaxScaler()
        scaler.fit([[0], [1]])
        y_test_orig, y_pred_orig, rmse, mae = evaluate_model(mock_model, X_test, y_test, scaler)
        self.assertEqual(y_test_orig.shape, (1,))
        self.assertEqual(y_pred_orig.shape, (1,))

if __name__ == '__main__':
    unittest.main()