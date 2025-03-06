import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
from scripts.training import (
    set_random_seeds, configure_gpu, load_data,
    build_model, train_model, save_model, validate_config
)

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Updated test config to include required keys and remove unused ones.
        self.test_config = {
            'data_dir': 'dummy/dir',
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'final_model_filename': 'final_model.h5'
        }

    @patch('scripts.training.random')
    @patch('scripts.training.np.random')
    @patch('scripts.training.tf.random')
    def test_set_random_seeds(self, mock_tf_random, mock_np_random, mock_random):
        set_random_seeds(42)
        mock_random.seed.assert_called_once_with(42)
        mock_np_random.seed.assert_called_once_with(42)
        mock_tf_random.set_seed.assert_called_once_with(42)

    @patch('scripts.training.tf.config.list_physical_devices')
    @patch('scripts.training.tf.config.experimental.set_memory_growth')
    def test_configure_gpu_with_devices(self, mock_set_memory, mock_list_devices):
        mock_device = MagicMock()
        mock_list_devices.return_value = [mock_device]
        configure_gpu()
        mock_set_memory.assert_called_once_with(mock_device, True)

    @patch('scripts.training.os.path.exists')
    @patch('scripts.training.np.load')
    def test_load_data_file_not_found(self, mock_load, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            load_data(self.test_config)

    @patch('scripts.training.create_lstm_model')
    def test_build_model(self, mock_create_model):
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        input_shape = (30, 10)
        model = build_model(input_shape, self.test_config)
        # Assert that create_lstm_model is called with the parameters used in the training code.
        mock_create_model.assert_called_once_with(
            input_shape=input_shape,
            lstm_units=self.test_config['lstm_units'],
            dropout_rate=self.test_config['dropout_rate'],
            learning_rate=self.test_config['learning_rate']
        )
        self.assertEqual(model, mock_model)

    @patch('scripts.training.os.path.join')
    def test_save_model(self, mock_join):
        mock_model = MagicMock()
        mock_join.return_value = 'dummy/dir/final_model.h5'
        save_model(mock_model, self.test_config)
        mock_model.save.assert_called_once_with('dummy/dir/final_model.h5')

    def test_validate_config_missing_params(self):
        invalid_config = {
            'lstm_units': [64, 32],
            'dropout_rate': 0.2
        }
        with self.assertRaises(ValueError) as context:
            validate_config(invalid_config)
        self.assertIn('Missing required config parameter', str(context.exception))

    @patch('scripts.training.TensorBoard')
    @patch('scripts.training.ReduceLROnPlateau')
    @patch('scripts.training.ModelCheckpoint')
    @patch('scripts.training.EarlyStopping')
    def test_train_model(self, mock_early_stopping, mock_checkpoint, mock_reduce_lr, mock_tensorboard):
        # Create mock callback instances.
        mock_es_instance = MagicMock()
        mock_cp_instance = MagicMock()
        mock_rlr_instance = MagicMock()
        mock_tb_instance = MagicMock()

        mock_early_stopping.return_value = mock_es_instance
        mock_checkpoint.return_value = mock_cp_instance
        mock_reduce_lr.return_value = mock_rlr_instance
        mock_tensorboard.return_value = mock_tb_instance

        mock_model = MagicMock()
        X_train = np.random.random((100, 30, 10))
        y_train = np.random.random((100, 1))
        X_val = np.random.random((20, 30, 10))
        y_val = np.random.random((20, 1))

        history = train_model(mock_model, X_train, y_train, X_val, y_val, self.test_config)
        # Extract the callbacks passed to fit.
        _, kwargs = mock_model.fit.call_args
        callbacks_passed = kwargs.get('callbacks', [])
        self.assertEqual(len(callbacks_passed), 4)
        self.assertIn(mock_es_instance, callbacks_passed)
        self.assertIn(mock_cp_instance, callbacks_passed)
        self.assertIn(mock_rlr_instance, callbacks_passed)
        self.assertIn(mock_tb_instance, callbacks_passed)
        mock_model.fit.assert_called_once_with(
            X_train, y_train,
            epochs=self.test_config['epochs'],
            batch_size=self.test_config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks_passed,
            verbose=1
        )

    @patch('scripts.training.ModelCheckpoint')
    @patch('scripts.training.EarlyStopping')
    def test_train_model_resource_exhausted(self, mock_early_stopping, mock_checkpoint):
        mock_model = MagicMock()
        mock_model.fit.side_effect = tf.errors.ResourceExhaustedError(
            None, None, "Resource exhausted"
        )
        X_train = np.random.random((100, 30, 10))
        y_train = np.random.random((100, 1))
        X_val = np.random.random((20, 30, 10))
        y_val = np.random.random((20, 1))

        with self.assertRaises(tf.errors.ResourceExhaustedError):
            train_model(mock_model, X_train, y_train, X_val, y_val, self.test_config)

    def test_validate_config_valid(self):
        try:
            validate_config(self.test_config)
        except ValueError:
            self.fail("validate_config raised ValueError unexpectedly")

if __name__ == '__main__':
    unittest.main()