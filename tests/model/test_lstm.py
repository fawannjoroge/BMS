import unittest
import tensorflow as tf
import numpy as np
from model.lstm import create_lstm_model, CustomMeanAbsoluteError

class TestLSTMModel(unittest.TestCase):
    def test_valid_model_creation(self):
        input_shape = (30, 10)
        model = create_lstm_model(input_shape)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 30, 10))
        self.assertEqual(model.output_shape, (None, 1))

    def test_bidirectional_model(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape, bidirectional=True)
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers))

    def test_batch_normalization(self):
        input_shape = (25, 8)
        model = create_lstm_model(input_shape, use_batchnorm=True)
        self.assertTrue(any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in model.layers))

    def test_custom_lstm_units(self):
        input_shape = (15, 6)
        lstm_units = [128, 64]
        model = create_lstm_model(input_shape, lstm_units=lstm_units)
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), len(lstm_units))
        self.assertEqual(lstm_layers[0].units, lstm_units[0])
        self.assertEqual(lstm_layers[1].units, lstm_units[1])

    def test_invalid_input_shape(self):
        invalid_shapes = [
            (0, 10),
            (-5, 3),
            (20,),
            (1, 2, 3),
            "invalid"
        ]
        for shape in invalid_shapes:
            with self.assertRaises(ValueError):
                create_lstm_model(shape)

    def test_custom_mae_metric(self):
        metric = CustomMeanAbsoluteError()
        y_true = tf.constant([[1.0], [2.0], [3.0]])
        y_pred = tf.constant([[1.1], [2.2], [2.8]])
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertIsInstance(result, tf.Tensor)
        metric.reset_states()
        self.assertEqual(float(metric.result()), 0.0)

    def test_model_training(self):
        input_shape = (10, 4)
        model = create_lstm_model(input_shape)
        x_train = np.random.random((100, 10, 4))
        y_train = np.random.random((100, 1))
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertIn('mean_absolute_error', history.history)

if __name__ == '__main__':
    unittest.main()
class TestLSTMModelAdvanced(unittest.TestCase):
    def test_learning_rate_validation(self):
        input_shape = (20, 5)
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, learning_rate=0)
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, learning_rate=-0.001)

    def test_dropout_rate_validation(self):
        input_shape = (20, 5)
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, dropout_rate=1.5)

    def test_lstm_units_validation(self):
        input_shape = (20, 5)
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, lstm_units=[-64, 32])
        with self.assertRaises(ValueError):
            create_lstm_model(input_shape, lstm_units=[0, 32])

    def test_model_with_single_lstm_layer(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape, lstm_units=[64])
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), 1)
        self.assertEqual(lstm_layers[0].units, 64)

    def test_model_with_three_lstm_layers(self):
        input_shape = (20, 5)
        lstm_units = [128, 64, 32]
        model = create_lstm_model(input_shape, lstm_units=lstm_units)
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), 3)
        for layer, units in zip(lstm_layers, lstm_units):
            self.assertEqual(layer.units, units)

    def test_dropout_layer_presence(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape, lstm_units=[64, 32], dropout_rate=0.3)
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        self.assertGreater(len(dropout_layers), 0)
        for layer in dropout_layers:
            self.assertEqual(layer.rate, 0.3)

    def test_model_prediction_shape(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape)
        test_input = np.random.random((3, 20, 5))
        predictions = model.predict(test_input, verbose=0)
        self.assertEqual(predictions.shape, (3, 1))

    def test_custom_mae_with_sample_weights(self):
        metric = CustomMeanAbsoluteError()
        y_true = tf.constant([[1.0], [2.0], [3.0]])
        y_pred = tf.constant([[1.1], [2.2], [2.8]])
        sample_weights = tf.constant([0.5, 1.0, 0.8])
        metric.update_state(y_true, y_pred, sample_weight=sample_weights)
        result = metric.result()
        self.assertIsInstance(result, tf.Tensor)
        metric.reset_states()
        self.assertEqual(float(metric.result()), 0.0)
