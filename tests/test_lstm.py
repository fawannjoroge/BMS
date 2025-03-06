import unittest
import tensorflow as tf
import numpy as np
from model.lstm import create_lstm_model  # No CustomMeanAbsoluteError in revised model

class TestLSTMModel(unittest.TestCase):
    def test_valid_model_creation(self):
        input_shape = (30, 10)
        model = create_lstm_model(input_shape)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 30, 10))
        self.assertEqual(model.output_shape, (None, 1))


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



    def test_model_training(self):
        input_shape = (10, 4)
        model = create_lstm_model(input_shape)
        x_train = np.random.random((100, 10, 4))
        y_train = np.random.random((100, 1))
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertIn('mae', history.history)  

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
        self.assertEqual(len(dropout_layers), 1)  # Only one Dropout layer after LSTM stack
        self.assertEqual(dropout_layers[0].rate, 0.3)
        # Check LSTM dropout for final layer
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(lstm_layers[-1].dropout, 0.3)  # Dropout in final LSTM

    def test_model_prediction_shape(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape)
        test_input = np.random.random((3, 20, 5))
        predictions = model.predict(test_input, verbose=0)
        self.assertEqual(predictions.shape, (3, 1))

    # Removed test_custom_mae_with_sample_weights (CustomMeanAbsoluteError is gone)

class TestLSTMModelExtended(unittest.TestCase):
    def test_model_with_large_lstm_units(self):
        input_shape = (20, 5)
        lstm_units = [512, 256, 128, 64]
        model = create_lstm_model(input_shape, lstm_units=lstm_units)
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), 4)
        for i, layer in enumerate(lstm_layers):
            self.assertEqual(layer.units, lstm_units[i])

    def test_zero_dropout_rate(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape, dropout_rate=0.0)
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        self.assertTrue(all(layer.rate == 0.0 for layer in dropout_layers))
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(lstm_layers[-1].dropout, 0.0)  # Check LSTM dropout

    def test_custom_learning_rate(self):
        input_shape = (20, 5)
        learning_rate = 0.0001
        model = create_lstm_model(input_shape, learning_rate=learning_rate)
        self.assertEqual(model.optimizer.learning_rate.numpy(), learning_rate)

    def test_model_with_minimal_timesteps(self):
        input_shape = (1, 5)
        model = create_lstm_model(input_shape)
        test_input = np.random.random((1, 1, 5))
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 1))

    def test_model_with_large_feature_dimension(self):
        input_shape = (20, 1000)
        model = create_lstm_model(input_shape)
        test_input = np.random.random((1, 20, 1000))
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 1))

    def test_float_learning_rate(self):
        input_shape = (20, 5)
        learning_rate = 1e-5
        model = create_lstm_model(input_shape, learning_rate=learning_rate)
        self.assertEqual(model.optimizer.learning_rate.numpy(), learning_rate)

    def test_model_weights_initialization(self):
        input_shape = (20, 5)
        model1 = create_lstm_model(input_shape)
        model2 = create_lstm_model(input_shape)
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        self.assertEqual(len(weights1), len(weights2))
        self.assertFalse(all(np.array_equal(w1, w2) for w1, w2 in zip(weights1, weights2)))

    def test_model_with_different_batch_sizes(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape)
        batch_sizes = [1, 16, 32, 64]
        for batch_size in batch_sizes:
            test_input = np.random.random((batch_size, 20, 5))
            prediction = model.predict(test_input, verbose=0)
            self.assertEqual(prediction.shape, (batch_size, 1))

    def test_model_serialization(self):
        input_shape = (20, 5)
        model = create_lstm_model(input_shape)
        config = model.get_config()
        self.assertIsInstance(config, dict)
        self.assertTrue(len(config) > 0)

if __name__ == '__main__':
    unittest.main()