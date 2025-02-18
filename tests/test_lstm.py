import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from model.lstm import create_lstm_model, CustomMeanAbsoluteError

class LSTMModelTest(test.TestCase):

    def setUp(self):
        super(LSTMModelTest, self).setUp()
        self.input_shape = (10, 5)
        
    def test_model_creation_basic(self):
        model = create_lstm_model(self.input_shape)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 10, 5))
        self.assertEqual(model.output_shape, (None, 1))

    def test_invalid_input_shape(self):
        invalid_shapes = [
            (0, 5),
            (-1, 3),
            (10,),
            (1, 2, 3),
            "invalid",
            [10, 5]
        ]
        for shape in invalid_shapes:
            with self.assertRaises(ValueError):
                create_lstm_model(shape)

    def test_custom_lstm_units(self):
        custom_units = [128, 64, 32]
        model = create_lstm_model(self.input_shape, lstm_units=custom_units)
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), len(custom_units))
        for layer, units in zip(lstm_layers, custom_units):
            self.assertEqual(layer.units, units)

    def test_dropout_rate(self):
        custom_dropout = 0.5
        model = create_lstm_model(self.input_shape, dropout_rate=custom_dropout)
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        self.assertTrue(all(layer.rate == custom_dropout for layer in dropout_layers))

    def test_model_training(self):
        batch_size = 32
        epochs = 1
        model = create_lstm_model(self.input_shape)
        x = np.random.random((batch_size,) + self.input_shape)
        y = np.random.random((batch_size, 1))
        history = model.fit(x, y, epochs=epochs, verbose=0)
        self.assertIn('loss', history.history)
        self.assertIn('mean_absolute_error', history.history)

    def test_custom_mae_metric(self):
        metric = CustomMeanAbsoluteError()
        y_true = tf.constant([[1.0], [2.0], [3.0]])
        y_pred = tf.constant([[1.1], [2.2], [2.8]])
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertIsInstance(result, tf.Tensor)
        metric.reset_states()
        self.assertEqual(float(metric.result()), 0.0)

    def test_model_prediction(self):
        model = create_lstm_model(self.input_shape)
        test_input = np.random.random((1,) + self.input_shape)
        prediction = model.predict(test_input)
        self.assertEqual(prediction.shape, (1, 1))
