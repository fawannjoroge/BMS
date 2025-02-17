import unittest
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import os

class TestMilkProductionModel(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'Production': [100, 110, 120, 115, 125, 130, 140, 135, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200]
        }, index=pd.date_range(start='2020-01-01', periods=20, freq='MS'))
        
    def test_data_loading(self):
        self.assertIsInstance(self.test_data, pd.DataFrame)
        self.assertEqual(self.test_data.index.freq, 'MS')
        
    def test_data_scaling(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.test_data)
        self.assertTrue(np.all(scaled_data >= 0))
        self.assertTrue(np.all(scaled_data <= 1))
        
    def test_timeseries_generator(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.test_data)
        n_input = 12
        generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
        self.assertEqual(len(generator), len(scaled_data) - n_input)
        
    def test_model_structure(self):
        model = Sequential()
        model.add(keras.layers.LSTM(100, activation='relu', input_shape=(12, 1)))
        model.add(keras.layers.Dense(1))
        self.assertEqual(len(model.layers), 2)
        self.assertIsInstance(model.layers[0], keras.layers.LSTM)
        self.assertIsInstance(model.layers[1], keras.layers.Dense)
        
    def test_prediction_shape(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.test_data)
        n_input = 12
        generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
        
        model = Sequential()
        model.add(keras.layers.LSTM(100, activation='relu', input_shape=(n_input, 1)))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        predictions = model.predict(generator)
        self.assertEqual(predictions.shape[1], 1)
        self.assertEqual(len(predictions), len(scaled_data) - n_input)
        
    def test_inverse_transform(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.test_data)
        original_data = scaler.inverse_transform(scaled_data)
        np.testing.assert_array_almost_equal(original_data, self.test_data.values)

if __name__ == '__main__':
    unittest.main()
