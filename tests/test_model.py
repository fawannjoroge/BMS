import unittest
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

class TestMilkProductionModel(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Production': [100, 110, 120, 115, 125, 130, 140, 135, 145, 150]
        })
        self.train_data = self.sample_data.iloc[:8]
        self.test_data = self.sample_data.iloc[8:]
        
    def test_data_scaling(self):
        scaler = MinMaxScaler()
        scaler.fit(self.train_data)
        scaled_train = scaler.transform(self.train_data)
        self.assertEqual(scaled_train.shape[1], 1)
        self.assertTrue(np.all(scaled_train >= 0))
        self.assertTrue(np.all(scaled_train <= 1))

    def test_timeseries_generator(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.train_data)
        n_input = 3
        n_features = 1
        generator = TimeseriesGenerator(scaled_data, 
                                      scaled_data,
                                      length=n_input,
                                      batch_size=1)
        X, y = generator[0]
        self.assertEqual(X.shape, (1, n_input, n_features))
        self.assertEqual(y.shape, (1, 1))

    def test_model_architecture(self):
        n_input = 12
        n_features = 1
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        
        self.assertEqual(len(model.layers), 2)
        self.assertIsInstance(model.layers[0], LSTM)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[1].units, 1)

    def test_model_compilation(self):
        n_input = 12
        n_features = 1
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        self.assertEqual(model.loss, 'mse')
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')

if __name__ == '__main__':
    unittest.main()
