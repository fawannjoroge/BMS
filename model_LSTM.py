import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load data, set Date as index and frequency to start of month
df = pd.read_csv('monthly_milk_production.csv', index_col='Date', parse_dates=True)
df.index.freq = 'MS'
print(df.head())

# Plot production over time
df.plot(figsize=(12, 6))
plt.title('Monthly Milk Production')
plt.show()

# Seasonal decomposition to see trend/seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df['Production'])
results.plot()
plt.show()

# Split into train and test
train = df.iloc[:156]
test = df.iloc[156:]

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Use 12 time steps for input
n_input = 12
n_features = 1

# Create a TimeseriesGenerator for training
from keras.preprocessing.sequence import TimeseriesGenerator
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Build LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(train_generator, epochs=5)

# Prepare test data generator
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=n_input, batch_size=1)

# Make predictions on test set
predictions = model.predict(test_generator)

# Inverse transform predictions back to original scale
predicted_values = scaler.inverse_transform(predictions)

# Actual values (align with predictions; skip first n_input points)
actual_values = scaler.inverse_transform(scaled_test[n_input:])

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(test.index[n_input:], actual_values, label='Actual')
plt.plot(test.index[n_input:], predicted_values, label='Predicted')
plt.title('Milk Production: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Production')
plt.legend()
plt.show()