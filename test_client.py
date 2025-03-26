import requests
import numpy as np

time_steps = 10
feature_dim = 5

random_inputs = np.random.uniform(low=0, high=100, size=(time_steps, feature_dim))

url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json={"inputs": random_inputs.tolist()})

print("Random Inputs:\n", random_inputs)
print("Prediction:", response.json())
