from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from config import CONFIG

def mae(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))

custom_objects = {
    'mae': mae
}
model_path = CONFIG['final_model_filename']
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'inputs' not in data:
            return jsonify({'error': 'Missing input data'}), 400
            
        input_data = np.array(data['inputs'])
        if input_data.shape[-2:] != (10, 5):
            return jsonify({'error': 'Input shape must be (10, 5)'}), 400
            
        input_data = input_data.reshape(1, 10, 5)
        prediction = model.predict(input_data)
        return jsonify({'predicted_range': prediction.tolist()})
    
    except ValueError as e:
        return jsonify({'error': 'Invalid input format'}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CONFIG.get('port', 5000), debug=CONFIG.get('debug', False))
