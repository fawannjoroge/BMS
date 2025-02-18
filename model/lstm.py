import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

class CustomMeanAbsoluteError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error", **kwargs):
        super(CustomMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self._mae = tf.keras.metrics.MeanAbsoluteError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        return self._mae.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self._mae.result()

    def reset_states(self):
        return self._mae.reset_state()

CustomMeanAbsoluteError.__name__ = "MeanAbsoluteError"

def create_lstm_model(input_shape, lstm_units=[64, 32], dropout_rate=0.2):
    """
    Create and compile an LSTM model.
    """
    if not (
        isinstance(input_shape, tuple) and 
        len(input_shape) == 2 and 
        all(isinstance(x, int) and x > 0 for x in input_shape)
    ):
        raise ValueError("input_shape must be a tuple of two positive integers: (time_steps, num_features)")
    
    inputs = Input(shape=input_shape)
    x = inputs

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = LSTM(units, return_sequences=return_sequences)(x)
        if i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)
    
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    mae_metric = CustomMeanAbsoluteError()
    model.compile(optimizer='adam', loss='mse', metrics=[mae_metric])
    return model

if __name__ == "__main__":
    dummy_input_shape = (50, 4)
    model = create_lstm_model(dummy_input_shape)
    model.summary()