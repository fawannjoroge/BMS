import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

class CustomMeanAbsoluteError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error", **kwargs):
        super(CustomMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self._mae = tf.keras.metrics.MeanAbsoluteError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._mae.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self._mae.result()

    def reset_states(self):
        self._mae.reset_state()

CustomMeanAbsoluteError.__name__ = "MeanAbsoluteError"

def create_lstm_model(
    input_shape,
    lstm_units=[64, 32],
    dropout_rate=0.2,
    bidirectional=False,
    use_batchnorm=True,
    learning_rate=0.001
):
    """
    Create and compile an LSTM model with optional BatchNormalization and bidirectional LSTM layers.
    """
    if not (
        isinstance(input_shape, tuple) and 
        len(input_shape) == 2 and 
        all(isinstance(x, int) and x > 0 for x in input_shape)
    ):
        raise ValueError("input_shape must be a tuple of two positive integers: (time_steps, num_features)")
    
    if not (isinstance(learning_rate, (int, float)) and learning_rate > 0):
        raise ValueError("learning_rate must be a positive number")
    
    if not (isinstance(dropout_rate, (int, float)) and 0 <= dropout_rate < 1):
        raise ValueError("dropout_rate must be in the range [0, 1)")
    
    if not (isinstance(lstm_units, list) and all(isinstance(units, int) and units > 0 for units in lstm_units)):
        raise ValueError("lstm_units must be a list of positive integers")
    
    inputs = Input(shape=input_shape)
    x = inputs

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        lstm_layer = LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            x = Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
        
        if use_batchnorm:
            x = BatchNormalization()(x)
        
        if return_sequences:
            x = Dropout(dropout_rate)(x)

    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    mae_metric = CustomMeanAbsoluteError()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  metrics=[mae_metric])
    return model

if __name__ == "__main__":
    dummy_input_shape = (50, 13)
    model = create_lstm_model(
        dummy_input_shape,
        lstm_units=[128, 64, 32],
        dropout_rate=0.3,
        bidirectional=True,
        use_batchnorm=True,
        learning_rate=0.001
    )
    model.summary()
    tf.keras.backend.clear_session()