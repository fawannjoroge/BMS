import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def create_lstm_model(
    input_shape,
    lstm_units=[64, 32],
    dropout_rate=0.2,
    learning_rate=0.001
):
    """
    Create and compile an LSTM model for regression tasks.
    
    Args:
        input_shape (tuple): (time_steps, num_features)
        lstm_units (list): Number of units in each LSTM layer
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
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
        x = LSTM(
            units, 
            return_sequences=return_sequences,
            dropout=dropout_rate if not return_sequences else 0.0
        )(x)
    
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae', 
        metrics=['mae']
    )
    return model

if __name__ == "__main__":
    input_shape = (50, 5)
    model = create_lstm_model(
        input_shape,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    model.summary()
    tf.keras.backend.clear_session()