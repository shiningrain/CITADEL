import tensorflow as tf

lstm_input = tf.placeholder(dtype=tf.float64, shape=[3, 1, 4])
lstm = tf.keras.layers.LSTM(512, dtype=tf.float64)
print(lstm.cell.dtype)  # prints None!!!
result = lstm(lstm_input)

