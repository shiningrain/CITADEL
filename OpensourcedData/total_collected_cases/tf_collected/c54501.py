import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

N = 128
fft_length = 1024

input = tfk.Input((N, fft_length // 2 + 1), dtype=tf.dtypes.complex64)
output = tfkl.Lambda(lambda x: tf.signal.irfft(x), output_shape=(N, fft_length))(input)

model = tfk.Model(inputs=input, outputs=output)

tf.get_logger().warning("Begin Tensorflow test...")
test_input = tf.complex(
    tf.random.normal((1, N, fft_length // 2 + 1), dtype=tf.dtypes.float32),
    tf.random.normal((1, N, fft_length // 2 + 1), dtype=tf.dtypes.float32))    
test_output = model(test_input)
tf.get_logger().warning("Tensorflow test finished...")

tf.get_logger().warning("Begin TFLite conversion...")
model.save('./tflite')
converter = tf.lite.TFLiteConverter.from_saved_model('./tflite')
tflite_model = converter.convert()
with open(os.path.join('./tflite', 'model.tflite'), "wb") as f:
    f.write(tflite_model)
tf.get_logger().warning("TFLite conversion finished...") 

