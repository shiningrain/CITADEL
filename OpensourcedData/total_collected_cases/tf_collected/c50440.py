from datetime import datetime
import tensorflow as tf

b = tf.random.uniform([32,28,28,1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

def test(trace_name):
    logs = f"logs/{trace_name}" + datetime.now().strftime("%Y%m%d-%H%M%S")

    epoch = 0
    tf.profiler.experimental.start(logs)
    for step in range(100):
        if 1 < step < 100:
            with tf.profiler.experimental.Trace(trace_name):
                model(b)
        else:
            model(b)
    tf.profiler.experimental.stop()

test("TraceContext") # works fine
test("Broken") # no step marker observed error

