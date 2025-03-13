import tensorflow as tf
import threading

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

@tf.function
def double(x):
  return x * 2.0

def test():
  input = tf.range(5, dtype=tf.float32)
  strategy.run(double, args=(input,))

test()
print("TPUStrategy.run works on primary thread")

thread = threading.Thread(target=test)
thread.start()
thread.join()

