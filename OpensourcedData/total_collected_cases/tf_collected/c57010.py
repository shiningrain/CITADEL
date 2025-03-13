import numpy as np
import tensorflow as tf


def f(dtype, disable_meta_optimizer: bool):
  tf.config.optimizer.set_experimental_options(
    options={
      "disable_meta_optimizer": disable_meta_optimizer,
      "min_graph_nodes": 1, # the graph will only consist of a single node, default is 4
    }
  )
  print(f"dtype={dtype} disable_meta_optimizer={disable_meta_optimizer}")
  with tf.Graph().as_default() as g, tf.compat.v1.Session(graph=g) as s:
    t = tf.constant(1.0, dtype=dtype)
    try:
      fetch = s.run(t)
      assert fetch.astype(np.float32) == np.full([], 1.0, np.float32)
    except AssertionError:
      print(f"Fail! Contents of fetched tensor: {fetch}")
    else:
      print(f"Success!")


for dtype in [tf.float32, tf.float16]:
  for disable_meta_optimizer in [True, False]:
    f(dtype, disable_meta_optimizer)

