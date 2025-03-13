import tensorflow as tf
from functools import partial

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)


def split_head(input, B, N, D, num_heads):
    assert D/num_heads == D//num_heads, "D must be divisible by num_heads"
   
    x = tf.reshape(input, 
               (B, N, num_heads, D//num_heads))
    return tf.transpose(x, perm = (0, 2, 1, 3))
       
B = 2048
N = 1024
D = 1024
H = 32
b_k = 15
input = tf.ones((B, N, D))

with strategy.scope():
      
 #tf.reshape(tf.range(B*N*D), (B,N,D))
  q = input
  k = input
  split_head = partial(split_head, B= B, N=N, D= D)
  q = split_head(q, num_heads= H)
  k = split_head(k, num_heads= H)
  print(B,N,D)
  k_T = tf.transpose(k, perm =(0, 1, 3, 2))
  print(k_T.shape)
  band_k = extract_band(input = k, b_k = b_k)
  print(band_k.shape)
  if N >= D//H:
      attention = q@tf.transpose(band_k, perm = (0, 1, 3, 2))
      print(attention.shape)
      attention = tf.reduce_mean(attention, axis = 1) # combine head
      attention_map = tf.linalg.diag(tf.transpose(attention, perm = (0, 2, 1) ), k = (-b_k//2, b_k//2))

