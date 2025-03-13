Empty arguments



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_sparse_ops
try:
  indices = []
  values = []
  shapes = []
  dense_inputs = []
  num_buckets = -461
  salt = []
  strong_hash = False
  out = gen_sparse_ops.sparse_cross_hashed(indices=indices,values=values,shapes=shapes,dense_inputs=dense_inputs,num_buckets=num_buckets,salt=salt,strong_hash=strong_hash,)
except Exception as e:
  print("Error:"+str(e))

