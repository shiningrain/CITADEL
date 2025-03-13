import tensorflow as tf

array = [
    tf.random.uniform((19,1,3,2)),
    tf.random.uniform((19,1,2,2)),
    tf.random.uniform((3,1,3,2)),
    tf.random.uniform((18,1,2,2)),
]

size = len(array)

@tf.function(experimental_compile=True) # The same happens with autograph=False
def add(array):
    
    tensor_array = tf.TensorArray(
        dtype=tf.float32,
        size=size,
        infer_shape=False,
        element_shape=tf.TensorShape([None, 1, None, 2]),
    )
    for i in range(size):
        tensor_array = tensor_array.write(i, array[i])
    # There would be a while_loop here

r = add(array)

