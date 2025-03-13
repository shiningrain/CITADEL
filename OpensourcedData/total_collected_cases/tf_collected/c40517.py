import time
import tensorflow as tf

@tf.function
def odeint_external(tensor):
    finished_user_t_ii = 0
    x = tensor[0]
    t = tensor[1]
    
    # array to store the result
    result = tf.TensorArray(dtype=tf.float32, size=t.shape[0])
    result = result.write(0, x)

    i = tf.constant(0, dtype=tf.int32)
    for _t in t[1:]:
        i = i+1
        result = result.write(i, tf.stack([tf.cos(_t), tf.sin(_t)]))

    return result.stack(), t


@tf.function
def parallelized_map_fn(tensor):
    return tf.map_fn(odeint_external, tensor)


# warm up
result = parallelized_map_fn((tf.random.normal((50, 2), 0, 1), tf.random.normal((50, 10000), 0, 1)))

start_t = time.time()
result = parallelized_map_fn((tf.random.normal((50, 2), 0, 1), tf.random.normal((50, 10000), 0, 1)))
print(time.time()-start_t)


Code 2:
python3
import time
import tensorflow as tf

@tf.function
def odeint_external(tensor):
    finished_user_t_ii = 0
    x = tensor[0]
    t = tensor[1]
    
    # array to store the result
    result = tf.TensorArray(dtype=tf.float32, size=t.shape[0])
    result = result.write(0, x)

    i = tf.constant(1, dtype=tf.int32)
    cond = lambda i, _t, r: i < 9999
    body = lambda i, _t, r: (i+1, _t, r.write(i, tf.stack([tf.cos(_t[i]), tf.sin(_t[i])])))
    result = tf.while_loop(cond=cond, body=body, loop_vars=(i, t[1:], result))
    
    return result[2].stack(), t

@tf.function
def parallelized_map_fn(tensor):
    return tf.map_fn(odeint_external, tensor, parallel_iterations=1)

# warm up
result = parallelized_map_fn((tf.random.normal((50, 2), 0, 1), tf.random.normal((50, 10000), 0, 1)))

start_t = time.time()
result = parallelized_map_fn((tf.random.normal((50, 2), 0, 1), tf.random.normal((50, 10000), 0, 1)))
print(time.time()-start_t)


Code 3:
python
import time
import tensorflow as tf

@tf.function
def odeint_external(tensor):
    finished_user_t_ii = 0
    x = tensor[0]
    t = tensor[1]
    
    # array to store the result
    result = tf.TensorArray(dtype=tf.float32, size=t.shape[0])
    result = result.write(0, x)

    i = tf.constant(0, dtype=tf.int32)
    for _t in t[1:]:
        i = i+1
        result = result.write(i, tf.stack([tf.cos(_t), tf.sin(_t)]))
    
    return result.stack(), t

x = tf.random.normal((50, 2), 0, 1)
t = tf.random.normal((50, 10000), 0, 1)

# warm up
result = [odeint_external((_x, _t)) for _x, _t in zip(x, t)]

start_t = time.time()
result = [odeint_external((_x, _t)) for _x, _t in zip(x, t)]
print(time.time()-start_t)

