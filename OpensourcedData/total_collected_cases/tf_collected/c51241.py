import tensorflow as tf

def func():
    return tf.reshape([[42]], 1)


func_jit = tf.function(func=func)

func()  # works
func_jit()  # fails

