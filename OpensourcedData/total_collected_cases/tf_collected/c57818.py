import tensorflow as tf

def get_tflite_callable(model, inp_dict):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        funcs=[model.__call__.get_concrete_function(**inp_dict)],
        trackable_obj=model,
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_bytes = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    runner = interpreter.get_signature_runner()
    return runner

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.const = tf.constant([-10.43154963850975037], dtype=tf.float64)

    @tf.function
    def __call__(self, x):
        c = tf.raw_ops.LeakyRelu(
            features=self.const, alpha=0.1,
        )
        x = tf.add(c, x)
        return x


inp = {
    "x": tf.constant([3.12363398], dtype=tf.float64),
}
m = MyModule()

out = m(**inp)
print(f'{out}') # t = <tf.Tensor: shape=(1,), dtype=float64, numpy=array([2.080479])>

runner = get_tflite_callable(m, inp)
out = runner(**inp)['output_0']
print(f'{out}\n{out.dtype}') # out = array([3.12363398])  out.dtype = dtype('float64')


