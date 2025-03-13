# gelu_problem.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
 

def gelu_approximate(x):
    # Copied and inlined from tf.nn.gelu(approximate=True) which exists in TF v2.4 but not TF v2.2
    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + 0.044715 * tf.pow(x, 3))))


def gelu_gradient(x, device):
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.device(device):
            y = gelu_approximate(x)
    return tape.gradient(y, x)
 
 
def main():
    print(f"TF version is {tf.__version__}")
    x = tf.linspace(-500.0, 500.0, 500)
    cpu = gelu_gradient(x, "/CPU:0")
    gpu = gelu_gradient(x, "/GPU:0")
    abs_error = np.abs(cpu - gpu)
    df = pd.DataFrame(dict(cpu=cpu, gpu=gpu, abs_error=abs_error), index=x)
    try:
        np.testing.assert_allclose(cpu, gpu, atol=1e-3)
    except (AssertionError,) as e:
        print("GPU and CPU gradients are not close!")
        print(e)
    df.plot(title="CPU vs. GPU gradients for tf.nn.gelu(approximate=True)")
    plt.show()
 
 
if __name__ == "__main__":
    main()

