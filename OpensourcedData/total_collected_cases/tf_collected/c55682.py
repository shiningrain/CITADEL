TL;DR if a function contains a softmax after a numerically masked input (i.e. after adding a very large negative penalty), and that function is compiled with `tf.function(jit_compile=True)`, its CPU output is very different from its non compiled counterpart. On GPU, the two outputs are very close.

Long description:
On NLP models, like `T5`, it is common to have multiple inputs in the same batch with different lengths. To handle them without `RaggedTensors`, we pad the shortest entries in the batch to the longest length, and send the padded tokens along an attention mask (containing 0s where padding happened, 1s otherwise). Internally, in the model, we convert this binary attention mask to a numerical one, containing a large penalty (e.g. `-1e9` in Hugging Face's `T5`) that will be added to the input of the attention layers' softmax, such that no attention is given to padding. As such, this numerically masked softmax is a common operation in large language models.

As part of our efforts to speed up text generation at Hugging Face, we struggled to reproduce forward passes of some models when they were compiled with `tf.function(jit_compile=True)` (https://github.com/huggingface/transformers/issues/16838). Upon further inspection, we noticed that:
- The problematic behavior was CPU-only;
- The issue came from the softmax operation, but only when masking was present;
- It is present even when the attention mask contains all 1s, meaning that the large negative penalty doesn't get applied at all to the inputs;
- Reducing the large penalty to a not so large penalty, like `-100`, still results in noticable mismatches.

The snippet below gives a simple example where the problem described above can be seen -- it passes on GPU but fails on CPU. Because it is easily reproducible, I'm not including the XLA compilation files (as suggested [here](https://www.tensorflow.org/xla#reproducible_bug_reports)), but feel free to request them :)



### Standlone code to reproduce the issue

shell
import tensorflow as tf


# same outcome for values <= -1e3
LARGE_PENALTY = -1e9


def simple_softmax(x):
    return tf.nn.softmax(x)


def masked_softmax(x, boolean_mask):
    numerical_mask = (1. - tf.cast(boolean_mask, dtype=tf.float32)) * LARGE_PENALTY
    masked_x = x + numerical_mask
    return tf.nn.softmax(masked_x)


xla_masked_softmax = tf.function(masked_softmax, jit_compile=True)
xla_simple_softmax = tf.function(simple_softmax, jit_compile=True)
x = tf.random.normal((1, 10))

# same outcome regardless of the boolean mask here
boolean_mask = tf.convert_to_tensor([[1] * 9 + [0] * 1], dtype=tf.int32)

# masks input outside of the compiled softmax -> works correctly on CPU and GPU
numerical_mask = (1. - tf.cast(boolean_mask, dtype=tf.float32)) * LARGE_PENALTY
masked_x = x + numerical_mask
xla_out = xla_simple_softmax(masked_x)
out = simple_softmax(masked_x)
print(tf.math.reduce_max(tf.math.abs(xla_out - out)).numpy())
assert tf.experimental.numpy.allclose(xla_out, out)

# masked_softmax -> fails regardless of the mask on CPU, works correctly on GPU
xla_out = xla_masked_softmax(x, boolean_mask)
out = masked_softmax(x, boolean_mask)
print(tf.math.reduce_max(tf.math.abs(xla_out - out)).numpy())
assert tf.experimental.numpy.allclose(xla_out, out)

