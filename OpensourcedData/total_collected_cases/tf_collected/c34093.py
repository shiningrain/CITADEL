
Without @tf.function, everything is fine. 

**Describe the expected behavior**
The gradients should be calculated well regardless of how many times we call the layer within a @tf.function.

**Code to reproduce the issue**
python
import tensorflow as tf


ft_col_numeric = tf.feature_column.numeric_column("test_input")
ft_col_buk     = tf.feature_column.bucketized_column(ft_col_numeric, boundaries=[1, 3, 5, 7])
ft_col_embed   = tf.feature_column.embedding_column(ft_col_buk, dimension=4)
ft_embed_layer = tf.keras.layers.DenseFeatures(ft_col_embed)

# crash when the call number is greater than 4
LAYER_CALL_NUM = 5

@tf.function
def run(inputs):
    with tf.GradientTape() as tape:
        res_list = []
        for i in range(LAYER_CALL_NUM):
            x = ft_embed_layer(inputs)
            res_list.append(x)
        y = tf.reduce_sum(sum(res_list))
    weights = ft_embed_layer.trainable_variables
    gradients = tape.gradient(y, weights)
    return gradients

test_input = tf.constant([0, 2, 4, 6, 8])
inputs = { "test_input" : test_input }
run(inputs)

