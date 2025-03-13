TypeError: '<' not supported between instances of 'FuncGraph' and '_WeakReferencableClass'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.7/dist-packages/six.py in raise_from(value, from_value)

ValueError: Error processing property '_dropout_mask_cache' of <ContextValueCache at 0x7fa31138ab10>


**Describe the expected behavior**
the variable method can safely be called after performing a forward pass with autograph and dropout.

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.

python
import tensorflow

class GRU(tf.Module):
  def __init__(self):
    super(GRU, self).__init__()
    self.cell = tf.keras.layers.GRUCell(units=2, dropout=0.1, recurrent_dropout=0.1)

  @tf.function
  def bad_infer(self, inputs, states):
    o, h = self.cell(inputs=inputs, states=states, training=True)
    return o, h

  def good_infer(self, inputs, states):
    o, h = self.cell(inputs=inputs, states=states, training=True)
    return o, h

gru = GRU()
inputs = tf.ones((1, 2))
states = gru.cell.get_initial_state(inputs)
targets = tf.ones(1, 2)

gru.good_infer(inputs=inputs, states=states)
gru.variables

gru.bad_infer(inputs=inputs, states=states)
gru.variables

